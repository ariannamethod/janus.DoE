/*
 * m.c — Democracy of Experts. the 4th element. the unhinged one.
 *
 * experts are born. experts die. the parliament votes on every token.
 * the tokenizer knows it's a tokenizer. the data parser judges your data.
 * the architecture recompiles itself per-input like a nervous breakdown
 * that somehow produces valid gradients.
 *
 * parameters persist. topology doesn't. each forward pass, the model decides:
 * - how many experts are alive right now (2..16)
 * - how many get to vote per token (variable k, not fixed)
 * - how deep to go (skip layers when input is boring)
 * the resonance-harmonic backbone holds it all together. barely.
 *
 * cc m.c -O3 -lm -lpthread -o m && ./m --depth 4
 *
 * depth is the only knob. turn it and watch democracy scale.
 *
 *   depth 2  → ~1.8M params, 4 experts, learns what committees are
 *   depth 4  → ~8.0M params, 4 experts, experts start specializing
 *   depth 6  → ~31M params, 6 experts, coalitions emerge
 *   depth 8  → ~67M params, 6 experts, political parties form
 *   depth 10 → ~165M params, 8 experts, full political spectrum
 *   depth 12 → ~283M params, 8 experts, parliamentary democracy
 *
 * what happens when you run it:
 * 1. loads or generates data (HF API / parquet / synthetic shame)
 * 2. trains BPE tokenizer that knows its own compression ratio
 * 3. builds ephemeral MoE with living experts
 * 4. trains with hand-written gradients through variable-k parliament
 * 5. watches experts be born (mitosis) and die (apoptosis)
 * 6. finetunes on personality.txt (optional but recommended)
 * 7. exports GGUF, drops you into chat with a parliament
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#ifdef __linux__
  #include <sys/statvfs.h>
#endif
#ifdef __APPLE__
  #include <sys/param.h>
  #include <sys/mount.h>
  #include <sys/sysctl.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * BLAS ACCELERATION — optional. 3-4x speedup on matmul.
 *   macOS:  cc m.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o m
 *   Linux:  cc m.c -O3 -lm -lpthread -DUSE_BLAS -lopenblas -o m
 *   GPU:    cc m.c -O3 -lm -lpthread -DUSE_CUBLAS -lcublas -lcudart -o m
 * ═══════════════════════════════════════════════════════════════════════════════ */
#ifdef USE_CUBLAS
  #include <cublas_v2.h>
  #include <cuda_runtime.h>
  static cublasHandle_t g_cublas;
  static int cublas_inited = 0;
  /* scratch buffers — grow-only, reused */
  static float *d_scratch[4] = {NULL,NULL,NULL,NULL};
  static size_t d_scratch_sz[4] = {0,0,0,0};
  static void cublas_init(void) {
      if (!cublas_inited) {
          cublasCreate(&g_cublas);
          /* TF32 on A100/H100: ~8x faster than FP32, negligible accuracy loss */
          cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
          struct cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
          printf("[gpu] %s — %.0f MB, compute %d.%d, TF32 enabled\n",
                 prop.name, (double)prop.totalGlobalMem/1e6, prop.major, prop.minor);
          cublas_inited = 1;
      }
  }
  static float* gpu_scratch(int slot, size_t bytes) {
      if (bytes > d_scratch_sz[slot]) {
          if (d_scratch[slot]) cudaFree(d_scratch[slot]);
          cudaMalloc((void**)&d_scratch[slot], bytes);
          d_scratch_sz[slot] = bytes;
      }
      return d_scratch[slot];
  }
  /* GPU weight cache — upload tensor to GPU once, skip on subsequent calls */
  static float* gpu_ensure(float *data, int n, float **d_ptr, int *dirty) {
      if (!*d_ptr) {
          cudaMalloc((void**)d_ptr, (size_t)n * 4);
          cudaMemcpy(*d_ptr, data, (size_t)n * 4, cudaMemcpyHostToDevice);
          *dirty = 0;
      } else if (*dirty) {
          cudaMemcpy(*d_ptr, data, (size_t)n * 4, cudaMemcpyHostToDevice);
          *dirty = 0;
      }
      return *d_ptr;
  }
#elif defined(USE_BLAS)
  #ifdef ACCELERATE
    #define ACCELERATE_NEW_LAPACK
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION — one knob. depth. everything else is derived.
 * pytorch has 47 config files. we have one integer and a political system.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MAX_EXPERTS 16
#define MAX_LAYERS 32
#define MIN_EXPERTS 2
#define CHUCK_WINDOW 16
#define CHUCK_DAMP_LO 0.5f
#define CHUCK_DAMP_HI 2.0f
#define CHUCK_B1 0.9f
#define CHUCK_B2 0.999f
#define CHUCK_EPS 1e-8f
#define CHUCK_PSI_CAP 0.3f
#define CHUCK_PSI_HALF 100.0f
#define CHUCK_MACRO_INT 500
#define CHUCK_MACRO_PAT 3
#define CHUCK_MACRO_DECAY 0.7f
#define CHUCK_REC_CD 20
#define CHUCK_REC_THR 0.25f
#define CHUCK_MEM_CAP 256
#define NOTORCH_RANK 4
#define HARMONIC_N 8

typedef struct {
    int depth, dim, n_heads, n_kv_heads, head_dim, hidden_dim;
    int vocab_size, seq_len;
    float norm_eps, rope_theta;
    int max_experts, initial_experts;
    float attn_clamp, aux_loss_w;
    float lr, weight_decay;
    int batch_size, max_steps, warmup_steps, log_every, bpe_merges, personality_steps, hf_pages;
    int explicit_steps;  /* 1 if --steps was given on CLI */
    int explicit_data;   /* 1 if --data was given on CLI */
    int active_depth;    /* layers executed per forward (ephemeral; <= depth) */
    float vitality_threshold;  /* ephemeral low-vitality cutoff for apoptosis counting */
    char data_url[512], data_path[256], gguf_path[256], personality_path[256];
} Config;

static Config config_from_depth(int depth) {
    Config c = {0};
    c.depth = depth;
    c.dim = depth * 64;
    c.dim = ((c.dim + 63) / 64) * 64;
    if (c.dim < 128) c.dim = 128;
    if (c.dim > 768) c.dim = 768;
    c.head_dim = 64;
    c.n_heads = c.dim / c.head_dim;
    if (c.n_heads < 1) c.n_heads = 1;
    if (c.dim <= 384) { c.n_kv_heads = c.n_heads; }
    else { c.n_kv_heads = c.n_heads / 2; if (c.n_kv_heads < 1) c.n_kv_heads = 1;
           while (c.n_heads % c.n_kv_heads != 0 && c.n_kv_heads > 1) c.n_kv_heads--; }
    c.max_experts = MAX_EXPERTS;
    c.initial_experts = (depth <= 4) ? 4 : (depth <= 8) ? 6 : 8; /* sane: d2→4, d4→4, d8→6, d12→8 */
    if (c.initial_experts > MAX_EXPERTS) c.initial_experts = MAX_EXPERTS;
    c.hidden_dim = (int)(c.dim * 1.5f);
    c.hidden_dim = ((c.hidden_dim + 63) / 64) * 64;
    c.seq_len = 256; c.norm_eps = 1e-5f; c.rope_theta = 10000.0f;
    c.attn_clamp = 30.0f; c.aux_loss_w = 0.01f;
    c.lr = 1e-4f; c.batch_size = 4; c.warmup_steps = 500;
    c.weight_decay = 0.01f; c.log_every = 20;
    /* max_steps computed later from actual token count — see post-tokenization */
    c.max_steps = 0; /* 0 = auto from data */
    c.bpe_merges = depth <= 4 ? 2000 : depth <= 8 ? 4000 : 8000; /* auto from depth (README); --bpe-merges overrides */
    c.active_depth = depth; c.vitality_threshold = 0.1f; c.personality_steps = 0; /* 0 = auto from personality size */
    c.explicit_steps = 0; c.explicit_data = 0;
    /* Auto HF pages from depth: need ~30 tok/param, ~0.5MB per page, tok≈bytes/3 */
    c.hf_pages = (depth <= 2) ? 120 : (depth <= 4) ? 500 : (depth <= 8) ? 4000 : 10000;
    snprintf(c.data_url, 512, "HuggingFaceFW/fineweb-edu");
    snprintf(c.data_path, 256, "m_data.txt");
    snprintf(c.gguf_path, 256, "m.gguf");
    snprintf(c.personality_path, 256, "personality.txt");
    return c;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * RNG — xorshift64*. the experts don't care which PRNG decided their fate.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static uint64_t rng_state = 42;
static uint64_t rng_next(void) { rng_state ^= rng_state<<13; rng_state ^= rng_state>>7; rng_state ^= rng_state<<17; return rng_state; }
static float rand_uniform(void) { return (float)(rng_next()&0x7FFFFFFF)/(float)0x7FFFFFFF; }
static float rand_normal(void) { float u1=rand_uniform(),u2=rand_uniform(); if(u1<1e-10f)u1=1e-10f; return sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2); }

/* ═══════════════════════════════════════════════════════════════════════════════
 * BPE TOKENIZER + SELF-AWARENESS EYE
 * the tokenizer that knows it's a tokenizer. tracks its own compression ratio,
 * OOV rate, entropy. feeds health signal into Chuck optimizer.
 * "i am become tokenizer, measurer of bytes."
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { char **items; int len, cap; } StrArr;
static void sa_push(StrArr *a, const char *s) { if(a->len>=a->cap){a->cap=a->cap?a->cap*2:16;a->items=realloc(a->items,sizeof(char*)*a->cap);}a->items[a->len++]=strdup(s); }
static void sa_free(StrArr *a) { for(int i=0;i<a->len;i++)free(a->items[i]);free(a->items);a->items=NULL;a->len=a->cap=0; }

#define TOK_MAX_VOCAB 16384
#define TOK_STOI_CAP 32768
typedef struct { char a[64]; char b[64]; } MergePair;
typedef struct { char *key; int val; } StoiEntry;
typedef struct { StoiEntry entries[TOK_STOI_CAP]; } StoiTable;
typedef struct { char *tokens[TOK_MAX_VOCAB]; int vocab_size; StoiTable stoi; int bos_id,eos_id; MergePair *merges; int n_merges; } Tokenizer;

/* Self-awareness eye for the tokenizer — it knows what it eats */
typedef struct {
    float compression_ratio; /* bytes_in / tokens_out, EMA */
    float oov_rate;          /* unknown token frequency, EMA */
    float entropy;           /* token distribution entropy */
    int total_encoded;       /* lifetime token count */
    float health;            /* composite signal for Chuck */
    float code_ratio;        /* % of input that looks like code */
    int code_mode;           /* 0=text, 1=code detected */
} TokenizerEye;

static unsigned int str_hash(const char *s){unsigned int h=5381;while(*s)h=h*33+(unsigned char)*s++;return h;}
static void stoi_init(StoiTable *t){for(int i=0;i<TOK_STOI_CAP;i++){t->entries[i].key=NULL;t->entries[i].val=-1;}}
static void stoi_put(StoiTable *t,const char *key,int val){unsigned int h=str_hash(key)%TOK_STOI_CAP;for(int i=0;i<TOK_STOI_CAP;i++){int idx=(h+i)%TOK_STOI_CAP;if(!t->entries[idx].key){t->entries[idx].key=strdup(key);t->entries[idx].val=val;return;}if(strcmp(t->entries[idx].key,key)==0){t->entries[idx].val=val;return;}}}
static int stoi_get(StoiTable *t,const char *key){unsigned int h=str_hash(key)%TOK_STOI_CAP;for(int i=0;i<TOK_STOI_CAP;i++){int idx=(h+i)%TOK_STOI_CAP;if(!t->entries[idx].key)return -1;if(strcmp(t->entries[idx].key,key)==0)return t->entries[idx].val;}return -1;}

static void tok_init(Tokenizer *tok){memset(tok,0,sizeof(Tokenizer));stoi_init(&tok->stoi);for(int i=0;i<256;i++){char h[8];snprintf(h,8,"0x%02x",i);tok->tokens[tok->vocab_size]=strdup(h);stoi_put(&tok->stoi,h,tok->vocab_size);tok->vocab_size++;}tok->tokens[tok->vocab_size]=strdup("<BOS>");stoi_put(&tok->stoi,"<BOS>",tok->vocab_size);tok->bos_id=tok->vocab_size++;tok->tokens[tok->vocab_size]=strdup("<EOS>");stoi_put(&tok->stoi,"<EOS>",tok->vocab_size);tok->eos_id=tok->vocab_size++;}
static void tok_add(Tokenizer *tok,const char *s){if(stoi_get(&tok->stoi,s)>=0)return;if(tok->vocab_size>=TOK_MAX_VOCAB)return;tok->tokens[tok->vocab_size]=strdup(s);stoi_put(&tok->stoi,s,tok->vocab_size);tok->vocab_size++;}

static char byte_category(unsigned char b){if((b>='a'&&b<='z')||(b>='A'&&b<='Z'))return'L';if(b>='0'&&b<='9')return'N';if(b==' '||b=='\n'||b=='\r'||b=='\t')return'Z';if(b>=0x80)return'L';return'P';}
typedef struct{unsigned char*data;int len;}ByteSeg;
typedef struct{ByteSeg*segs;int len,cap;}SegArr;
static void seg_push(SegArr*a,unsigned char*data,int len){if(a->len>=a->cap){a->cap=a->cap?a->cap*2:64;a->segs=realloc(a->segs,sizeof(ByteSeg)*a->cap);}a->segs[a->len].data=malloc(len);memcpy(a->segs[a->len].data,data,len);a->segs[a->len].len=len;a->len++;}
static void seg_free(SegArr*a){for(int i=0;i<a->len;i++)free(a->segs[i].data);free(a->segs);memset(a,0,sizeof(SegArr));}

static SegArr unicode_segment(const char*text,int text_len){SegArr r={0};if(!text||text_len==0)return r;unsigned char buf[4096];int bl=0;char cc=0;const unsigned char*p=(const unsigned char*)text;for(int i=0;i<text_len;i++){char cat=byte_category(p[i]);if(cat!=cc&&bl>0){seg_push(&r,buf,bl);bl=0;}cc=cat;if(bl<(int)sizeof(buf)-1)buf[bl++]=p[i];else{seg_push(&r,buf,bl);bl=0;buf[bl++]=p[i];}}if(bl>0)seg_push(&r,buf,bl);return r;}

#define PAIR_CAP 131072
typedef struct{char a[64];char b[64];int count;int used;}PairEntry;
static unsigned int pair_hash(const char*a,const char*b){unsigned int h=5381;for(const char*p=a;*p;p++)h=h*33+(unsigned char)*p;h=h*33+0xFF;for(const char*p=b;*p;p++)h=h*33+(unsigned char)*p;return h;}

static void tok_train_bpe(Tokenizer*tok,const char*text,int tl,int nm){
    printf("[bpe] training %d merges on %d bytes...\n",nm,tl);
    SegArr segs=unicode_segment(text,tl);if(segs.len==0){seg_free(&segs);return;}
    int ns=segs.len;StrArr*ss=calloc(ns,sizeof(StrArr));
    for(int s=0;s<ns;s++)for(int b=0;b<segs.segs[s].len;b++){char h[8];snprintf(h,8,"0x%02x",segs.segs[s].data[b]);sa_push(&ss[s],h);}
    seg_free(&segs);if(tok->merges)free(tok->merges);tok->merges=calloc(nm,sizeof(MergePair));tok->n_merges=0;
    PairEntry*pairs=calloc(PAIR_CAP,sizeof(PairEntry));
    for(int it=0;it<nm;it++){
        memset(pairs,0,sizeof(PairEntry)*PAIR_CAP);
        for(int s=0;s<ns;s++){StrArr*sq=&ss[s];for(int i=0;i<sq->len-1;i++){unsigned int h=pair_hash(sq->items[i],sq->items[i+1])%PAIR_CAP;for(int p=0;p<64;p++){int idx=(h+p)%PAIR_CAP;if(!pairs[idx].used){strncpy(pairs[idx].a,sq->items[i],63);strncpy(pairs[idx].b,sq->items[i+1],63);pairs[idx].count=1;pairs[idx].used=1;break;}if(strcmp(pairs[idx].a,sq->items[i])==0&&strcmp(pairs[idx].b,sq->items[i+1])==0){pairs[idx].count++;break;}}}}
        int bc=1,bi=-1;for(int i=0;i<PAIR_CAP;i++)if(pairs[i].used&&pairs[i].count>bc){bc=pairs[i].count;bi=i;}
        if(bi<0)break;
        char nt[128];snprintf(nt,128,"%s+%s",pairs[bi].a,pairs[bi].b);
        strncpy(tok->merges[tok->n_merges].a,pairs[bi].a,63);strncpy(tok->merges[tok->n_merges].b,pairs[bi].b,63);tok->n_merges++;
        for(int s=0;s<ns;s++){StrArr*sq=&ss[s];StrArr mg={0};int i=0;while(i<sq->len){if(i<sq->len-1&&strcmp(sq->items[i],pairs[bi].a)==0&&strcmp(sq->items[i+1],pairs[bi].b)==0){sa_push(&mg,nt);i+=2;}else{sa_push(&mg,sq->items[i]);i++;}}sa_free(sq);*sq=mg;}
        tok_add(tok,nt);
        if((it+1)%500==0)printf("[bpe] %d/%d merges (vocab=%d)\n",it+1,nm,tok->vocab_size);
    }
    free(pairs);for(int s=0;s<ns;s++)sa_free(&ss[s]);free(ss);
    printf("[bpe] done: %d merges, vocab=%d\n",tok->n_merges,tok->vocab_size);
}

static void tok_save_merges(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "%d\n", tok->n_merges);
    for (int i = 0; i < tok->n_merges; i++)
        fprintf(f, "%s\t%s\n", tok->merges[i].a, tok->merges[i].b);
    fclose(f);
    printf("[bpe] saved %d merges to %s\n", tok->n_merges, path);
}

static int tok_load_merges(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int nm = 0;
    if (fscanf(f, "%d\n", &nm) != 1 || nm <= 0) { fclose(f); return 0; }
    if (nm > TOK_MAX_VOCAB) nm = TOK_MAX_VOCAB;   /* bound merges to vocab capacity */
    if (tok->merges) free(tok->merges);
    tok->merges = calloc(nm, sizeof(MergePair));
    if (!tok->merges) { fclose(f); return 0; }
    tok->n_merges = 0;
    for (int i = 0; i < nm; i++) {
        char a[64], b[64];
        if (fscanf(f, "%63s\t%63s\n", a, b) != 2) break;
        if (tok->vocab_size >= TOK_MAX_VOCAB) break;   /* vocab full — stop adding merges */
        strncpy(tok->merges[i].a, a, 63);
        strncpy(tok->merges[i].b, b, 63);
        /* Rebuild vocab: add merged token */
        char nt[128]; snprintf(nt, 128, "%s+%s", a, b);
        if (stoi_get(&tok->stoi, nt) < 0) {
            tok->tokens[tok->vocab_size] = strdup(nt);
            stoi_put(&tok->stoi, nt, tok->vocab_size);
            tok->vocab_size++;
        }
        tok->n_merges++;
    }
    fclose(f);
    printf("[bpe] loaded %d merges from %s (vocab=%d)\n", tok->n_merges, path, tok->vocab_size);
    return tok->n_merges > 0;
}

/* ═══ TOKENIZER CACHE — the tokenizer remembers what it ate (encode memoization) ═══ */
#define TOKCACHE_CAP 1024
#define TOKCACHE_MAX_LEN 128
typedef struct { uint64_t hash; char *text; int *ids; int n_ids; int text_len; int hits; int alive; } TokCacheEntry;
typedef struct { TokCacheEntry entries[TOKCACHE_CAP]; int total_hits; int total_misses; float hit_rate; int vocab_size; } TokCache;
static void tokcache_init(TokCache *tc, int vocab_size) {
    memset(tc, 0, sizeof(TokCache));
    tc->vocab_size = vocab_size;
}
static uint64_t tokcache_hash(const char *text, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) { h ^= (uint64_t)(unsigned char)text[i]; h *= 1099511628211ULL; }
    return h;
}
static int *tokcache_lookup(TokCache *tc, const char *text, int len, int *n_ids) {
    if (len > TOKCACHE_MAX_LEN || len == 0) return NULL;
    uint64_t h = tokcache_hash(text, len);
    int idx = (int)(h % TOKCACHE_CAP);
    for (int probe = 0; probe < 8; probe++) {
        int i = (idx + probe) % TOKCACHE_CAP;
        if (tc->entries[i].alive && tc->entries[i].hash == h && tc->entries[i].text_len == len && memcmp(tc->entries[i].text, text, len) == 0) {
            tc->entries[i].hits++; tc->total_hits++;
            tc->hit_rate = 0.95f * tc->hit_rate + 0.05f;
            *n_ids = tc->entries[i].n_ids; return tc->entries[i].ids;
        }
    }
    tc->total_misses++; tc->hit_rate = 0.95f * tc->hit_rate; return NULL;
}
static void tokcache_insert(TokCache *tc, const char *text, int len, int *ids, int n_ids) {
    if (len > TOKCACHE_MAX_LEN || len == 0 || n_ids <= 0) return;
    uint64_t h = tokcache_hash(text, len);
    int idx = (int)(h % TOKCACHE_CAP);
    int target = idx; int min_hits = tc->entries[idx].alive ? tc->entries[idx].hits : -1;
    for (int probe = 0; probe < 8; probe++) {
        int i = (idx + probe) % TOKCACHE_CAP;
        if (!tc->entries[i].alive) { target = i; break; }
        if (tc->entries[i].hits < min_hits) { min_hits = tc->entries[i].hits; target = i; }
    }
    TokCacheEntry *e = &tc->entries[target];
    if (e->alive) { free(e->ids); free(e->text); }
    e->hash = h; e->text_len = len; e->hits = 0; e->alive = 1; e->n_ids = n_ids;
    e->ids = malloc(n_ids * sizeof(int)); memcpy(e->ids, ids, n_ids * sizeof(int));
    e->text = malloc(len); memcpy(e->text, text, len);   /* keep the key to reject hash collisions */
}
static TokCache g_tokcache; static int g_tokcache_ready = 0;

static int*tok_encode(Tokenizer*tok,const char*text,int tl,int*out_len){
    if (!g_tokcache_ready) { tokcache_init(&g_tokcache, tok->vocab_size); g_tokcache_ready = 1; }
    { int hn; int *h = tokcache_lookup(&g_tokcache, text, tl, &hn);
      if (h) { int *cp = malloc(hn*sizeof(int)); memcpy(cp, h, hn*sizeof(int)); *out_len = hn; return cp; } }
    SegArr segs=unicode_segment(text,tl);int*ids=NULL;int ni=0,ci=0;
    for(int s=0;s<segs.len;s++){StrArr sy={0};for(int b=0;b<segs.segs[s].len;b++){char h[8];snprintf(h,8,"0x%02x",segs.segs[s].data[b]);sa_push(&sy,h);}
    if(tok->n_merges>0&&sy.len>=2){int ch=1;while(ch&&sy.len>=2){ch=0;int br=tok->n_merges,bp=-1;for(int i=0;i<sy.len-1;i++)for(int m=0;m<br;m++)if(strcmp(sy.items[i],tok->merges[m].a)==0&&strcmp(sy.items[i+1],tok->merges[m].b)==0){br=m;bp=i;break;}if(bp>=0){char nt[128];snprintf(nt,128,"%s+%s",tok->merges[br].a,tok->merges[br].b);StrArr mg={0};for(int i=0;i<sy.len;i++){if(i==bp){sa_push(&mg,nt);i++;}else sa_push(&mg,sy.items[i]);}sa_free(&sy);sy=mg;ch=1;}}}
    for(int i=0;i<sy.len;i++){int id=stoi_get(&tok->stoi,sy.items[i]);if(id<0)id=0;if(ni>=ci){ci=ci?ci*2:256;ids=realloc(ids,sizeof(int)*ci);}ids[ni++]=id;}sa_free(&sy);}
    tokcache_insert(&g_tokcache, text, tl, ids, ni);
    seg_free(&segs);*out_len=ni;return ids;
}

static char*tok_decode(Tokenizer*tok,int*ids,int ni,int*out_len){
    /* first pass: count decoded bytes (one per "0xNN" group) — a merge token can decode
     * to far more than 8 bytes, so size the buffer exactly instead of the old ni*8. */
    size_t cap=0;
    for(int i=0;i<ni;i++){if(ids[i]<0||ids[i]>=tok->vocab_size)continue;if(ids[i]==tok->bos_id||ids[i]==tok->eos_id)continue;
        const char*cp=tok->tokens[ids[i]];while(*cp){if(cp[0]=='0'&&cp[1]=='x'){cap++;cp+=4;if(*cp=='+')cp++;}else cp++;}}
    char*buf=malloc(cap+1);int pos=0;
    for(int i=0;i<ni;i++){if(ids[i]<0||ids[i]>=tok->vocab_size)continue;if(ids[i]==tok->bos_id||ids[i]==tok->eos_id)continue;
    const char*nm=tok->tokens[ids[i]];const char*p=nm;while(*p){if(p[0]=='0'&&p[1]=='x'){unsigned int bv;if(sscanf(p,"0x%02x",&bv)==1)buf[pos++]=(char)bv;p+=4;if(*p=='+')p++;}else p++;}}
    buf[pos]='\0';*out_len=pos;return buf;
}

/* Tokenizer self-awareness update: called after each encode batch */
static void tok_eye_update(TokenizerEye *eye, int bytes_in, int tokens_out, int *ids, int n_ids, int vocab_size) {
    float ratio = (tokens_out > 0) ? (float)bytes_in / tokens_out : 1.0f;
    eye->compression_ratio = (eye->total_encoded == 0) ? ratio : 0.95f * eye->compression_ratio + 0.05f * ratio;
    /* token distribution entropy */
    int *counts = calloc(vocab_size, sizeof(int));
    int oov = 0;
    for (int i = 0; i < n_ids; i++) { if (ids[i] >= 0 && ids[i] < vocab_size) counts[ids[i]]++; else oov++; }
    float ent = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (counts[i] > 0) { float p = (float)counts[i] / n_ids; ent -= p * logf(p + 1e-10f); }
    }
    free(counts);
    eye->entropy = ent;
    float oov_r = (n_ids > 0) ? (float)oov / n_ids : 0;
    eye->oov_rate = (eye->total_encoded == 0) ? oov_r : 0.95f * eye->oov_rate + 0.05f * oov_r;
    eye->total_encoded += tokens_out;
    /* health: high compression + low OOV + moderate entropy = good */
    eye->health = fminf(1.0f, eye->compression_ratio / 4.0f) * (1.0f - eye->oov_rate);
    if (eye->entropy < 1.0f) eye->health *= 0.8f; /* repetitive input warning */
}

/* Code detection — does this text look like source code? */
static void tok_eye_detect_code(TokenizerEye *eye, const char *text, int len) {
    int code_signals = 0, total_lines = 0;
    const char *p = text, *end = text + len;
    while (p < end) {
        const char *nl = memchr(p, '\n', end - p);
        int ll = nl ? (int)(nl - p) : (int)(end - p);
        total_lines++;
        /* Code heuristics: indentation, braces, semicolons, keywords */
        if (ll > 0 && (p[0] == ' ' || p[0] == '\t')) code_signals++; /* indented */
        for (int i = 0; i < ll - 1; i++) {
            if (p[i] == '{' || p[i] == '}') { code_signals++; break; }
            if (p[i] == '(' && p[i+1] == ')') { code_signals++; break; }
            if (p[i] == '-' && p[i+1] == '>') { code_signals++; break; }
            if (p[i] == '=' && p[i+1] == '=') { code_signals++; break; }
            if (p[i] == '/' && p[i+1] == '/') { code_signals++; break; }
            if (p[i] == '#' && (ll > i+7 && strncmp(p+i, "#include", 8) == 0)) { code_signals += 3; break; }
            if (p[i] == '#' && (ll > i+6 && strncmp(p+i, "#define", 7) == 0)) { code_signals += 3; break; }
        }
        if (ll > 0 && p[ll-1] == ';') code_signals++;
        p = nl ? nl + 1 : end;
    }
    eye->code_ratio = (total_lines > 0) ? (float)code_signals / (total_lines * 2) : 0;
    if (eye->code_ratio > 1.0f) eye->code_ratio = 1.0f;
    eye->code_mode = (eye->code_ratio > 0.3f) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-AWARE DATA PARSER
 * tracks data quality. judges your dataset. feeds signal into training.
 * "your data is bad and you should feel bad" — the parser, probably.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float quality;        /* estimated data quality (0..1) */
    float domain_shift;   /* cosine distance from running mean */
    float noise_level;    /* non-text byte ratio */
    int total_parsed;
    float health;
    float *tok_mean;      /* running mean of token distribution */
    int tok_mean_dim;
} ParserEye;

static void parser_eye_init(ParserEye *eye, int vocab_size) {
    memset(eye, 0, sizeof(ParserEye));
    eye->quality = 1.0f; eye->health = 1.0f;
    eye->tok_mean = calloc(vocab_size, sizeof(float));
    eye->tok_mean_dim = vocab_size;
}

static void parser_eye_update(ParserEye *eye, int *tokens, int n_tokens, const char *raw_text, int raw_len) {
    /* noise: non-printable bytes */
    int noise = 0;
    for (int i = 0; i < raw_len; i++) {
        unsigned char c = (unsigned char)raw_text[i];
        if (c < 32 && c != '\n' && c != '\r' && c != '\t') noise++;
    }
    float nr = (raw_len > 0) ? (float)noise / raw_len : 0;
    eye->noise_level = 0.9f * eye->noise_level + 0.1f * nr;
    /* domain shift: cosine with running mean */
    float *cur = calloc(eye->tok_mean_dim, sizeof(float));
    for (int i = 0; i < n_tokens; i++) if (tokens[i] >= 0 && tokens[i] < eye->tok_mean_dim) cur[tokens[i]] += 1.0f;
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < eye->tok_mean_dim; i++) { dot += cur[i] * eye->tok_mean[i]; na += cur[i]*cur[i]; nb += eye->tok_mean[i]*eye->tok_mean[i]; }
    float cos_sim = (na > 0 && nb > 0) ? dot / (sqrtf(na) * sqrtf(nb) + 1e-8f) : 1.0f;
    eye->domain_shift = 1.0f - cos_sim;
    /* update running mean */
    float alpha = (eye->total_parsed == 0) ? 1.0f : 0.01f;
    for (int i = 0; i < eye->tok_mean_dim; i++) eye->tok_mean[i] = (1.0f-alpha)*eye->tok_mean[i] + alpha*cur[i];
    free(cur);
    eye->total_parsed++;
    eye->quality = (1.0f - eye->noise_level) * (1.0f - fminf(1.0f, eye->domain_shift * 2.0f));
    eye->health = eye->quality;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR — a float pointer that knows its size. still revolutionary in 2026.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *data; int size, rows, cols;
#ifdef USE_CUBLAS
    float *d_data;  /* GPU-resident copy — upload once, reuse */
    int    gpu_dirty; /* 1 = CPU changed, needs re-upload */
#endif
} Tensor;
static Tensor *tnew(int s){Tensor*t=calloc(1,sizeof(Tensor));t->data=calloc(s,sizeof(float));t->size=s;t->rows=1;t->cols=s;return t;}
static Tensor *tnew2d(int r,int co){Tensor*t=calloc(1,sizeof(Tensor));t->data=calloc(r*co,sizeof(float));t->size=r*co;t->rows=r;t->cols=co;return t;}
static void tinit(Tensor*t,float std){for(int i=0;i<t->size;i++)t->data[i]=rand_normal()*std;}
static void tfree(Tensor*t){if(t){
#ifdef USE_CUBLAS
    if(t->d_data)cudaFree(t->d_data);
#endif
    free(t->data);free(t);}}
/* Mark tensor as dirty — GPU copy needs re-upload after adam step */
static void tmark_dirty(Tensor*t){
#ifdef USE_CUBLAS
    if(t)t->gpu_dirty=1;
#else
    (void)t;
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MATH OPS — rmsnorm, matvec, softmax, rope, silu. the building blocks.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float silu_f(float x){return x/(1.0f+expf(-x));}
static float silu_bwd(float x){float s=1.0f/(1.0f+expf(-x));return s+x*s*(1.0f-s);}

static void rmsnorm(float*out,float*x,float*w,int d,float eps){float ss=0;for(int i=0;i<d;i++)ss+=x[i]*x[i];float inv=1.0f/sqrtf(ss/d+eps);for(int i=0;i<d;i++)out[i]=x[i]*inv*w[i];}
static void matvec(float*out,float*W,float*x,int r,int co){
#ifdef USE_CUBLAS
    cublas_init();
    float *dW=gpu_scratch(0,(size_t)r*co*4), *dx=gpu_scratch(1,(size_t)co*4), *dy=gpu_scratch(2,(size_t)r*4);
    cudaMemcpy(dW, W, (size_t)r*co*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t)co*4, cudaMemcpyHostToDevice);
    float a=1,b=0;
    cublasSgemv(g_cublas, CUBLAS_OP_T, co, r, &a, dW, co, dx, 1, &b, dy, 1);
    cudaMemcpy(out, dy, (size_t)r*4, cudaMemcpyDeviceToHost);
#elif defined(USE_BLAS)
    cblas_sgemv(CblasRowMajor,CblasNoTrans,r,co,1.0f,W,co,x,1,0.0f,out,1);
#else
    for(int i=0;i<r;i++){float s=0;float*row=W+i*co;for(int j=0;j<co;j++)s+=row[j]*x[j];out[i]=s;}
#endif
}
/* matvec with GPU weight cache — W stays on GPU between calls */
static void matvec_t(float*out,Tensor*W,float*x,int r,int co){
#ifdef USE_CUBLAS
    cublas_init();
    float *dW=gpu_ensure(W->data, W->size, &W->d_data, &W->gpu_dirty);
    float *dx=gpu_scratch(1,(size_t)co*4), *dy=gpu_scratch(2,(size_t)r*4);
    cudaMemcpy(dx, x, (size_t)co*4, cudaMemcpyHostToDevice);
    float a=1,b=0;
    cublasSgemv(g_cublas, CUBLAS_OP_T, co, r, &a, dW, co, dx, 1, &b, dy, 1);
    cudaMemcpy(out, dy, (size_t)r*4, cudaMemcpyDeviceToHost);
#else
    matvec(out, W->data, x, r, co);
#endif
}
static void softmax_n(float*x,int n){float mx=x[0];for(int i=1;i<n;i++)if(x[i]>mx)mx=x[i];float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}for(int i=0;i<n;i++)x[i]/=s;}

static void apply_rope(float*v,int pos,float*cc,float*sc,int hd){int h=hd/2,off=pos*h;for(int i=0;i<h;i++){float x0=v[i],x1=v[i+h];v[i]=x0*cc[off+i]-x1*sc[off+i];v[i+h]=x0*sc[off+i]+x1*cc[off+i];}}
static void rope_bwd(float*dv,int pos,float*cc,float*sc,int hd){int h=hd/2,off=pos*h;for(int i=0;i<h;i++){float d0=dv[i],d1=dv[i+h];dv[i]=d0*cc[off+i]+d1*sc[off+i];dv[i+h]=-d0*sc[off+i]+d1*cc[off+i];}}

/* matmul: C[M,N] = A[M,K] * B[N,K]^T */
static void mm_fwd(float*C,float*A,float*B,int M,int N,int K){
#ifdef USE_CUBLAS
    cublas_init();
    float *dA=gpu_scratch(0,(size_t)M*K*4), *dB=gpu_scratch(1,(size_t)N*K*4), *dC=gpu_scratch(2,(size_t)M*N*4);
    cudaMemcpy(dA, A, (size_t)M*K*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, (size_t)N*K*4, cudaMemcpyHostToDevice);
    float a=1,b=0;
    cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &a, dB, K, dA, K, &b, dC, N);
    cudaMemcpy(C, dC, (size_t)M*N*4, cudaMemcpyDeviceToHost);
#elif defined(USE_BLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    for(int m=0;m<M;m++){float*cm=C+m*N,*am=A+m*K;for(int n=0;n<N;n++){float s=0;float*bn=B+n*K;for(int k=0;k<K;k++)s+=am[k]*bn[k];cm[n]=s;}}
#endif
}
/* mm_fwd with GPU weight cache — B (weight tensor) stays on GPU */
static void mm_fwd_t(float*C,float*A,Tensor*B,int M,int N,int K){
#ifdef USE_CUBLAS
    cublas_init();
    float *dB=gpu_ensure(B->data, B->size, &B->d_data, &B->gpu_dirty);
    float *dA=gpu_scratch(0,(size_t)M*K*4), *dC=gpu_scratch(2,(size_t)M*N*4);
    cudaMemcpy(dA, A, (size_t)M*K*4, cudaMemcpyHostToDevice);
    float a=1,b=0;
    cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &a, dB, K, dA, K, &b, dC, N);
    cudaMemcpy(C, dC, (size_t)M*N*4, cudaMemcpyDeviceToHost);
#else
    mm_fwd(C, A, B->data, M, N, K);
#endif
}
static void mm_bwd(float*dA,float*dB,float*dC,float*A,float*B,int M,int N,int K){
#ifdef USE_CUBLAS
    cublas_init();
    size_t szMK=(size_t)M*K*4, szNK=(size_t)N*K*4, szMN=(size_t)M*N*4;
    float *ddC=gpu_scratch(0,szMN), *d_dA=gpu_scratch(1,szMK>szNK?szMK:szNK);
    float *d_other=gpu_scratch(2,szNK>szMK?szNK:szMK);
    float a=1,b1=1;
    cudaMemcpy(ddC, dC, szMN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_other, B, szNK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dA, dA, szMK, cudaMemcpyHostToDevice);
    cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &a, d_other, N, ddC, N, &b1, d_dA, K);
    cudaMemcpy(dA, d_dA, szMK, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_other, A, szMK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dA, dB, szNK, cudaMemcpyHostToDevice);
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, &a, d_other, K, ddC, N, &b1, d_dA, K);
    cudaMemcpy(dB, d_dA, szNK, cudaMemcpyDeviceToHost);
#elif defined(USE_BLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1.0f, dC, N, B, K, 1.0f, dA, K);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, K, M, 1.0f, dC, N, A, K, 1.0f, dB, K);
#else
    for(int m=0;m<M;m++){float*dc=dC+m*N,*am=A+m*K;for(int n=0;n<N;n++){float d=dc[n];if(d==0)continue;float*bn=B+n*K;for(int k=0;k<K;k++){dA[m*K+k]+=d*bn[k];dB[n*K+k]+=d*am[k];}}}
#endif
}
/* mm_bwd with GPU weight cache — B (weight tensor) stays on GPU */
static void mm_bwd_t(float*dA,float*dB_data,float*dC,float*A,Tensor*B,int M,int N,int K){
#ifdef USE_CUBLAS
    cublas_init();
    size_t szMK=(size_t)M*K*4, szNK=(size_t)N*K*4, szMN=(size_t)M*N*4;
    float *d_B=gpu_ensure(B->data, B->size, &B->d_data, &B->gpu_dirty);
    float *ddC=gpu_scratch(0,szMN), *d_dA=gpu_scratch(1,szMK>szNK?szMK:szNK);
    float a=1,b1=1;
    /* dA[M,K] += dC[M,N] * B[N,K] */
    cudaMemcpy(ddC, dC, szMN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dA, dA, szMK, cudaMemcpyHostToDevice);
    cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &a, d_B, N, ddC, N, &b1, d_dA, K);
    cudaMemcpy(dA, d_dA, szMK, cudaMemcpyDeviceToHost);
    /* dB[N,K] += dC^T * A */
    float *d_other=gpu_scratch(2,szMK);
    cudaMemcpy(d_other, A, szMK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dA, dB_data, szNK, cudaMemcpyHostToDevice);
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, &a, d_other, K, ddC, N, &b1, d_dA, K);
    cudaMemcpy(dB_data, d_dA, szNK, cudaMemcpyDeviceToHost);
#else
    mm_bwd(dA, dB_data, dC, A, B->data, M, N, K);
#endif
}
static void rn_fwd(float*o,float*x,float*w,int T,int D,float eps){for(int t=0;t<T;t++){float*xt=x+t*D,*ot=o+t*D;float ss=0;for(int i=0;i<D;i++)ss+=xt[i]*xt[i];float inv=1.0f/sqrtf(ss/D+eps);for(int i=0;i<D;i++)ot[i]=xt[i]*inv*w[i];}}
static void rn_bwd(float*dx,float*dw,float*dout,float*x,float*w,int T,int D,float eps){for(int t=0;t<T;t++){float*xt=x+t*D,*dot_=dout+t*D,*dxt=dx+t*D;float ss=0;for(int i=0;i<D;i++)ss+=xt[i]*xt[i];float var=ss/D+eps;float inv=1.0f/sqrtf(var);float cs=0;for(int i=0;i<D;i++)cs+=dot_[i]*w[i]*xt[i];float c2=cs/(D*var);for(int i=0;i<D;i++){dxt[i]+=(dot_[i]*w[i]-xt[i]*c2)*inv;dw[i]+=dot_[i]*xt[i]*inv;}}}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIVING EXPERTS — birth (mitosis), death (apoptosis), vitality.
 * experts aren't just weight matrices. they're organisms. they have
 * a heartbeat (vitality), a specialty (frequency in harmonic space),
 * and a lifespan. overloaded experts split. neglected experts die.
 * this is democracy applied to feed-forward networks. darwin would approve.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor *w_gate, *w_up, *w_down;
    float frequency;        /* position in harmonic space [0, 2π] */
    float vitality;         /* 0.0=dying, 1.0=peak */
    float specialization;   /* entropy of token distribution routed here */
    int age;                /* steps since birth */
    int tokens_seen;        /* lifetime token count */
    int alive;              /* 0=dead slot, 1=active */
    int low_vitality_count; /* consecutive steps with vitality < 0.1 */
    float entropy_hist[16]; /* for harmonic analysis */
    int eh_pos;
} Expert;

/* ═══════════════════════════════════════════════════════════════════════════════
 * PARLIAMENT ROUTER — variable-k election system.
 * each token triggers an election. experts campaign (dot product with input).
 * consensus determines how many experts get consulted:
 *   low consensus → more experts needed → k increases
 *   high consensus → clear winner → k decreases
 * harmonic resonance modulates votes. frequency gaps trigger new experts.
 * this is representative democracy applied to tensor routing.
 * gerrymandering not yet implemented (TODO for v2).
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor *w_vote;         /* [MAX_EXPERTS, dim] — voting weights */
    float consensus;        /* 0=chaos, 1=unanimous. computed per-token average */
    float faction_power[MAX_EXPERTS]; /* accumulated influence */
    int election_count;
    float temperature;      /* ephemeral vote sharpness (1=neutral); set per step from expert_temperature */
} Parliament;

/* ═══════════════════════════════════════════════════════════════════════════════
 * HARMONIC RESONANCE ENGINE — from AML core, adapted for experts.
 * each expert has a frequency. input gets fourier-decomposed.
 * experts that resonate with input get boosted. frequency gaps
 * are where new experts are born. inspired by Schumann resonances
 * and the fundamental conviction that everything is a wave.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float amplitudes[HARMONIC_N];   /* fourier decomposition of input */
    float dominant_freq;            /* strongest frequency component */
    float confidence;               /* how peaky the spectrum is */
} HarmonicState;

static void harmonic_decompose(HarmonicState *hs, float *entropy_hist, int hist_len) {
    /* fourier decomposition of entropy history — same idea as am_harmonic_forward */
    float max_amp = 0;
    int max_k = 0;
    for (int k = 0; k < HARMONIC_N && k < hist_len/2; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < hist_len; n++) {
            float angle = 6.2831853f * k * n / hist_len;
            re += entropy_hist[n] * cosf(angle);
            im += entropy_hist[n] * sinf(angle);
        }
        hs->amplitudes[k] = sqrtf(re*re + im*im) / hist_len;
        if (k > 0 && hs->amplitudes[k] > max_amp) { max_amp = hs->amplitudes[k]; max_k = k; }
    }
    hs->dominant_freq = (hist_len > 0) ? 6.2831853f * max_k / hist_len : 0;
    /* confidence: how much energy is in the dominant harmonic vs total */
    float total = 0;
    for (int k = 0; k < HARMONIC_N; k++) total += hs->amplitudes[k];
    hs->confidence = (total > 1e-8f) ? max_amp / total : 0;
}

static float expert_resonance(float expert_freq, HarmonicState *hs) {
    /* how much does this expert resonate with the current input spectrum? */
    float res = 0;
    for (int k = 0; k < HARMONIC_N; k++) {
        float freq_k = 6.2831853f * k / HARMONIC_N;
        float dist = fabsf(expert_freq - freq_k);
        if (dist > 3.14159f) dist = 6.2831853f - dist; /* wrap around */
        float weight = expf(-dist * dist * 2.0f); /* gaussian kernel */
        res += hs->amplitudes[k] * weight;
    }
    return res;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MODEL WEIGHTS — ephemeral topology, persistent parameters.
 * the topology changes every step. params stay. it's like a government:
 * politicians come and go, but the bureaucracy is forever.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor *attn_norm, *ffn_norm;
    Tensor *wq, *wk, *wv, *wo;
    Parliament parliament;
    Expert experts[MAX_EXPERTS];
    int n_alive;           /* how many experts are currently alive */
} LayerW;

typedef struct {
    Tensor *tok_emb, *output, *output_norm;
    LayerW *layers;
    int n_layers;
} ModelW;

static void init_expert(Expert *e, int dim, int hidden_dim, float freq, float init_std) {
    e->w_gate = tnew2d(hidden_dim, dim); tinit(e->w_gate, init_std);
    e->w_up = tnew2d(hidden_dim, dim); tinit(e->w_up, init_std);
    e->w_down = tnew2d(dim, hidden_dim); memset(e->w_down->data, 0, e->w_down->size * sizeof(float));
    e->frequency = freq;
    e->vitality = 0.7f;
    e->specialization = 0;
    e->age = 0;
    e->tokens_seen = 0;
    e->alive = 1;
    e->low_vitality_count = 0;
    memset(e->entropy_hist, 0, sizeof(e->entropy_hist));
    e->eh_pos = 0;
}

static void free_expert(Expert *e) {
    tfree(e->w_gate); tfree(e->w_up); tfree(e->w_down);
    e->w_gate = e->w_up = e->w_down = NULL;
    e->alive = 0;
    e->vitality = 0;
}

static void init_weights(ModelW *w, Config *c) {
    float es = 1.0f / sqrtf((float)c->dim), ls = 1.0f / sqrtf((float)c->dim);
    int qd = c->n_heads * c->head_dim, kd = c->n_kv_heads * c->head_dim;
    w->tok_emb = tnew2d(c->vocab_size, c->dim); tinit(w->tok_emb, es);
    w->output = tnew2d(c->vocab_size, c->dim); tinit(w->output, ls);
    w->output_norm = tnew(c->dim); for (int i = 0; i < c->dim; i++) w->output_norm->data[i] = 1.0f;
    w->n_layers = c->depth;
    w->layers = calloc(c->depth, sizeof(LayerW));
    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l];
        lw->attn_norm = tnew(c->dim); lw->ffn_norm = tnew(c->dim);
        for (int i = 0; i < c->dim; i++) { lw->attn_norm->data[i] = 1.0f; lw->ffn_norm->data[i] = 1.0f; }
        lw->wq = tnew2d(qd, c->dim); tinit(lw->wq, ls);
        lw->wk = tnew2d(kd, c->dim); tinit(lw->wk, ls);
        lw->wv = tnew2d(kd, c->dim); tinit(lw->wv, ls);
        lw->wo = tnew2d(c->dim, qd); memset(lw->wo->data, 0, lw->wo->size * sizeof(float));
        /* Parliament */
        lw->parliament.w_vote = tnew2d(MAX_EXPERTS, c->dim); tinit(lw->parliament.w_vote, 0.01f);
        lw->parliament.consensus = 0.5f;
        lw->parliament.election_count = 0;
        memset(lw->parliament.faction_power, 0, sizeof(lw->parliament.faction_power));
        /* Initialize living experts with harmonic spacing */
        lw->n_alive = c->initial_experts;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (e < c->initial_experts) {
                float freq = 6.2831853f * e / c->initial_experts;
                init_expert(&lw->experts[e], c->dim, c->hidden_dim, freq, ls);
            } else {
                memset(&lw->experts[e], 0, sizeof(Expert));
            }
        }
    }
}

/* Parliament election: returns k_t (variable), fills indices and weights */
static int parliament_elect(Parliament *p, Expert *experts, float *input, int dim,
                            HarmonicState *hs, int *selected, float *weights) {
    int n_alive = 0;
    int alive_idx[MAX_EXPERTS];
    for (int e = 0; e < MAX_EXPERTS; e++) if (experts[e].alive) alive_idx[n_alive++] = e;
    if (n_alive < MIN_EXPERTS) return 0; /* shouldn't happen */

    /* Compute votes: dot(w_vote[e], input) + harmonic resonance */
    float votes[MAX_EXPERTS];
    float max_vote = -1e30f;
    for (int i = 0; i < n_alive; i++) {
        int e = alive_idx[i];
        float *row = p->w_vote->data + e * dim;
        float dot = 0;
        for (int j = 0; j < dim; j++) dot += row[j] * input[j];
        /* harmonic resonance modulation */
        float res = expert_resonance(experts[e].frequency, hs);
        votes[e] = dot + 0.1f * res;
        if (votes[e] > max_vote) max_vote = votes[e];
    }
    /* Consensus: how peaked is the vote distribution? (normalized std) */
    float mean_v = 0;
    for (int i = 0; i < n_alive; i++) mean_v += votes[alive_idx[i]];
    mean_v /= n_alive;
    float var_v = 0;
    for (int i = 0; i < n_alive; i++) { float d = votes[alive_idx[i]] - mean_v; var_v += d*d; }
    var_v /= n_alive;
    float std_v = sqrtf(var_v + 1e-8f);
    /* consensus ∈ [0,1]: high std → high consensus (clear winner) */
    float consensus = fminf(1.0f, std_v / (fabsf(mean_v) + 1.0f));
    p->consensus = 0.9f * p->consensus + 0.1f * consensus;

    /* Variable k: low consensus → more experts. high → fewer. */
    int k = (int)(n_alive * (1.0f - p->consensus));
    if (k < 2) k = 2;
    if (k > n_alive) k = n_alive;

    /* Top-k selection */
    int used[MAX_EXPERTS] = {0};
    for (int ki = 0; ki < k; ki++) {
        float bv = -1e30f; int bi = 0;
        for (int i = 0; i < n_alive; i++) {
            int e = alive_idx[i];
            if (!used[e] && votes[e] > bv) { bv = votes[e]; bi = e; }
        }
        selected[ki] = bi;
        weights[ki] = votes[bi];
        used[bi] = 1;
    }
    /* Softmax over selected — temperature modulates vote sharpness (ephemeral expert_temperature) */
    float temp = (p->temperature > 0.05f) ? p->temperature : 1.0f;
    float mx = weights[0];
    for (int i = 1; i < k; i++) if (weights[i] > mx) mx = weights[i];
    float sum = 0;
    for (int i = 0; i < k; i++) { weights[i] = expf((weights[i] - mx) / temp); sum += weights[i]; }
    for (int i = 0; i < k; i++) weights[i] /= sum;

    p->election_count++;
    return k;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INFERENCE FORWARD — single token through the living parliament.
 * attention → parliament election → variable experts → residual.
 * ephemeral depth: skip layers when input complexity is low.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *x, *xb, *xb2, *xb3, *hb, *hb2;
    float *q, *k, *v, *att, *logits;
    float *key_cache, *value_cache, *cos_cache, *sin_cache;
    float *expert_out;
    HarmonicState hs;
} RunState;

static RunState alloc_run(Config *c) {
    RunState s; int kd = c->n_kv_heads * c->head_dim;
    s.x = calloc(c->dim, sizeof(float)); s.xb = calloc(c->dim, sizeof(float));
    s.xb2 = calloc(c->dim, sizeof(float)); s.xb3 = calloc(c->dim, sizeof(float));
    s.hb = calloc(c->hidden_dim, sizeof(float)); s.hb2 = calloc(c->hidden_dim, sizeof(float));
    s.q = calloc(c->n_heads * c->head_dim, sizeof(float));
    s.k = calloc(kd, sizeof(float)); s.v = calloc(kd, sizeof(float));
    s.att = calloc(c->n_heads * c->seq_len, sizeof(float));
    s.logits = calloc(c->vocab_size, sizeof(float));
    s.key_cache = calloc(c->depth * c->seq_len * kd, sizeof(float));
    s.value_cache = calloc(c->depth * c->seq_len * kd, sizeof(float));
    int half = c->head_dim / 2;
    s.cos_cache = calloc(c->seq_len * half, sizeof(float));
    s.sin_cache = calloc(c->seq_len * half, sizeof(float));
    for (int p = 0; p < c->seq_len; p++)
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(c->rope_theta, (float)(2*i) / (float)c->head_dim);
            float ang = (float)p * freq;
            s.cos_cache[p*half+i] = cosf(ang);
            s.sin_cache[p*half+i] = sinf(ang);
        }
    s.expert_out = calloc(c->dim, sizeof(float));
    memset(&s.hs, 0, sizeof(HarmonicState));
    return s;
}

static float *forward_token(ModelW *w, Config *c, RunState *s, int token, int pos) {
    int D = c->dim, kd = c->n_kv_heads * c->head_dim, hd = c->head_dim, H = c->hidden_dim;
    int hg = c->n_heads / c->n_kv_heads; float sc = 1.0f / sqrtf((float)hd);
    memcpy(s->x, w->tok_emb->data + token * D, D * sizeof(float));

    /* dynamic depth: each forward runs only the ephemeral active_layers (README: "how deep to go") */
    int adepth = (c->active_depth > 0 && c->active_depth <= c->depth) ? c->active_depth : c->depth;
    for (int l = 0; l < adepth; l++) {
        LayerW *lw = &w->layers[l];
        /* Attention */
        rmsnorm(s->xb, s->x, lw->attn_norm->data, D, c->norm_eps);
        matvec_t(s->q, lw->wq, s->xb, c->n_heads*hd, D);
        matvec_t(s->k, lw->wk, s->xb, c->n_kv_heads*hd, D);
        matvec_t(s->v, lw->wv, s->xb, c->n_kv_heads*hd, D);
        for (int h = 0; h < c->n_heads; h++) apply_rope(s->q+h*hd, pos, s->cos_cache, s->sin_cache, hd);
        for (int h = 0; h < c->n_kv_heads; h++) apply_rope(s->k+h*hd, pos, s->cos_cache, s->sin_cache, hd);
        int co = l * c->seq_len * kd + pos * kd;
        memcpy(s->key_cache + co, s->k, kd * sizeof(float));
        memcpy(s->value_cache + co, s->v, kd * sizeof(float));

        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / hg; float *qh = s->q + h*hd; float *att = s->att + h * c->seq_len;
            for (int t = 0; t <= pos; t++) {
                int ko = l*c->seq_len*kd + t*kd + kvh*hd;
                float dot = 0; for (int d = 0; d < hd; d++) dot += qh[d] * s->key_cache[ko+d];
                att[t] = dot * sc;
            }
            if (c->attn_clamp > 0) { float inv = 1.0f / c->attn_clamp; for (int t = 0; t <= pos; t++) att[t] = c->attn_clamp * tanhf(att[t] * inv); }
            softmax_n(att, pos + 1);
            float *xb2h = s->xb2 + h * hd; memset(xb2h, 0, hd * sizeof(float));
            for (int t = 0; t <= pos; t++) { float a = att[t]; int vo = l*c->seq_len*kd + t*kd + kvh*hd; for (int d = 0; d < hd; d++) xb2h[d] += a * s->value_cache[vo+d]; }
        }
        matvec_t(s->xb, lw->wo, s->xb2, D, D);
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];

        /* Parliament MoE FFN */
        rmsnorm(s->xb, s->x, lw->ffn_norm->data, D, c->norm_eps);
        memset(s->expert_out, 0, D * sizeof(float));

        int selected[MAX_EXPERTS]; float weights[MAX_EXPERTS];
        int k = parliament_elect(&lw->parliament, lw->experts, s->xb, D, &s->hs, selected, weights);

        for (int ki = 0; ki < k; ki++) {
            Expert *exp = &lw->experts[selected[ki]];
            matvec_t(s->hb, exp->w_gate, s->xb, H, D);
            matvec_t(s->hb2, exp->w_up, s->xb, H, D);
            for (int i = 0; i < H; i++) { float act = silu_f(s->hb[i]); s->hb[i] = act * s->hb2[i]; }
            matvec_t(s->xb2, exp->w_down, s->hb, D, H);
            for (int i = 0; i < D; i++) s->expert_out[i] += weights[ki] * s->xb2[i];
        }
        for (int i = 0; i < D; i++) s->x[i] += s->expert_out[i];
    }
    rmsnorm(s->x, s->x, w->output_norm->data, D, c->norm_eps);
    matvec_t(s->logits, w->output, s->x, c->vocab_size, D);
    return s->logits;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRAINING FORWARD — saves everything for backward. through living experts.
 * through parliament. through variable k. memory goes brrr.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { float *gate_pre, *up_pre, *act_out, *proj_out; } ExpertAct;
typedef struct {
    float *inp, *xn, *q, *k, *v, *attn_sc, *attn_sc_raw, *attn_out;
    float *attn_proj, *res_aa, *ffn_xn;
    float *moe_out;
    /* Parliament state per token */
    int *top_idx;     /* [T * MAX_EXPERTS] — variable k per token */
    float *top_wt;    /* [T * MAX_EXPERTS] */
    int *top_k;       /* [T] — k for each token */
    ExpertAct *ea;    /* [T * MAX_EXPERTS] */
} LayerAct;

typedef struct {
    LayerAct *layers; float *final_n, *logits, *residual;
    float *dr, *dxn, *dq, *dk, *dv, *dao, *dfxn, *dhb, *dhb2, *deo;
    float *cos_c, *sin_c; int T;
    HarmonicState hs; float aux_loss;
} TrainState;

static TrainState alloc_ts(Config *c) {
    TrainState s = {0}; int T = c->seq_len, D = c->dim, kv = c->n_kv_heads * c->head_dim;
    int qd = c->n_heads * c->head_dim, H = c->hidden_dim;
    s.T = T; s.layers = calloc(c->depth, sizeof(LayerAct));
    for (int l = 0; l < c->depth; l++) {
        LayerAct *la = &s.layers[l];
        la->inp = calloc(T*D, 4); la->xn = calloc(T*D, 4); la->q = calloc(T*qd, 4);
        la->k = calloc(T*kv, 4); la->v = calloc(T*kv, 4);
        la->attn_sc = calloc(T * c->n_heads * T, 4); la->attn_sc_raw = calloc(T * c->n_heads * T, 4); la->attn_out = calloc(T*qd, 4);
        la->attn_proj = calloc(T*D, 4); la->res_aa = calloc(T*D, 4); la->ffn_xn = calloc(T*D, 4);
        la->moe_out = calloc(T*D, 4);
        la->top_idx = calloc(T * MAX_EXPERTS, sizeof(int));
        la->top_wt = calloc(T * MAX_EXPERTS, sizeof(float));
        la->top_k = calloc(T, sizeof(int));
        la->ea = calloc(T * MAX_EXPERTS, sizeof(ExpertAct));
        for (int i = 0; i < T * MAX_EXPERTS; i++) {
            la->ea[i].gate_pre = calloc(H, 4); la->ea[i].up_pre = calloc(H, 4);
            la->ea[i].act_out = calloc(H, 4); la->ea[i].proj_out = calloc(D, 4);
        }
    }
    s.residual = calloc(T*D, 4); s.dr = calloc(T*D, 4); s.dxn = calloc(T*D, 4);
    s.dq = calloc(T*qd, 4); s.dk = calloc(T*kv, 4); s.dv = calloc(T*kv, 4);
    s.dao = calloc(T*qd, 4); s.dfxn = calloc(T*D, 4);
    s.dhb = calloc(T*H, 4); s.dhb2 = calloc(T*H, 4); s.deo = calloc(T*D, 4);
    int half = c->head_dim / 2;
    s.cos_c = calloc(T*half, 4); s.sin_c = calloc(T*half, 4);
    for (int p = 0; p < T; p++) for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(c->rope_theta, (float)(2*i) / (float)c->head_dim);
        float ang = (float)p * freq;
        s.cos_c[p*half+i] = cosf(ang); s.sin_c[p*half+i] = sinf(ang);
    }
    return s;
}

/* SFT tokenizer: split text by <human>/<ai> tags, build token + mask arrays.
 * mask[i]=1 means token i is AI response (compute loss), 0 means human (skip loss).
 * If no tags found, all mask=1 (continuation mode, backwards compatible). */
static void sft_tokenize(Tokenizer *tok, const char *text, int len,
                          int **out_tok, int **out_mask, int *out_n) {
    int tcap = 4096, tn = 0;
    int *toks = malloc(tcap * sizeof(int));
    int *mask = malloc(tcap * sizeof(int));
    int is_ai = 0;
    const char *p = text, *end = text + len;
    int has_tags = (strstr(text, "<human>") || strstr(text, "<ai>"));
    while (p < end) {
        const char *nh = NULL, *na = NULL;
        for (const char *s = p; s <= end - 7; s++)
            if (memcmp(s, "<human>", 7) == 0) { nh = s; break; }
        for (const char *s = p; s <= end - 4; s++)
            if (memcmp(s, "<ai>", 4) == 0) { na = s; break; }
        const char *tag = NULL; int tlen = 0, next_ai = 0;
        if (nh && (!na || nh < na)) { tag = nh; tlen = 7; next_ai = 0; }
        else if (na) { tag = na; tlen = 4; next_ai = 1; }
        const char *seg_end = tag ? tag : end;
        int seg_len = (int)(seg_end - p);
        if (seg_len > 0) {
            int sn; int *st = tok_encode(tok, p, seg_len, &sn);
            while (tn + sn >= tcap) { tcap *= 2; toks = realloc(toks, tcap * sizeof(int)); mask = realloc(mask, tcap * sizeof(int)); }
            for (int i = 0; i < sn; i++) { toks[tn] = st[i]; mask[tn] = has_tags ? is_ai : 1; tn++; }
            free(st);
        }
        if (tag) { is_ai = next_ai; p = tag + tlen; } else break;
    }
    *out_tok = toks; *out_mask = mask; *out_n = tn;
}

static float train_fwd(ModelW *w, Config *c, TrainState *s, int *tokens, int *targets, int T) {
    int D = c->dim, kv = c->n_kv_heads * c->head_dim, qd = c->n_heads * c->head_dim;
    int hd = c->head_dim, hg = c->n_heads / c->n_kv_heads, H = c->hidden_dim;
    float sc = 1.0f / sqrtf((float)hd);
    s->aux_loss = 0;

    for (int t = 0; t < T; t++) memcpy(s->residual + t*D, w->tok_emb->data + tokens[t]*D, D * sizeof(float));

    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l]; LayerAct *la = &s->layers[l];
        memcpy(la->inp, s->residual, T*D*4);

        /* Attention */
        rn_fwd(la->xn, s->residual, lw->attn_norm->data, T, D, c->norm_eps);
        mm_fwd_t(la->q, la->xn, lw->wq, T, qd, D);
        mm_fwd_t(la->k, la->xn, lw->wk, T, kv, D);
        mm_fwd_t(la->v, la->xn, lw->wv, T, kv, D);
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < c->n_heads; h++) apply_rope(la->q + t*qd + h*hd, t, s->cos_c, s->sin_c, hd);
            for (int h = 0; h < c->n_kv_heads; h++) apply_rope(la->k + t*kv + h*hd, t, s->cos_c, s->sin_c, hd);
        }
        memset(la->attn_out, 0, T*qd*4);
        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / hg;
            for (int t = 0; t < T; t++) {
                float *qt = la->q + t*qd + h*hd;
                float *att = la->attn_sc + (t * c->n_heads + h) * T;
                for (int sp = 0; sp <= t; sp++) { float *ks = la->k + sp*kv + kvh*hd; float dot = 0; for (int d = 0; d < hd; d++) dot += qt[d]*ks[d]; att[sp] = dot*sc; }
                if (c->attn_clamp > 0) { float inv = 1.0f / c->attn_clamp; for (int sp = 0; sp <= t; sp++) att[sp] = c->attn_clamp * tanhf(att[sp] * inv); }
                /* Save post-clamp, pre-softmax scores for backward dtanh */
                float *raw = la->attn_sc_raw + (t * c->n_heads + h) * T;
                for (int sp = 0; sp <= t; sp++) raw[sp] = att[sp];
                float mx = -1e30f; for (int sp = 0; sp <= t; sp++) if (att[sp] > mx) mx = att[sp];
                float se = 0; for (int sp = 0; sp <= t; sp++) { att[sp] = expf(att[sp]-mx); se += att[sp]; }
                for (int sp = 0; sp <= t; sp++) att[sp] /= se; for (int sp = t+1; sp < T; sp++) att[sp] = 0;
                float *oh = la->attn_out + t*qd + h*hd;
                for (int sp = 0; sp <= t; sp++) { float a = att[sp]; float *vs = la->v + sp*kv + kvh*hd; for (int d = 0; d < hd; d++) oh[d] += a*vs[d]; }
            }
        }
        mm_fwd_t(la->attn_proj, la->attn_out, lw->wo, T, D, qd);
        for (int i = 0; i < T*D; i++) s->residual[i] += la->attn_proj[i];
        memcpy(la->res_aa, s->residual, T*D*4);

        /* Parliament MoE FFN */
        rn_fwd(la->ffn_xn, s->residual, lw->ffn_norm->data, T, D, c->norm_eps);
        memset(la->moe_out, 0, T*D*4);

        for (int t = 0; t < T; t++) {
            float *xn_t = la->ffn_xn + t*D;
            int *ti = la->top_idx + t * MAX_EXPERTS;
            float *tw = la->top_wt + t * MAX_EXPERTS;
            int k = parliament_elect(&lw->parliament, lw->experts, xn_t, D, &s->hs, ti, tw);
            la->top_k[t] = k;

            for (int ki = 0; ki < k; ki++) {
                int eI = ti[ki]; float eW = tw[ki];
                Expert *exp = &lw->experts[eI];
                ExpertAct *ea = &la->ea[t * MAX_EXPERTS + ki];
                matvec_t(ea->gate_pre, exp->w_gate, xn_t, H, D);
                matvec_t(ea->up_pre, exp->w_up, xn_t, H, D);
                for (int i = 0; i < H; i++) { float act = silu_f(ea->gate_pre[i]); ea->act_out[i] = act * ea->up_pre[i]; }
                matvec_t(ea->proj_out, exp->w_down, ea->act_out, D, H);
                float *mo = la->moe_out + t*D;
                for (int i = 0; i < D; i++) mo[i] += eW * ea->proj_out[i];
            }
        }
        /* Aux loss: load balancing across experts */
        { float frac[MAX_EXPERTS] = {0}; int na = 0;
          for (int e = 0; e < MAX_EXPERTS; e++) if (lw->experts[e].alive) na++;
          for (int t = 0; t < T; t++) { int k = la->top_k[t];
            int *ti = la->top_idx + t * MAX_EXPERTS; float *tw = la->top_wt + t * MAX_EXPERTS;
            for (int ki = 0; ki < k; ki++) frac[ti[ki]] += tw[ki]; }
          for (int e = 0; e < MAX_EXPERTS; e++) frac[e] /= (float)T;
          float al = 0;
          for (int e = 0; e < MAX_EXPERTS; e++) if (lw->experts[e].alive) al += frac[e] * frac[e];
          s->aux_loss += c->aux_loss_w * (float)na * al; }
        for (int i = 0; i < T*D; i++) s->residual[i] += la->moe_out[i];
    }

    s->final_n = calloc(T*D, 4); rn_fwd(s->final_n, s->residual, w->output_norm->data, T, D, c->norm_eps);
    s->logits = calloc(T * c->vocab_size, 4); mm_fwd_t(s->logits, s->final_n, w->output, T, c->vocab_size, D);

    float loss = 0; int nv = 0;
    for (int t = 0; t < T; t++) {
        if (targets[t] < 0) continue;
        float *lt = s->logits + t * c->vocab_size;
        float mx = lt[0]; for (int j = 1; j < c->vocab_size; j++) if (lt[j] > mx) mx = lt[j];
        float se = 0; for (int j = 0; j < c->vocab_size; j++) se += expf(lt[j] - mx);
        loss += -(lt[targets[t]] - mx - logf(se)); nv++;
    }
    return nv > 0 ? loss / nv + s->aux_loss : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRAINING BACKWARD — analytical gradients through variable-k parliament.
 * the hardest backward pass in the quartet. moe.c had fixed k=2.
 * here k varies per token. softmax Jacobian is k_t × k_t, different size
 * for every token. we pre-allocate grad_buf[MAX_EXPERTS] and use first k_t
 * slots. no malloc in the hot path. the parliament's decisions are reversible.
 * the experts' shame is not.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void train_bwd(ModelW *w, Config *c, TrainState *s, int *tokens, int *targets, int T, float **g) {
    int D = c->dim, kv = c->n_kv_heads * c->head_dim, qd = c->n_heads * c->head_dim;
    int hd = c->head_dim, H = c->hidden_dim, hg = c->n_heads / c->n_kv_heads, V = c->vocab_size;
    float sc = 1.0f / sqrtf((float)hd);
    int *layer_gi = NULL;
    int nv = 0; for (int t = 0; t < T; t++) if (targets[t] >= 0) nv++;
    if (nv == 0) goto done;
    float inv_n = 1.0f / (float)nv;

    /* d_logits — cross entropy backward */
    float *dl = calloc(T*V, 4);
    for (int t = 0; t < T; t++) {
        if (targets[t] < 0) continue;
        float *lt = s->logits + t*V; float *d = dl + t*V;
        float mx = lt[0]; for (int j = 1; j < V; j++) if (lt[j] > mx) mx = lt[j];
        float se = 0; for (int j = 0; j < V; j++) { d[j] = expf(lt[j]-mx); se += d[j]; }
        for (int j = 0; j < V; j++) d[j] = (d[j]/se) * inv_n;
        d[targets[t]] -= inv_n;
    }

    /* LM head bwd */
    float *dfn = calloc(T*D, 4);
    mm_bwd_t(dfn, g[1], dl, s->final_n, w->output, T, V, D);
    /* Final norm bwd */
    memset(s->dr, 0, T*D*4);
    rn_bwd(s->dr, g[2], dfn, s->residual, w->output_norm->data, T, D, c->norm_eps);
    free(dfn); free(dl);

    /* Precompute gradient index offsets per layer — because expert count varies.
     * collect_params layout: [tok_emb, output, output_norm, layer0(7 fixed + n_alive*3), layer1(...), ...] */
    layer_gi = calloc(c->depth, sizeof(int));
    {
        int off = 3;
        for (int l = 0; l < c->depth; l++) {
            layer_gi[l] = off;
            off += 7; /* attn_norm, wq, wk, wv, wo, ffn_norm, w_vote */
            for (int e = 0; e < MAX_EXPERTS; e++)
                if (w->layers[l].experts[e].alive) off += 3; /* w_gate, w_up, w_down */
        }
    }

    /* Build expert→grad_index map per layer (which slot in g[] for each expert's weights) */
    int expert_gi[MAX_LAYERS][MAX_EXPERTS]; /* gi of w_gate for expert e; w_up=gi+1, w_down=gi+2 */
    for (int l = 0; l < c->depth; l++) {
        int off = layer_gi[l] + 7;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (w->layers[l].experts[e].alive) {
                expert_gi[l][e] = off;
                off += 3;
            } else {
                expert_gi[l][e] = -1;
            }
        }
    }

    /* Layers in reverse */
    for (int l = c->depth - 1; l >= 0; l--) {
        LayerW *lw = &w->layers[l]; LayerAct *la = &s->layers[l];
        int gi = layer_gi[l];

        /* MoE backward through variable-k parliament */
        float *dmo = calloc(T*D, 4);
        memcpy(dmo, s->dr, T*D*4);

        memset(s->dfxn, 0, T*D*4);
        int vote_gi = gi + 6; /* w_vote gradient index */


        /* Aux loss backward: frac[] computed once per layer */
        float aux_frac[MAX_EXPERTS] = {0}; int aux_na = 0;
        for (int e = 0; e < MAX_EXPERTS; e++) if (lw->experts[e].alive) aux_na++;
        for (int tt = 0; tt < T; tt++) { int kk = la->top_k[tt];
          for (int kki = 0; kki < kk; kki++)
            aux_frac[la->top_idx[tt*MAX_EXPERTS+kki]] += la->top_wt[tt*MAX_EXPERTS+kki]; }
        for (int e = 0; e < MAX_EXPERTS; e++) aux_frac[e] /= (float)T;

        /* Expert backward — per token, per selected expert */
        for (int t = 0; t < T; t++) {
            int k = la->top_k[t];
            int *ti = la->top_idx + t * MAX_EXPERTS;
            float *tw = la->top_wt + t * MAX_EXPERTS;
            float *dm = dmo + t*D;
            float *xn_t = la->ffn_xn + t*D;

            for (int ki = 0; ki < k; ki++) {
                int eI = ti[ki]; float eW = tw[ki];
                Expert *exp = &lw->experts[eI];
                ExpertAct *ea = &la->ea[t * MAX_EXPERTS + ki];
                int egi = expert_gi[l][eI]; /* grad index for this expert's w_gate */

                /* dw_down + da (hidden grad) — merged single pass */
                float *da = calloc(H, 4);
                float *hid = NULL;
                if (egi >= 0) {
                    hid = calloc(H, 4);
                    for (int j = 0; j < H; j++) hid[j] = silu_f(ea->gate_pre[j]) * ea->up_pre[j];
                }
                for (int i = 0; i < D; i++) {
                    float dp = eW * dm[i];
                    for (int j = 0; j < H; j++) {
                        da[j] += dp * exp->w_down->data[i*H+j];
                        if (hid) g[egi+2][i*H+j] += dp * hid[j];
                    }
                }
                if (hid) free(hid);

                /* dgate, dup → dw_gate, dw_up */
                for (int i = 0; i < H; i++) {
                    float gp = ea->gate_pre[i], up = ea->up_pre[i];
                    float gd = silu_bwd(gp); float act = silu_f(gp);
                    float dg = da[i] * up * gd, du = da[i] * act;
                    /* dx (input gradient) */
                    for (int j = 0; j < D; j++) s->dfxn[t*D+j] += dg * exp->w_gate->data[i*D+j] + du * exp->w_up->data[i*D+j];
                    /* Expert weight gradients — accumulated into correct g[] slots */
                    if (egi >= 0) {
                        for (int j = 0; j < D; j++) {
                            g[egi][i*D+j] += dg * xn_t[j];     /* dw_gate */
                            g[egi+1][i*D+j] += du * xn_t[j];   /* dw_up */
                        }
                    }
                }
                free(da);
            }

            /* ═══ Variable-k parliament backward (CRITICAL) ═══
             * softmax Jacobian is k_t × k_t for this token.
             * dweights[ki] = dot(expert_output[ki], dout)
             * dvotes[ki] = weights[ki] * (dweights[ki] - dot(weights, dweights))
             * pre-allocated grad_buf[MAX_EXPERTS] — no malloc */
            float dw_buf[MAX_EXPERTS], dvote_buf[MAX_EXPERTS];
            for (int ki = 0; ki < k; ki++) {
                ExpertAct *ea = &la->ea[t * MAX_EXPERTS + ki];
                float d = 0;
                for (int i = 0; i < D; i++) d += dm[i] * ea->proj_out[i];
                dw_buf[ki] = d;
            }
            float dot_wd = 0;
            for (int ki = 0; ki < k; ki++) dot_wd += tw[ki] * dw_buf[ki];
            for (int ki = 0; ki < k; ki++) {
                dvote_buf[ki] = tw[ki] * (dw_buf[ki] - dot_wd) / ((lw->parliament.temperature > 0.05f) ? lw->parliament.temperature : 1.0f);   /* match forward softmax(v/temperature) */
                int eI = ti[ki];
                /* gradient to w_vote and to input */
                float *vote_row = lw->parliament.w_vote->data + eI * D;
                for (int j = 0; j < D; j++) {
                    s->dfxn[t*D+j] += dvote_buf[ki] * vote_row[j];
                    /* w_vote grad — accumulated into g[vote_gi] */
                    g[vote_gi][eI*D+j] += dvote_buf[ki] * xn_t[j];
                }
            }
            /* Aux loss gradient through softmax Jacobian */
            { float daux[MAX_EXPERTS];
              for (int ki = 0; ki < k; ki++)
                daux[ki] = c->aux_loss_w * (float)aux_na * 2.0f * aux_frac[ti[ki]] / (float)T;
              float dot_aux = 0;
              for (int ki = 0; ki < k; ki++) dot_aux += tw[ki] * daux[ki];
              for (int ki = 0; ki < k; ki++) {
                float dv = tw[ki] * (daux[ki] - dot_aux) / ((lw->parliament.temperature > 0.05f) ? lw->parliament.temperature : 1.0f);
                int eI = ti[ki];
                float *vote_row = lw->parliament.w_vote->data + eI * D;
                for (int j = 0; j < D; j++) {
                    s->dfxn[t*D+j] += dv * vote_row[j];
                    g[vote_gi][eI*D+j] += dv * xn_t[j];
                }
              } }
        }
        free(dmo);

        /* FFN norm bwd */
        rn_bwd(s->dr, g[gi+5], s->dfxn, la->res_aa, lw->ffn_norm->data, T, D, c->norm_eps);

        /* Attention backward */
        float *dap = calloc(T*D, 4);
        memcpy(dap, s->dr, T*D*4);

        memset(s->dao, 0, T*qd*4);
        mm_bwd_t(s->dao, g[gi+4], dap, la->attn_out, lw->wo, T, D, qd);
        free(dap);

        memset(s->dq, 0, T*qd*4); memset(s->dk, 0, T*kv*4); memset(s->dv, 0, T*kv*4);
        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / hg;
            for (int t = 0; t < T; t++) {
                float *doh = s->dao + t*qd + h*hd;
                float *att = la->attn_sc + (t * c->n_heads + h) * T;
                float *da = calloc(T, 4);
                for (int sp = 0; sp <= t; sp++) {
                    float *vs = la->v + sp*kv + kvh*hd; float d = 0;
                    for (int d2 = 0; d2 < hd; d2++) d += doh[d2] * vs[d2]; da[sp] = d;
                    float a = att[sp]; float *dvs = s->dv + sp*kv + kvh*hd;
                    for (int d2 = 0; d2 < hd; d2++) dvs[d2] += a * doh[d2];
                }
                float dot_ad = 0; for (int sp = 0; sp <= t; sp++) dot_ad += att[sp] * da[sp];
                float *qt = la->q + t*qd + h*hd; float *dqt = s->dq + t*qd + h*hd;
                float *raw = la->attn_sc_raw + (t * c->n_heads + h) * T;
                for (int sp = 0; sp <= t; sp++) {
                    float ds = att[sp] * (da[sp] - dot_ad);
                    /* Backward through attn_clamp: dy/dx = 1 - (y/C)^2 */
                    if (c->attn_clamp > 0) {
                        float yc = raw[sp] / c->attn_clamp;
                        ds *= (1.0f - yc * yc);
                    }
                    ds *= sc;
                    float *ks = la->k + sp*kv + kvh*hd; float *dks = s->dk + sp*kv + kvh*hd;
                    for (int d2 = 0; d2 < hd; d2++) { dqt[d2] += ds*ks[d2]; dks[d2] += ds*qt[d2]; }
                }
                free(da);
            }
        }
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < c->n_heads; h++) rope_bwd(s->dq + t*qd + h*hd, t, s->cos_c, s->sin_c, hd);
            for (int h = 0; h < c->n_kv_heads; h++) rope_bwd(s->dk + t*kv + h*hd, t, s->cos_c, s->sin_c, hd);
        }
        memset(s->dxn, 0, T*D*4);
        mm_bwd_t(s->dxn, g[gi+1], s->dq, la->xn, lw->wq, T, qd, D);
        mm_bwd_t(s->dxn, g[gi+2], s->dk, la->xn, lw->wk, T, kv, D);
        mm_bwd_t(s->dxn, g[gi+3], s->dv, la->xn, lw->wv, T, kv, D);

        float *ds = calloc(T*D, 4); memcpy(ds, s->dr, T*D*4); memset(s->dr, 0, T*D*4);
        rn_bwd(s->dr, g[gi], s->dxn, la->inp, lw->attn_norm->data, T, D, c->norm_eps);
        for (int i = 0; i < T*D; i++) s->dr[i] += ds[i]; free(ds);
    }

    /* Embedding bwd */
    for (int t = 0; t < T; t++) { float *de = g[0] + tokens[t]*D; float *dr = s->dr + t*D; for (int i = 0; i < D; i++) de[i] += dr[i]; }
done:
    free(layer_gi);
    free(s->logits); s->logits = NULL; free(s->final_n); s->final_n = NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PARAMETER COLLECTION — collects all trainable tensors + living expert weights.
 * living experts are collected dynamically because they can be born/die.
 * dead expert weights are skipped. this is parameter-space democracy.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { Tensor **tensors; int count, cap; } ParamList;

static ParamList collect_params(ModelW *w, Config *c) {
    int mx = 3 + w->n_layers * (7 + MAX_EXPERTS * 3);
    ParamList p; p.tensors = calloc(mx, sizeof(Tensor*)); p.count = 0; p.cap = mx;
    p.tensors[p.count++] = w->tok_emb;
    p.tensors[p.count++] = w->output;
    p.tensors[p.count++] = w->output_norm;
    for (int l = 0; l < w->n_layers; l++) {
        LayerW *lw = &w->layers[l];
        p.tensors[p.count++] = lw->attn_norm;   /* gi+0 */
        p.tensors[p.count++] = lw->wq;           /* gi+1 */
        p.tensors[p.count++] = lw->wk;           /* gi+2 */
        p.tensors[p.count++] = lw->wv;           /* gi+3 */
        p.tensors[p.count++] = lw->wo;           /* gi+4 */
        p.tensors[p.count++] = lw->ffn_norm;     /* gi+5 */
        p.tensors[p.count++] = lw->parliament.w_vote; /* gi+6 */
        /* Living expert weights */
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (lw->experts[e].alive) {
                p.tensors[p.count++] = lw->experts[e].w_gate;
                p.tensors[p.count++] = lw->experts[e].w_up;
                p.tensors[p.count++] = lw->experts[e].w_down;
            }
        }
    }
    return p;
}

static long count_params(ModelW *w, Config *c) {
    long total = (long)c->vocab_size * c->dim * 2 + c->dim; /* emb + output + norm */
    int qd = c->n_heads * c->head_dim, kd = c->n_kv_heads * c->head_dim;
    for (int l = 0; l < c->depth; l++) {
        total += c->dim * 2; /* norms */
        total += (long)qd * c->dim + (long)kd * c->dim * 2 + (long)c->dim * qd; /* attn */
        total += (long)MAX_EXPERTS * c->dim; /* w_vote */
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (w->layers[l].experts[e].alive)
                total += (long)c->dim * c->hidden_dim * 3;
        }
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MITOSIS & APOPTOSIS — the lifecycle of experts.
 * overloaded expert + high vitality → splits into two (mitosis).
 * child inherits parent weights + gaussian noise. frequency offset.
 * neglected expert → vitality drops → 8 consecutive low steps → dies (apoptosis).
 * the most generalist expert splits; the child gets a frequency offset.
 * this is cellular biology applied to mixture-of-experts.
 * the immune system is the harmonic backbone.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void update_vitality(LayerW *lw, Config *c, int total_tokens) {
    int n_alive = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (lw->experts[e].alive) n_alive++;
    if (n_alive == 0) return;
    float fair_share = (float)total_tokens / n_alive;

    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!lw->experts[e].alive) continue;
        Expert *exp = &lw->experts[e];
        float ratio = (fair_share > 0) ? (float)exp->tokens_seen / fair_share : 1.0f;
        /* Winners grow, losers shrink */
        exp->vitality += (ratio - 1.0f) * 0.05f;
        exp->vitality = fmaxf(0.0f, fminf(1.0f, exp->vitality));
        exp->age++;
        /* Track low vitality for apoptosis */
        if (exp->vitality < c->vitality_threshold) exp->low_vitality_count++;
        else exp->low_vitality_count = 0;
        /* Harmonic: roll the per-step activation history (read below + by harmonic analysis) */
        exp->entropy_hist[exp->eh_pos % 16] = exp->vitality; exp->eh_pos++;
        /* Specialization = Shannon surprisal of this expert's usage share + activation variance:
         * low usage / steady activation ⇒ specialized; high usage / volatile ⇒ generalist. */
        float share = (total_tokens > 0) ? (float)exp->tokens_seen / (float)total_tokens : 0.0f;
        float ehm = 0; for (int h = 0; h < 16; h++) ehm += exp->entropy_hist[h]; ehm /= 16.0f;
        float ehv = 0; for (int h = 0; h < 16; h++) { float d = exp->entropy_hist[h] - ehm; ehv += d*d; } ehv /= 16.0f;
        exp->specialization = ((share > 1e-6f) ? -share * logf(share) : 0.0f) + ehv;
        /* Reset tokens_seen for next step */
        exp->tokens_seen = 0;
    }
    lw->n_alive = n_alive;
}

static int try_mitosis(LayerW *lw, Config *c) {
    /* Find overloaded expert with high vitality */
    int n_alive = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (lw->experts[e].alive) n_alive++;
    if (n_alive >= MAX_EXPERTS) return 0;

    int parent = -1; float best_score = -1e9f;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!lw->experts[e].alive) continue;
        Expert *exp = &lw->experts[e];
        /* overloaded (high vitality) + old enough; among those, split the most GENERALIST
         * (highest vitality, lowest specialization) — specialists are preserved, not cloned. */
        if (exp->vitality > 0.8f && exp->age >= 20) {
            float score = exp->vitality - exp->specialization;
            if (score > best_score) { best_score = score; parent = e; }
        }
    }
    if (parent < 0) return 0;

    /* Find dead slot */
    int child = -1;
    for (int e = 0; e < MAX_EXPERTS; e++) if (!lw->experts[e].alive) { child = e; break; }
    if (child < 0) return 0;

    /* Birth: child inherits parent weights + noise, frequency offset */
    Expert *p = &lw->experts[parent];
    float child_freq = p->frequency + 3.14159f / (n_alive + 1);
    if (child_freq > 6.2831853f) child_freq -= 6.2831853f;
    init_expert(&lw->experts[child], c->dim, c->hidden_dim, child_freq, 0);
    /* Copy parent weights + noise */
    Expert *ch = &lw->experts[child];
    float noise = 0.01f;
    for (int i = 0; i < p->w_gate->size; i++) ch->w_gate->data[i] = p->w_gate->data[i] + rand_normal() * noise;
    for (int i = 0; i < p->w_up->size; i++) ch->w_up->data[i] = p->w_up->data[i] + rand_normal() * noise;
    for (int i = 0; i < p->w_down->size; i++) ch->w_down->data[i] = p->w_down->data[i] + rand_normal() * noise;
    ch->vitality = 0.5f;
    p->vitality *= 0.8f; /* parent weakened by reproduction */
    lw->n_alive++;
    return 1;
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * CHUCK OPTIMIZER — 9 levels of self-awareness. from lee.c, adapted.
 * the optimizer that thinks about thinking about optimizing.
 * 9 self-awareness mechanisms, all live in chuck_step: (1) loss-trend dampening,
 * (2) stagnation-escape noise, (3) macro-patience lr decay, (4) activation health
 * from the eyes, (5) Ψ subjectivity / memory recall, (6) adaptive gradient
 * clipping, (7) global gradient clipping, (8) Ψ-recall λ (eff_lambda), (9) grad-norm EMA.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float hist[CHUCK_WINDOW]; float loss_ema, gnorm_ema;
    float dampen, noise, sigma, psi, psi_w, eff_lambda;
    float macro_ema, best_macro, lr_scale;
    int macro_stag, macro_drops;
    int pos, full, stag, global_step;
} ChuckState;

/* Chuck persistent memory — reservoir sampling */
typedef struct { float loss, gnorm, lambda, delta_loss; } ChuckMem;
static ChuckMem chuck_mem[CHUCK_MEM_CAP];
static int chuck_mem_n = 0;

static void chuck_mem_save(ChuckMem *snap) {
    if (chuck_mem_n < CHUCK_MEM_CAP) { chuck_mem[chuck_mem_n++] = *snap; }
    else { int idx = (int)(rand_uniform() * (chuck_mem_n + 1));
           if (idx < CHUCK_MEM_CAP) chuck_mem[idx] = *snap; chuck_mem_n++; }
}

static float chuck_mem_recall(float loss, float gnorm) {
    if (chuck_mem_n == 0) return -1;
    int n = (chuck_mem_n < CHUCK_MEM_CAP) ? chuck_mem_n : CHUCK_MEM_CAP;
    float best_dist = 1e30f; int best_i = 0;
    for (int i = 0; i < n; i++) {
        float dl = (loss - chuck_mem[i].loss) / (loss + 1e-8f);
        float dg = (gnorm - chuck_mem[i].gnorm) / (gnorm + 1e-8f);
        float dist = dl*dl + dg*dg;
        if (dist < best_dist) { best_dist = dist; best_i = i; }
    }
    return chuck_mem[best_i].lambda;
}

static void chuck_step(ChuckState *ck, float loss,
                       ParamList *params, float **grads,
                       TokenizerEye *tok_eye, ParserEye *parser_eye) {
    /* ═══ Level 1: Global self-awareness (loss trend) ═══ */
    if (ck->loss_ema == 0) ck->loss_ema = loss;
    else ck->loss_ema = 0.99f * ck->loss_ema + 0.01f * loss;
    ck->hist[ck->pos % CHUCK_WINDOW] = ck->loss_ema;
    ck->pos++;
    if (ck->pos >= CHUCK_WINDOW) ck->full = 1;
    if (ck->full) {
        int q = CHUCK_WINDOW / 4;
        float recent = 0, old = 0;
        for (int i = 0; i < q; i++) {
            recent += ck->hist[(ck->pos-1-i) % CHUCK_WINDOW];
            old += ck->hist[(ck->pos-CHUCK_WINDOW+i) % CHUCK_WINDOW];
        }
        recent /= q; old /= q;
        float trend = (recent - old) / (old + 1e-8f);
        if (trend > 0.01f) ck->dampen *= 0.95f;
        else if (trend < -0.05f) ck->dampen *= 1.05f;
        if (fabsf(trend) < 0.001f) { ck->stag++; if (ck->stag > 8) { ck->noise = 0.001f; ck->stag = 0; } }
        else { ck->stag = 0; ck->noise *= 0.9f; }
        if (ck->dampen < CHUCK_DAMP_LO) ck->dampen = CHUCK_DAMP_LO;
        if (ck->dampen > CHUCK_DAMP_HI) ck->dampen = CHUCK_DAMP_HI;
    }

    /* ═══ Level 9: Macro patience ═══ */
    ck->global_step++;
    if (ck->macro_ema == 0) ck->macro_ema = loss;
    else ck->macro_ema = 0.999f * ck->macro_ema + 0.001f * loss;
    if (ck->global_step % CHUCK_MACRO_INT == 0 && ck->global_step > CHUCK_WINDOW) {
        if (ck->macro_ema > ck->best_macro * 0.999f) {
            ck->macro_stag++;
            if (ck->macro_stag >= CHUCK_MACRO_PAT) { ck->lr_scale *= CHUCK_MACRO_DECAY; if (ck->lr_scale < 0.15f) ck->lr_scale = 0.15f; ck->macro_stag = 0; ck->macro_drops++; }
        } else { ck->best_macro = ck->macro_ema; ck->macro_stag = 0; }
    }

    /* ═══ Level 4: Activation health (σ) — from tokenizer + parser eyes ═══ */
    ck->sigma = 1.0f;
    if (tok_eye && tok_eye->health < 0.7f) ck->sigma *= tok_eye->health / 0.7f;
    if (parser_eye && parser_eye->health < 0.7f) ck->sigma *= parser_eye->health / 0.7f;

    /* ═══ Level 6: Ψ subjectivity ═══ */
    float gnorm_sq = 0;
    for (int i = 0; i < params->count; i++) for (int j = 0; j < params->tensors[i]->size; j++) gnorm_sq += grads[i][j] * grads[i][j];
    float gnorm = sqrtf(gnorm_sq + 1e-8f);
    ck->psi_w = (chuck_mem_n > 0) ? fminf(CHUCK_PSI_CAP, (float)chuck_mem_n / ((float)chuck_mem_n + CHUCK_PSI_HALF)) : 0;
    float lambda_psi = ck->dampen;
    if (chuck_mem_n > 0) {
        float lp = chuck_mem_recall(loss, gnorm);
        if (lp > 0) { ck->psi = lp - ck->dampen; lambda_psi = ck->dampen + ck->psi_w * ck->psi;
                       if (lambda_psi < CHUCK_DAMP_LO) lambda_psi = CHUCK_DAMP_LO; if (lambda_psi > CHUCK_DAMP_HI) lambda_psi = CHUCK_DAMP_HI; }
    }
    /* Record memory on regime change */
    if (ck->full) {
        ChuckMem snap = { loss, gnorm, ck->dampen, 0 };
        if (ck->global_step % CHUCK_REC_CD == 0) chuck_mem_save(&snap);
    }


    /* ═══ Apply Adam with Chuck modulation ═══ */
    ck->eff_lambda = lambda_psi;   /* psi-recall-adjusted dampen — consumed by the real update step */
    /* ═══ Level 5: adaptive gradient clipping — threshold tracks the running grad-norm EMA ═══ */
    if (ck->gnorm_ema == 0) ck->gnorm_ema = gnorm;
    else ck->gnorm_ema = 0.97f * ck->gnorm_ema + 0.03f * gnorm;
    float adaptive_clip = fmaxf(0.5f, fminf(2.0f, 1.5f * ck->gnorm_ema));
    if (gnorm > 3.0f * ck->gnorm_ema) adaptive_clip *= 0.5f;
    float gn = 0; for (int i = 0; i < params->count; i++) for (int j = 0; j < params->tensors[i]->size; j++) gn += grads[i][j] * grads[i][j]; gn = sqrtf(gn);
    if (gn > adaptive_clip) { float s = adaptive_clip / gn; for (int i = 0; i < params->count; i++) for (int j = 0; j < params->tensors[i]->size; j++) grads[i][j] *= s; }
    /* (2) stagnation-escape noise — kick gradients out of a plateau (ck->noise decays back to 0) */
    if (ck->noise > 1e-6f) for (int i = 0; i < params->count; i++) for (int j = 0; j < params->tensors[i]->size; j++) grads[i][j] += ck->noise * rand_normal();
}

/* Adam optimizer state */
typedef struct { float *m, *v; int size; } AdamS;
typedef struct { AdamS *states; int np; float b1,b2,eps; int t; } Adam;

static Adam *adam_new(ParamList *p) {
    Adam *o = calloc(1, sizeof(Adam)); o->np = p->count; o->states = calloc(p->count, sizeof(AdamS));
    o->b1 = 0.9f; o->b2 = 0.95f; /* was 0.999, fixed 2026-03-05 */ o->eps = 1e-8f;
    for (int i = 0; i < p->count; i++) { int sz = p->tensors[i]->size; o->states[i].m = calloc(sz, 4); o->states[i].v = calloc(sz, 4); o->states[i].size = sz; }
    return o;
}

static void adam_step(Adam *o, ParamList *p, float **g, float lr, float wd) {
    o->t++; float bc1 = 1.0f - powf(o->b1, (float)o->t), bc2 = 1.0f - powf(o->b2, (float)o->t);
    for (int i = 0; i < o->np && i < p->count; i++) {
        Tensor *t = p->tensors[i]; float *gr = g[i]; AdamS *s = &o->states[i];
        if (s->size != t->size) continue;
        int apply_wd = (wd > 0 && t->rows > 1 && i >= 2);
        for (int j = 0; j < t->size; j++) {
            if (apply_wd) t->data[j] -= lr * wd * t->data[j];
            s->m[j] = o->b1 * s->m[j] + (1.0f-o->b1) * gr[j];
            s->v[j] = o->b2 * s->v[j] + (1.0f-o->b2) * gr[j] * gr[j];
            t->data[j] -= lr * (s->m[j]/bc1) / (sqrtf(s->v[j]/bc2) + o->eps);
        }
        tmark_dirty(t); /* GPU copy needs re-upload */
    }
}
static void adam_free(Adam*o){if(!o)return;for(int i=0;i<o->np;i++){free(o->states[i].m);free(o->states[i].v);}free(o->states);free(o);}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA — HF API, parquet, synthetic. same three sources as moe.c.
 * the parser eye judges your data. deal with it.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int hf_extract_texts(const char *json, int json_len, FILE *out) {
    int count = 0;
    const char *p = json, *end = json + json_len;
    while (p < end) {
        p = strstr(p, "\"text\":\"");
        if (!p) break; p += 8;
        const char *start = p;
        while (p < end && !(*p == '"' && *(p-1) != '\\')) p++;
        if (p >= end) break;
        for (const char *s = start; s < p; s++) {
            if (*s == '\\' && s + 1 < p) { s++;
                if (*s == 'n') fputc('\n', out); else if (*s == 't') fputc('\t', out);
                else if (*s == '\\') fputc('\\', out); else if (*s == '"') fputc('"', out);
                else if (*s == 'u' && s + 4 < p) { fputc('?', out); s += 4; }
                else fputc(*s, out);
            } else fputc(*s, out);
        }
        fputc('\n', out); count++; p++;
    }
    return count;
}

/* Snappy decompressor for parquet */
static int snappy_decompress(const uint8_t *src, int slen, uint8_t *dst, int dlen) {
    int si=0,di=0; uint32_t ulen=0; int shift=0;
    while(si<slen){uint8_t b=src[si++];ulen|=(uint32_t)(b&0x7F)<<shift;if(!(b&0x80))break;shift+=7;}
    if((int)ulen>dlen)return -1;
    while(si<slen&&di<(int)ulen){
        uint8_t tag=src[si++];int type=tag&3;
        if(type==0){int len=(tag>>2)+1;if((tag>>2)>=60){int nb=(tag>>2)-59;len=0;for(int i=0;i<nb&&si<slen;i++)len|=src[si++]<<(i*8);len++;}if(si+len>slen||di+len>(int)ulen)return -1;memcpy(dst+di,src+si,len);si+=len;di+=len;}
        else{int len,off;if(type==1){len=((tag>>2)&7)+4;if(si>=slen)return -1;off=((tag>>5)<<8)|src[si++];}else if(type==2){len=(tag>>2)+1;if(si+1>=slen)return -1;off=src[si]|(src[si+1]<<8);si+=2;}else{len=(tag>>2)+1;if(si+3>=slen)return -1;off=src[si]|(src[si+1]<<8)|(src[si+2]<<16)|(src[si+3]<<24);si+=4;}if(off==0||di-off<0)return -1;for(int i=0;i<len;i++)dst[di+i]=dst[di-off+i];di+=len;}
    }
    return di;
}

/* Thrift Compact Protocol decoder */
typedef struct{const uint8_t*data;int pos,len;}TR;
static uint64_t tr_varint(TR*r){uint64_t v=0;int s=0;while(r->pos<r->len){uint8_t b=r->data[r->pos++];v|=(uint64_t)(b&0x7F)<<s;if(!(b&0x80))break;s+=7;}return v;}
static int64_t tr_zigzag(TR*r){uint64_t v=tr_varint(r);return(int64_t)((v>>1)^-(v&1));}
static char*tr_string(TR*r){uint64_t l=tr_varint(r);char*s=malloc(l+1);if(r->pos+(int)l<=r->len){memcpy(s,r->data+r->pos,l);r->pos+=(int)l;}s[l]=0;return s;}
static void tr_skip(TR*r,int type);
static void tr_skip_struct(TR*r){int prev=0;while(r->pos<r->len){uint8_t b=r->data[r->pos++];if(b==0)break;int ft=b&0xF,delta=(b>>4)&0xF;if(delta==0){prev=(int)(int16_t)tr_zigzag(r);}else prev+=delta;tr_skip(r,ft);}}
static void tr_skip(TR*r,int type){switch(type){case 1:case 2:break;case 3:case 4:case 5:case 6:tr_zigzag(r);break;case 7:r->pos+=8;break;case 8:{uint64_t l=tr_varint(r);r->pos+=(int)l;break;}case 9:case 10:{uint8_t h=r->data[r->pos++];int cnt=(h>>4)&0xF,et=h&0xF;if(cnt==0xF)cnt=(int)tr_varint(r);for(int i=0;i<cnt;i++)tr_skip(r,et);break;}case 11:{uint8_t h=r->data[r->pos++];int kt=(h>>4)&0xF,vt=h&0xF;int cnt=(int)tr_varint(r);for(int i=0;i<cnt;i++){tr_skip(r,kt);tr_skip(r,vt);}break;}case 12:tr_skip_struct(r);break;}}

/* Parquet reader */
typedef struct{char*name;int64_t data_off,dict_off,comp_size,nval;int codec;}PqCol;
typedef struct{PqCol*cols;int n;int64_t nrows;}PqMeta;

static PqMeta pq_footer(const uint8_t*f,int64_t sz){
    PqMeta m={0};uint32_t flen=*(uint32_t*)(f+sz-8);TR r={f+sz-8-flen,0,(int)flen};int prev=0;
    while(r.pos<r.len){uint8_t b=r.data[r.pos++];if(b==0)break;int ft=b&0xF,delta=(b>>4)&0xF;int fid=delta?prev+delta:(int)(int16_t)tr_zigzag(&r);prev=fid;
    if(fid==1&&ft==5)tr_zigzag(&r);else if(fid==2&&ft==9){uint8_t h=r.data[r.pos++];int cnt=(h>>4)&0xF;if(cnt==0xF)cnt=(int)tr_varint(&r);for(int i=0;i<cnt;i++)tr_skip_struct(&r);}
    else if(fid==3&&ft==6)m.nrows=(int64_t)tr_zigzag(&r);
    else if(fid==4&&ft==9){uint8_t h=r.data[r.pos++];int rg_cnt=(h>>4)&0xF;if(rg_cnt==0xF)rg_cnt=(int)tr_varint(&r);
    for(int rg=0;rg<rg_cnt;rg++){int rp=0;while(r.pos<r.len){uint8_t rb=r.data[r.pos++];if(rb==0)break;int rt=rb&0xF,rd=(rb>>4)&0xF;int rf=rd?rp+rd:(int)(int16_t)tr_zigzag(&r);rp=rf;
    if(rf==1&&rt==9){uint8_t ch=r.data[r.pos++];int cc=(ch>>4)&0xF;if(cc==0xF)cc=(int)tr_varint(&r);
    for(int ci=0;ci<cc;ci++){PqCol col={0};col.dict_off=-1;int cp=0;
    while(r.pos<r.len){uint8_t cb=r.data[r.pos++];if(cb==0)break;int ct_=cb&0xF,cd_=(cb>>4)&0xF;int cf=cd_?cp+cd_:(int)(int16_t)tr_zigzag(&r);cp=cf;
    if(cf==3&&ct_==12){int mp=0;while(r.pos<r.len){uint8_t mb=r.data[r.pos++];if(mb==0)break;int mt=mb&0xF,md=(mb>>4)&0xF;int mf=md?mp+md:(int)(int16_t)tr_zigzag(&r);mp=mf;
    if(mf==3&&mt==9){uint8_t lh=r.data[r.pos++];int lc=(lh>>4)&0xF;if(lc==0xF)lc=(int)tr_varint(&r);for(int li=0;li<lc;li++){char*s=tr_string(&r);if(li==lc-1)col.name=s;else free(s);}}
    else if(mf==4&&mt==5)col.codec=(int)tr_zigzag(&r);else if(mf==5&&mt==6)col.nval=(int64_t)tr_zigzag(&r);
    else if(mf==7&&mt==6)col.comp_size=(int64_t)tr_zigzag(&r);else if(mf==9&&mt==6)col.data_off=(int64_t)tr_zigzag(&r);
    else if(mf==11&&mt==6)col.dict_off=(int64_t)tr_zigzag(&r);else tr_skip(&r,mt);}}else tr_skip(&r,ct_);}
    m.n++;m.cols=realloc(m.cols,m.n*sizeof(PqCol));m.cols[m.n-1]=col;}}else tr_skip(&r,rt);}}}else tr_skip(&r,ft);}
    return m;
}

typedef struct{int type,comp_sz,uncomp_sz,nval;}PgHdr;
static PgHdr pq_page_hdr(const uint8_t*data,int len,int*hlen){TR r={data,0,len};PgHdr h={0};int prev=0;while(r.pos<r.len){uint8_t b=r.data[r.pos++];if(b==0)break;int ft=b&0xF,delta=(b>>4)&0xF;int fid=delta?prev+delta:(int)(int16_t)tr_zigzag(&r);prev=fid;if(fid==1&&ft==5)h.type=(int)tr_zigzag(&r);else if(fid==2&&ft==5)h.uncomp_sz=(int)tr_zigzag(&r);else if(fid==3&&ft==5)h.comp_sz=(int)tr_zigzag(&r);else if((fid==5||fid==7||fid==8)&&ft==12){int dp=0;while(r.pos<r.len){uint8_t db=r.data[r.pos++];if(db==0)break;int dt=db&0xF,dd=(db>>4)&0xF;int df=dd?dp+dd:(int)(int16_t)tr_zigzag(&r);dp=df;if(df==1&&dt==5)h.nval=(int)tr_zigzag(&r);else tr_skip(&r,dt);}}else tr_skip(&r,ft);}*hlen=r.pos;return h;}

static int pq_extract(const uint8_t*file,int64_t fsz,PqCol*col,FILE*out){
    int64_t pos=(col->dict_off>=0)?col->dict_off:col->data_off;int64_t end=col->data_off+col->comp_size;
    int total=0;char**dict=NULL;int*dlens=NULL,dsz=0;
    while(pos<end&&pos<fsz){int hlen;PgHdr ph=pq_page_hdr(file+pos,(int)(fsz-pos),&hlen);pos+=hlen;if(ph.comp_sz<=0||pos+ph.comp_sz>fsz)break;
    uint8_t*pd;int plen;int nf=0;
    if(col->codec==1){pd=malloc(ph.uncomp_sz);plen=snappy_decompress(file+pos,ph.comp_sz,pd,ph.uncomp_sz);if(plen<0){free(pd);pos+=ph.comp_sz;continue;}nf=1;}else{pd=(uint8_t*)(file+pos);plen=ph.comp_sz;}
    if(ph.type==2){dsz=ph.nval;dict=calloc(dsz,sizeof(char*));dlens=calloc(dsz,sizeof(int));int dp=0;for(int i=0;i<dsz&&dp+4<=plen;i++){int32_t sl=*(int32_t*)(pd+dp);dp+=4;if(dp+sl>plen)break;dict[i]=malloc(sl);memcpy(dict[i],pd+dp,sl);dlens[i]=sl;dp+=sl;}}
    else if(ph.type==0||ph.type==3){int dp=0;
    if(dsz>0){if(dp>=plen)goto nxt;int bw=pd[dp++];for(int v=0;v<ph.nval&&dp<plen;){uint8_t rh=pd[dp++];
    if(rh&1){int count=(rh>>1)*8,bytes=(count*bw+7)/8;uint64_t buf=0;int bb=0,bp=dp;for(int i=0;i<count&&v<ph.nval;i++,v++){while(bb<bw&&bp<dp+bytes&&bp<plen){buf|=(uint64_t)pd[bp++]<<bb;bb+=8;}int idx=(int)(buf&((1ULL<<bw)-1));buf>>=bw;bb-=bw;if(idx>=0&&idx<dsz){fwrite(dict[idx],1,dlens[idx],out);fputc('\n',out);total++;}}dp+=bytes;}
    else{int count=rh>>1,idx=0,nb=(bw+7)/8;for(int b=0;b<nb&&dp<plen;b++)idx|=pd[dp++]<<(b*8);for(int i=0;i<count&&v<ph.nval;i++,v++){if(idx>=0&&idx<dsz){fwrite(dict[idx],1,dlens[idx],out);fputc('\n',out);total++;}}}}}
    else{for(int v=0;v<ph.nval&&dp+4<=plen;v++){int32_t sl=*(int32_t*)(pd+dp);dp+=4;if(sl<0||dp+sl>plen)break;fwrite(pd+dp,1,sl,out);fputc('\n',out);dp+=sl;total++;}}}
    nxt:if(nf)free(pd);pos+=ph.comp_sz;}
    if(dict){for(int i=0;i<dsz;i++)free(dict[i]);free(dict);free(dlens);}return total;
}

static int load_parquet(const char*path,const char*out_path,const char*col_name){
    FILE*f=fopen(path,"rb");if(!f)return -1;fseek(f,0,SEEK_END);int64_t fsz=ftell(f);fseek(f,0,SEEK_SET);
    uint8_t*file=malloc(fsz);fread(file,1,fsz,f);fclose(f);
    if(fsz<12||memcmp(file,"PAR1",4)!=0||memcmp(file+fsz-4,"PAR1",4)!=0){free(file);return -1;}
    PqMeta meta=pq_footer(file,fsz);printf("[parquet] %lld rows, %d column chunks\n",(long long)meta.nrows,meta.n);
    FILE*out=fopen(out_path,"w");if(!out){free(file);return -1;}int total=0;
    for(int i=0;i<meta.n;i++){if(meta.cols[i].name&&strcmp(meta.cols[i].name,col_name)==0)total+=pq_extract(file,fsz,&meta.cols[i],out);}
    fclose(out);for(int i=0;i<meta.n;i++)free(meta.cols[i].name);free(meta.cols);free(file);
    printf("[parquet] extracted %d texts from '%s'\n",total,col_name);return total>0?0:-1;
}

#define HF_BATCH 100
#define HF_PAGES_DEFAULT 50

static int get_data(Config *c) {
    struct stat st;
    if (stat(c->data_path, &st) == 0 && st.st_size > 1000) { printf("[data] found %s (%.1f MB)\n", c->data_path, (float)st.st_size/1048576); return 0; }
    int url_safe = 1;
    for (const char *up = c->data_url; *up; up++) { char ch = *up;
        if (!((ch>='a'&&ch<='z')||(ch>='A'&&ch<='Z')||(ch>='0'&&ch<='9')||ch=='/'||ch=='-'||ch=='_'||ch=='.')) { url_safe = 0; break; } }
    if (c->data_url[0] && url_safe) {
        int hfp = c->hf_pages > 0 ? c->hf_pages : HF_PAGES_DEFAULT;
        printf("[data] fetching %s from HuggingFace (%d pages)...\n", c->data_url, hfp);
        FILE *out = fopen(c->data_path, "w"); if (!out) goto synthetic;
        char tmp[280]; snprintf(tmp, sizeof(tmp), "doe_fetch_%d.json", (int)getpid()); int total = 0;   /* fixed path — never interpolate user --data into a shell command */
        for (int page = 0; page < hfp; page++) {
            const char *cfg = strstr(c->data_url, "fineweb-edu") ? "sample-10BT" : "default";
            char cmd[1024]; snprintf(cmd, sizeof(cmd), "curl -sL 'https://datasets-server.huggingface.co/rows?dataset=%s&config=%s&split=train&offset=%d&length=%d' -o '%s'", c->data_url, cfg, page*HF_BATCH, HF_BATCH, tmp);
            if (system(cmd) != 0) continue;
            if (stat(tmp, &st) != 0 || st.st_size < 500) continue;
            FILE *jf = fopen(tmp, "r"); if (!jf) continue;
            char *json = malloc(st.st_size+1); int jl = (int)fread(json, 1, st.st_size, jf); json[jl] = 0; fclose(jf);
            int n = hf_extract_texts(json, jl, out); free(json); total += n;
            if ((page+1) % 10 == 0) printf("[data] page %d/%d — %d texts so far\n", page+1, hfp, total);
        }
        fclose(out); unlink(tmp);
        if (total > 0) { stat(c->data_path, &st); printf("[data] downloaded %d texts (%.1f MB)\n", total, (float)st.st_size/1048576); return 0; }
    }
    synthetic:
    printf("[data] creating synthetic dataset...\n");
    FILE *f = fopen(c->data_path, "w"); if (!f) return -1;
    const char *s[] = { "The quick brown fox jumps over the lazy dog.", "Machine learning is a subset of artificial intelligence.", "Neural networks learn complex patterns from data.",
        "Mixture of experts routes tokens to specialized networks.", "The parliament votes on which expert handles each token.",
        "Self-aware systems monitor their own performance metrics.", "Harmonic resonance connects experts through frequency space.",
        "Experts are born when demand exceeds capacity.", "Experts die when they stop being useful.", "Democracy in neural networks is underrated.", NULL };
    for (int r = 0; r < 500; r++) for (int i = 0; s[i]; i++) fprintf(f, "%s\n", s[i]);
    fclose(f); return 0;
}

static char *load_text(const char *p, int *len) { FILE *f = fopen(p, "r"); if (!f) { *len = 0; return NULL; } fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc(sz+1); *len = (int)fread(t, 1, sz, f); t[*len] = '\0'; fclose(f); return t; }

/* ═══════════════════════════════════════════════════════════════════════════════
 * GGUF EXPORT — binary format for llama.cpp / grokky.go.
 * exports living experts only. dead ones don't get a tombstone.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void w32(FILE*f,uint32_t v){fwrite(&v,4,1,f);}
static void w64(FILE*f,uint64_t v){fwrite(&v,8,1,f);}
static void wstr(FILE*f,const char*s){uint64_t l=strlen(s);w64(f,l);fwrite(s,1,l,f);}
static void wkv_s(FILE*f,const char*k,const char*v){wstr(f,k);w32(f,8);wstr(f,v);}
static void wkv_u(FILE*f,const char*k,uint32_t v){wstr(f,k);w32(f,4);w32(f,v);}
static void wkv_f(FILE*f,const char*k,float v){wstr(f,k);w32(f,6);fwrite(&v,4,1,f);}
static void wti(FILE*f,const char*name,Tensor*t,uint64_t*off){wstr(f,name);if(t->rows>1){w32(f,2);w64(f,t->cols);w64(f,t->rows);}else{w32(f,1);w64(f,t->size);}w32(f,0);w64(f,*off);*off+=t->size*4;}

static void export_gguf(ModelW *w, Config *c) {
    FILE *f = fopen(c->gguf_path, "wb"); if (!f) { printf("[gguf] failed\n"); return; }
    /* Count alive experts for tensor count */
    int n_expert_tensors = 0;
    for (int l = 0; l < c->depth; l++) for (int e = 0; e < MAX_EXPERTS; e++) if (w->layers[l].experts[e].alive) n_expert_tensors += 3;
    int nt = 3 + c->depth * 7 + n_expert_tensors; /* emb + norm + output + per-layer(norm,qkvo,ffn_norm,w_vote) + expert weights */
    w32(f, 0x46554747); w32(f, 3); w64(f, nt); w64(f, 13);
    wkv_s(f, "general.architecture", "llama"); wkv_s(f, "general.name", "m");
    wkv_u(f, "llama.block_count", c->depth); wkv_u(f, "llama.embedding_length", c->dim);
    wkv_u(f, "llama.attention.head_count", c->n_heads); wkv_u(f, "llama.attention.head_count_kv", c->n_kv_heads);
    wkv_u(f, "llama.feed_forward_length", c->hidden_dim); wkv_u(f, "llama.context_length", c->seq_len);
    wkv_f(f, "llama.attention.layer_norm_rms_epsilon", c->norm_eps); wkv_f(f, "llama.rope.freq_base", c->rope_theta);
    wkv_u(f, "m.max_experts", MAX_EXPERTS); wkv_u(f, "m.initial_experts", c->initial_experts);
    wkv_s(f, "tokenizer.ggml.model", "gpt2");

    uint64_t off = 0;
    wti(f, "token_embd.weight", w->tok_emb, &off);
    wti(f, "output_norm.weight", w->output_norm, &off);
    wti(f, "output.weight", w->output, &off);
    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l]; char n[96];
        snprintf(n, 96, "blk.%d.attn_norm.weight", l); wti(f, n, lw->attn_norm, &off);
        snprintf(n, 96, "blk.%d.attn_q.weight", l); wti(f, n, lw->wq, &off);
        snprintf(n, 96, "blk.%d.attn_k.weight", l); wti(f, n, lw->wk, &off);
        snprintf(n, 96, "blk.%d.attn_v.weight", l); wti(f, n, lw->wv, &off);
        snprintf(n, 96, "blk.%d.attn_output.weight", l); wti(f, n, lw->wo, &off);
        snprintf(n, 96, "blk.%d.ffn_norm.weight", l); wti(f, n, lw->ffn_norm, &off);
        snprintf(n, 96, "blk.%d.ffn_gate_inp.weight", l); wti(f, n, lw->parliament.w_vote, &off);
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (!lw->experts[e].alive) continue;
            snprintf(n, 96, "blk.%d.ffn_gate.%d.weight", l, e); wti(f, n, lw->experts[e].w_gate, &off);
            snprintf(n, 96, "blk.%d.ffn_up.%d.weight", l, e); wti(f, n, lw->experts[e].w_up, &off);
            snprintf(n, 96, "blk.%d.ffn_down.%d.weight", l, e); wti(f, n, lw->experts[e].w_down, &off);
        }
    }
    long p = ftell(f); long al = ((p+31)/32)*32; for (long i = p; i < al; i++) fputc(0, f);
    #define WD(t) fwrite((t)->data, 4, (t)->size, f)
    WD(w->tok_emb); WD(w->output_norm); WD(w->output);
    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l]; WD(lw->attn_norm); WD(lw->wq); WD(lw->wk); WD(lw->wv); WD(lw->wo); WD(lw->ffn_norm); WD(lw->parliament.w_vote);
        for (int e = 0; e < MAX_EXPERTS; e++) { if (!lw->experts[e].alive) continue; WD(lw->experts[e].w_gate); WD(lw->experts[e].w_up); WD(lw->experts[e].w_down); }
    }
    fclose(f);
    struct stat st2; stat(c->gguf_path, &st2);
    printf("[gguf] exported %s (%.1f MB)\n", c->gguf_path, (float)st2.st_size / 1048576);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GGUF LOADER — reverse of export. DOE finds its own weights and remembers.
 * no --load flag needed. the organism recognizes itself.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int load_own_gguf(const char *path, ModelW *w, Config *c) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    /* Read header */
    uint32_t magic; if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) { fclose(f); return 0; }
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t n_tensors, n_kv; fread(&n_tensors, 8, 1, f); fread(&n_kv, 8, 1, f);
    /* Parse KV — verify it's ours */
    int is_ours = 0; int found_depth = 0, found_dim = 0;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t kl; fread(&kl, 8, 1, f); char key[256] = {0};
        if (kl > 255) { fclose(f); return 0; }
        fread(key, 1, kl, f); uint32_t vtype; fread(&vtype, 4, 1, f);
        if (vtype == 8) { /* string */
            uint64_t vl; fread(&vl, 8, 1, f);
            char val[256] = {0}; int rl = vl > 255 ? 255 : (int)vl;
            fread(val, 1, rl, f); if (vl > 255) fseek(f, vl - 255, SEEK_CUR);
            if (strstr(key, "general.name") && strcmp(val, "m") == 0) is_ours = 1;
        } else if (vtype == 4) { /* uint32 */
            uint32_t val; fread(&val, 4, 1, f);
            if (strstr(key, "block_count")) { found_depth = val; }
            if (strstr(key, "embedding_length")) { found_dim = val; }
        } else if (vtype == 6) { /* float32 */
            float val; fread(&val, 4, 1, f);
        } else { fclose(f); return 0; } /* unknown type */
    }
    if (!is_ours) { fclose(f); printf("[load] %s is not ours (name != 'm')\n", path); return 0; }
    if (found_depth != c->depth || found_dim != c->dim) {
        fclose(f); printf("[load] %s dim/depth mismatch (file: d%d dim%d, us: d%d dim%d)\n",
                          path, found_depth, found_dim, c->depth, c->dim); return 0;
    }
    /* Parse tensor info — just get names and offsets */
    typedef struct { char name[96]; uint32_t ndim; uint64_t shape[4]; uint64_t offset; uint64_t nbytes; } TI;
    TI *tinfo = calloc(n_tensors, sizeof(TI));
    if (!tinfo) { fclose(f); return 0; }
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nl; fread(&nl, 8, 1, f);
        if (nl > 95) { free(tinfo); fclose(f); return 0; }
        fread(tinfo[i].name, 1, nl, f); tinfo[i].name[nl] = '\0';
        fread(&tinfo[i].ndim, 4, 1, f);
        uint64_t sz = 1;
        for (uint32_t d = 0; d < tinfo[i].ndim; d++) { fread(&tinfo[i].shape[d], 8, 1, f); sz *= tinfo[i].shape[d]; }
        uint32_t dtype; fread(&dtype, 4, 1, f); /* 0 = f32 */
        fread(&tinfo[i].offset, 8, 1, f);
        tinfo[i].nbytes = sz * 4;
    }
    /* Align to 32 bytes */
    long hdr_end = ftell(f);
    long data_start = ((hdr_end + 31) / 32) * 32;
    /* Atomic guard: every tensor's bytes must fit in the file and token_embd must exist,
     * checked BEFORE any tensor is read into w — so a truncated/partial candidate is
     * rejected without mutating a fresh model. After this passes, no LOAD can short-read. */
    fseek(f, 0, SEEK_END); long fsz = ftell(f);
    if (data_start > fsz) { free(tinfo); fclose(f); return 0; }
    uint64_t region = (uint64_t)(fsz - data_start);   /* bytes available for tensor data */
    int have_emb = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        /* UB-free bounds (all unsigned, no overflow): tensor data must fit in [data_start, fsz) */
        if (tinfo[i].offset > region || tinfo[i].nbytes > region - tinfo[i].offset) { free(tinfo); fclose(f); return 0; }
        if (strcmp(tinfo[i].name, "token_embd.weight") == 0) {
            have_emb = 1;
            if (tinfo[i].nbytes != (uint64_t)w->tok_emb->size * 4) { free(tinfo); fclose(f); return 0; }   /* wrong vocab/dim — not ours */
        }
    }
    if (!have_emb) { free(tinfo); fclose(f); return 0; }
    /* Load tensors by name */
    int loaded = 0;   /* the atomic guard above already proved every tensor fits, so reads can't short-read */
    #define LOAD_T(tensor, tname) do { \
        for (uint64_t _i = 0; _i < n_tensors; _i++) { \
            if (strcmp(tinfo[_i].name, tname) == 0 && (tensor) && (tensor)->size * 4 == (int)tinfo[_i].nbytes) { \
                fseek(f, data_start + tinfo[_i].offset, SEEK_SET); \
                if (fread((tensor)->data, 4, (tensor)->size, f) == (size_t)(tensor)->size) loaded++; break; \
            } \
        } \
    } while(0)
    /* returning variant: 1 only on a FULL read — used to reject truncated/partial experts */
    #define LOAD_TR(tensor, tname) ({ int _ok = 0; \
        for (uint64_t _i = 0; _i < n_tensors; _i++) { \
            if (strcmp(tinfo[_i].name, tname) == 0 && (tensor) && (tensor)->size * 4 == (int)tinfo[_i].nbytes) { \
                fseek(f, data_start + tinfo[_i].offset, SEEK_SET); \
                if (fread((tensor)->data, 4, (tensor)->size, f) == (size_t)(tensor)->size) { loaded++; _ok = 1; } break; \
            } \
        } _ok; })
    int tok_ok = LOAD_TR(w->tok_emb, "token_embd.weight");
    LOAD_T(w->output_norm, "output_norm.weight");
    LOAD_T(w->output, "output.weight");
    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l]; char n[96];
        snprintf(n, 96, "blk.%d.attn_norm.weight", l); LOAD_T(lw->attn_norm, n);
        snprintf(n, 96, "blk.%d.attn_q.weight", l); LOAD_T(lw->wq, n);
        snprintf(n, 96, "blk.%d.attn_k.weight", l); LOAD_T(lw->wk, n);
        snprintf(n, 96, "blk.%d.attn_v.weight", l); LOAD_T(lw->wv, n);
        snprintf(n, 96, "blk.%d.attn_output.weight", l); LOAD_T(lw->wo, n);
        snprintf(n, 96, "blk.%d.ffn_norm.weight", l); LOAD_T(lw->ffn_norm, n);
        snprintf(n, 96, "blk.%d.ffn_gate_inp.weight", l); LOAD_T(lw->parliament.w_vote, n);
        int snap_alive[MAX_EXPERTS] = {0};
        for (int e = 0; e < MAX_EXPERTS; e++) {
            char ng[96], nu[96], nd[96];
            snprintf(ng, 96, "blk.%d.ffn_gate.%d.weight", l, e);
            snprintf(nu, 96, "blk.%d.ffn_up.%d.weight", l, e);
            snprintf(nd, 96, "blk.%d.ffn_down.%d.weight", l, e);
            /* an expert is real only if ALL THREE of its tensors are present */
            int hg = 0, hu = 0, hd = 0;
            for (uint64_t ti = 0; ti < n_tensors; ti++) {
                const char *tn = tinfo[ti].name;
                if (!strcmp(tn, ng)) hg = 1; else if (!strcmp(tn, nu)) hu = 1; else if (!strcmp(tn, nd)) hd = 1;
            }
            if (!(hg && hu && hd)) continue;
            if (!lw->experts[e].alive) {
                float freq = 6.2831853f * e / MAX_EXPERTS;
                init_expert(&lw->experts[e], c->dim, c->hidden_dim, freq, 0);
                lw->experts[e].vitality = 0.8f;
                lw->experts[e].age = 100;
                lw->n_alive++;
            }
            int ok = LOAD_TR(lw->experts[e].w_gate, ng)
                   + LOAD_TR(lw->experts[e].w_up,   nu)
                   + LOAD_TR(lw->experts[e].w_down, nd);
            if (ok == 3) snap_alive[e] = 1;
            else { free_expert(&lw->experts[e]); lw->n_alive--; }   /* partial/truncated — not a real expert */
        }
        /* kill phantoms: experts left alive by fresh init but absent from this snapshot */
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (lw->experts[e].alive && !snap_alive[e]) { free_expert(&lw->experts[e]); lw->n_alive--; }
    }
    #undef LOAD_T
    #undef LOAD_TR
    free(tinfo); fclose(f);
    printf("[load] restored %d tensors from %s — DOE remembers.\n", loaded, path);
    return (tok_ok && loaded >= 2 + 6 * c->depth) ? 1 : 0;   /* require the fixed tensors, not just token_embd */
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MYCELIUM — the GGUF forest. during training, the system grows multiple GGUF
 * snapshots like mushrooms from mycelium. each snapshot captures a different
 * expert configuration (different population, different consensus, different
 * harmonic state). on startup, the system discovers existing snapshots and
 * picks the fittest one. "the forest is the network. the mushrooms are models."
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MYCELIUM_DIR "mycelium"
#define MYCELIUM_MAX 64
#define MYCELIUM_INTERVAL_MIN 1000 /* save snapshot every N steps (at most 5 per training) */

typedef struct {
    char path[256];
    int step;
    int n_experts;
    float loss;
    float consensus;
    float fitness;      /* composite score: lower loss + higher consensus = better */
} MyceliumSpore;

typedef struct {
    MyceliumSpore spores[MYCELIUM_MAX];
    int n_spores;
    int best_idx;       /* index of fittest spore */
} MyceliumState;

static void mycelium_init(MyceliumState *ms) {
    memset(ms, 0, sizeof(MyceliumState));
    ms->best_idx = -1;
    mkdir(MYCELIUM_DIR, 0755);
}

static float mycelium_fitness(float loss, float consensus, int n_experts) {
    /* lower loss = better, higher consensus = clearer decisions, moderate experts = sweet spot */
    float loss_score = 1.0f / (loss + 0.01f);
    float cons_score = consensus;
    float exp_score = 1.0f - fabsf((float)n_experts / MAX_EXPERTS - 0.5f); /* peak at 50% capacity */
    return loss_score * 0.5f + cons_score * 0.3f + exp_score * 0.2f;
}

static void mycelium_save_spore(MyceliumState *ms, ModelW *w, Config *c,
                                 int step, float loss, float consensus) {
    if (ms->n_spores >= MYCELIUM_MAX) {
        /* replace worst spore */
        int worst = 0;
        for (int i = 1; i < ms->n_spores; i++)
            if (ms->spores[i].fitness < ms->spores[worst].fitness) worst = i;
        /* overwrite worst */
        int n_exp = 0;
        for (int l = 0; l < c->depth; l++) n_exp += w->layers[l].n_alive;
        MyceliumSpore *sp = &ms->spores[worst];
        snprintf(sp->path, 256, "%s/m_s%d_e%d_l%.3f.gguf", MYCELIUM_DIR, step, n_exp, loss);
        sp->step = step; sp->n_experts = n_exp; sp->loss = loss; sp->consensus = consensus;
        sp->fitness = mycelium_fitness(loss, consensus, n_exp);
        /* swap gguf_path, export, restore */
        char orig[256]; snprintf(orig, 256, "%s", c->gguf_path);
        snprintf(c->gguf_path, 256, "%s", sp->path);
        export_gguf(w, c);
        snprintf(c->gguf_path, 256, "%s", orig);
        return;
    }
    int n_exp = 0;
    for (int l = 0; l < c->depth; l++) n_exp += w->layers[l].n_alive;
    MyceliumSpore *sp = &ms->spores[ms->n_spores];
    snprintf(sp->path, 256, "%s/m_s%d_e%d_l%.3f.gguf", MYCELIUM_DIR, step, n_exp, loss);
    sp->step = step; sp->n_experts = n_exp; sp->loss = loss; sp->consensus = consensus;
    sp->fitness = mycelium_fitness(loss, consensus, n_exp);
    char orig[256]; snprintf(orig, 256, "%s", c->gguf_path);
    snprintf(c->gguf_path, 256, "%s", sp->path);
    export_gguf(w, c);
    snprintf(c->gguf_path, 256, "%s", orig);
    ms->n_spores++;
    /* update best */
    ms->best_idx = 0;
    for (int i = 1; i < ms->n_spores; i++)
        if (ms->spores[i].fitness > ms->spores[ms->best_idx].fitness) ms->best_idx = i;
    printf("  [mycelium] spore %d: step=%d experts=%d loss=%.4f fitness=%.2f (%d total)\n",
           ms->n_spores-1, step, n_exp, loss, sp->fitness, ms->n_spores);
}

/* Scan mycelium directory for existing spores */
static void mycelium_discover(MyceliumState *ms) {
    char cmd[512];
    snprintf(cmd, 512, "ls %s/m_s*_e*_l*.gguf 2>/dev/null", MYCELIUM_DIR);
    FILE *p = popen(cmd, "r");
    if (!p) return;
    char line[256];
    while (fgets(line, sizeof(line), p) && ms->n_spores < MYCELIUM_MAX) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        MyceliumSpore *sp = &ms->spores[ms->n_spores];
        snprintf(sp->path, 256, "%s", line);
        /* parse metrics from filename: m_s{step}_e{experts}_l{loss}.gguf */
        int s_val = 0, e_val = 0; float l_val = 0;
        char *ps = strstr(line, "_s"); if (ps) s_val = atoi(ps+2);
        char *pe = strstr(line, "_e"); if (pe) e_val = atoi(pe+2);
        char *pl = strstr(line, "_l"); if (pl) l_val = atof(pl+2);
        sp->step = s_val; sp->n_experts = e_val; sp->loss = l_val;
        sp->consensus = 0.5f; /* unknown from filename */
        sp->fitness = mycelium_fitness(l_val, 0.5f, e_val);
        ms->n_spores++;
    }
    pclose(p);
    if (ms->n_spores > 0) {
        ms->best_idx = 0;
        for (int i = 1; i < ms->n_spores; i++)
            if (ms->spores[i].fitness > ms->spores[ms->best_idx].fitness) ms->best_idx = i;
        printf("[mycelium] discovered %d spores. best: %s (fitness=%.2f)\n",
               ms->n_spores, ms->spores[ms->best_idx].path, ms->spores[ms->best_idx].fitness);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * META-LEARNING TRACK — the system learns from its own choices.
 * after each GGUF snapshot, record: what config was used, how well it worked.
 * over time, build a map of "this many experts + this consensus + this data
 * quality = this much loss reduction." the model trains itself on itself.
 * meta-cognition for neural networks. descartes is spinning.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define META_LOG "mycelium/meta.log"
#define META_HIST_CAP 128

typedef struct {
    int step;
    int n_experts;
    float consensus;
    float loss;
    float tok_health;
    float parser_health;
    float harmonic_conf;
    float fitness;
    float delta_loss;  /* loss improvement since last meta entry */
} MetaEntry;

typedef struct {
    MetaEntry history[META_HIST_CAP];
    int n_entries;
    float predicted_fitness; /* what we predicted vs what happened */
    float prediction_error;  /* running error of our meta-predictions */
    float config_bias[4];    /* learned biases: [expert_count, consensus, tok_health, parser_health] */
} MetaTrack;

static void meta_init(MetaTrack *mt) {
    memset(mt, 0, sizeof(MetaTrack));
    mt->config_bias[0] = 0.5f; /* neutral initial biases */
    mt->config_bias[1] = 0.5f;
    mt->config_bias[2] = 0.5f;
    mt->config_bias[3] = 0.5f;
}

static void meta_record(MetaTrack *mt, int step, int n_experts, float consensus,
                        float loss, float tok_health, float parser_health,
                        float harmonic_conf, float prev_loss) {
    if (mt->n_entries >= META_HIST_CAP) {
        /* shift left — drop oldest */
        memmove(mt->history, mt->history + 1, (META_HIST_CAP-1) * sizeof(MetaEntry));
        mt->n_entries = META_HIST_CAP - 1;
    }
    MetaEntry *e = &mt->history[mt->n_entries];
    e->step = step; e->n_experts = n_experts; e->consensus = consensus;
    e->loss = loss; e->tok_health = tok_health; e->parser_health = parser_health;
    e->harmonic_conf = harmonic_conf;
    e->delta_loss = (prev_loss > 0) ? prev_loss - loss : 0;
    e->fitness = mycelium_fitness(loss, consensus, n_experts);
    mt->n_entries++;

    /* Meta-learn: update config biases based on what worked */
    if (mt->n_entries >= 2) {
        MetaEntry *prev = &mt->history[mt->n_entries - 2];
        float improvement = prev->loss - loss;
        float lr_meta = 0.01f;
        /* If loss improved, strengthen biases toward current config */
        /* If loss worsened, weaken them */
        float signal = (improvement > 0) ? 1.0f : -0.5f;
        mt->config_bias[0] += lr_meta * signal * ((float)n_experts / MAX_EXPERTS - 0.5f);
        mt->config_bias[1] += lr_meta * signal * (consensus - 0.5f);
        mt->config_bias[2] += lr_meta * signal * (tok_health - 0.5f);
        mt->config_bias[3] += lr_meta * signal * (parser_health - 0.5f);
        for (int i = 0; i < 4; i++) {
            if (mt->config_bias[i] < 0.01f) mt->config_bias[i] = 0.01f;
            if (mt->config_bias[i] > 0.99f) mt->config_bias[i] = 0.99f;
        }
        /* Track prediction accuracy */
        float pred_err = fabsf(mt->predicted_fitness - e->fitness);
        mt->prediction_error = 0.9f * mt->prediction_error + 0.1f * pred_err;
    }
    /* Predict fitness for next step (used by ephemeral config) */
    mt->predicted_fitness = mycelium_fitness(loss * 0.99f, consensus, n_experts);
}

static void meta_save(MetaTrack *mt) {
    FILE *f = fopen(META_LOG, "a");
    if (!f) return;
    MetaEntry *e = &mt->history[mt->n_entries - 1];
    fprintf(f, "step=%d experts=%d consensus=%.3f loss=%.4f tok=%.2f parser=%.2f harmonic=%.2f fitness=%.2f delta=%.4f bias=[%.2f,%.2f,%.2f,%.2f]\n",
            e->step, e->n_experts, e->consensus, e->loss, e->tok_health, e->parser_health,
            e->harmonic_conf, e->fitness, e->delta_loss,
            mt->config_bias[0], mt->config_bias[1], mt->config_bias[2], mt->config_bias[3]);
    fclose(f);
}




/* ═══════════════════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — temporal self-awareness from AML.
 * inference is the present: ephemeral, no memory, topology born and dying.
 * training is the past: weights persist, experience accumulates.
 * drift is the distance between what the system remembers and what it sees now.
 *
 * every N steps, take a "photograph" of the full system state:
 * expert population, consensus distribution, loss landscape, harmonic spectrum,
 * tokenizer health, parser quality. compare current to historical snapshots.
 * large drift → the world changed → rebuild aggressively.
 * small drift → stable → conserve energy, exploit what works.
 *
 * the AML principle: identity = ε + γ + αδ. the system has a persistent self
 * (γ = weights, training memory) and an ephemeral experience (ε = inference).
 * drift measures how far ε has drifted from γ. when they diverge too much,
 * something fundamental changed. when they converge, the system found itself.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define DRIFT_SNAPSHOTS 32
#define DRIFT_INTERVAL 50 /* snapshot every N steps */
#define DRIFT_DIM 12     /* dimensions of state vector */

typedef struct {
    float state[DRIFT_DIM];
    /* state layout:
     * [0] loss
     * [1] avg_consensus (across layers)
     * [2] n_experts_ratio (n_alive / MAX_EXPERTS, per layer avg)
     * [3] tok_health
     * [4] parser_health
     * [5] harmonic_dominant_freq
     * [6] harmonic_confidence
     * [7] chuck_lambda
     * [8] chuck_sigma
     * [9] chuck_lr_scale
     * [10] avg_vitality (across all alive experts)
     * [11] meta_prediction_error
     */
    int step;
} DriftSnapshot;

typedef struct {
    DriftSnapshot history[DRIFT_SNAPSHOTS];
    int n_snapshots;
    int head;           /* circular buffer pointer */
    float drift;        /* current drift magnitude */
    float drift_ema;    /* exponential moving average of drift */
    float drift_accel;  /* rate of change of drift (second derivative) */
    float stability;    /* 0=chaotic, 1=stable. inverse of drift_ema. */
    /* Temporal spectrum: fourier decomposition of drift history */
    float drift_hist[DRIFT_SNAPSHOTS];
    int drift_hist_len;
} CalendarDrift;

static void drift_init(CalendarDrift *cd) {
    memset(cd, 0, sizeof(CalendarDrift));
    cd->stability = 0.5f;
}

static void drift_snapshot(CalendarDrift *cd, float loss, ModelW *w, Config *c,
                           TokenizerEye *tok_eye, ParserEye *parser_eye,
                           HarmonicState *hs, ChuckState *chuck, MetaTrack *meta) {
    DriftSnapshot *snap = &cd->history[cd->head];

    /* Compute state vector */
    snap->state[0] = loss;

    /* avg consensus across layers */
    float avg_cons = 0;
    for (int l = 0; l < c->depth; l++) avg_cons += w->layers[l].parliament.consensus;
    snap->state[1] = avg_cons / c->depth;

    /* expert ratio */
    float avg_exp = 0;
    for (int l = 0; l < c->depth; l++) avg_exp += (float)w->layers[l].n_alive / MAX_EXPERTS;
    snap->state[2] = avg_exp / c->depth;

    snap->state[3] = tok_eye->health;
    snap->state[4] = parser_eye->health;
    snap->state[5] = hs->dominant_freq;
    snap->state[6] = hs->confidence;
    snap->state[7] = chuck->dampen;
    snap->state[8] = chuck->sigma;
    snap->state[9] = chuck->lr_scale;

    /* avg vitality */
    float avg_vit = 0; int n_alive = 0;
    for (int l = 0; l < c->depth; l++)
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (w->layers[l].experts[e].alive) { avg_vit += w->layers[l].experts[e].vitality; n_alive++; }
    snap->state[10] = (n_alive > 0) ? avg_vit / n_alive : 0;
    snap->state[11] = meta->prediction_error;

    snap->step = chuck->global_step;

    /* Compute drift against previous snapshot */
    if (cd->n_snapshots > 0) {
        int prev_idx = (cd->head - 1 + DRIFT_SNAPSHOTS) % DRIFT_SNAPSHOTS;
        DriftSnapshot *prev = &cd->history[prev_idx];
        float dist_sq = 0;
        for (int i = 0; i < DRIFT_DIM; i++) {
            float d = snap->state[i] - prev->state[i];
            /* normalize by magnitude to make dimensions comparable */
            float scale = fabsf(prev->state[i]) + 0.01f;
            dist_sq += (d / scale) * (d / scale);
        }
        float new_drift = sqrtf(dist_sq / DRIFT_DIM);

        /* drift acceleration = how fast drift is changing */
        float prev_drift = cd->drift;
        cd->drift_accel = new_drift - prev_drift;
        cd->drift = new_drift;
        cd->drift_ema = 0.9f * cd->drift_ema + 0.1f * new_drift;
        cd->stability = 1.0f / (1.0f + cd->drift_ema * 5.0f);
    }

    /* Record drift history for temporal fourier analysis */
    if (cd->drift_hist_len < DRIFT_SNAPSHOTS) cd->drift_hist[cd->drift_hist_len++] = cd->drift;
    else { memmove(cd->drift_hist, cd->drift_hist + 1, (DRIFT_SNAPSHOTS-1) * sizeof(float)); cd->drift_hist[DRIFT_SNAPSHOTS-1] = cd->drift; }

    /* Advance circular buffer */
    cd->head = (cd->head + 1) % DRIFT_SNAPSHOTS;
    if (cd->n_snapshots < DRIFT_SNAPSHOTS) cd->n_snapshots++;
}

/* Compare current state against a specific historical snapshot */

/* Find the historical snapshot most similar to current state */
static int drift_find_resonance(CalendarDrift *cd) {
    if (cd->n_snapshots < 3) return -1;
    float min_dist = 1e30f; int best = -1;
    int curr = (cd->head - 1 + DRIFT_SNAPSHOTS) % DRIFT_SNAPSHOTS;
    for (int i = 2; i < cd->n_snapshots; i++) { /* skip last 2 (too recent) */
        int idx = (cd->head - 1 - i + DRIFT_SNAPSHOTS * 2) % DRIFT_SNAPSHOTS;
        float dist = 0;
        for (int j = 0; j < DRIFT_DIM; j++) {
            float d = cd->history[curr].state[j] - cd->history[idx].state[j];
            float scale = fabsf(cd->history[idx].state[j]) + 0.01f;
            dist += (d / scale) * (d / scale);
        }
        dist = sqrtf(dist / DRIFT_DIM);
        if (dist < min_dist) { min_dist = dist; best = i; }
    }
    return best; /* steps ago when system was in a similar state */
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * EPHEMERAL METRIC-DRIVEN CONFIG — the architecture doesn't care how many
 * weights it has. it adapts. each step, a hash of all metrics determines:
 * - which experts are "hot" (get extra weight)
 * - how many layers are active
 * - vitality threshold for mitosis/apoptosis
 * the unpredictable combination means every run produces a different topology.
 * same weights, different architecture. this is the nicole principle incarnate.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float expert_temperature;  /* modulates parliament vote sharpness */
    float vitality_threshold;  /* adaptive threshold for birth/death */
    int active_layers;         /* how many layers actually execute this step */
    float mitosis_eagerness;   /* how eager the system is to birth new experts */
    float apoptosis_mercy;     /* how merciful toward dying experts */
    uint64_t config_hash;      /* the metric hash that produced this config */
} EphemeralConfig;

static EphemeralConfig ephemeral_compute(TokenizerEye *tok_eye, ParserEye *parser_eye,
                                          HarmonicState *hs, float consensus,
                                          MetaTrack *mt, CalendarDrift *cd, int depth) {
    EphemeralConfig ec;
    /* Hash all metrics into a single chaotic value — including drift */
    uint64_t h = 0x5DEECE66DULL;
    uint32_t bits[8];
    memcpy(&bits[0], &tok_eye->health, 4);
    memcpy(&bits[1], &parser_eye->health, 4);
    memcpy(&bits[2], &hs->dominant_freq, 4);
    memcpy(&bits[3], &consensus, 4);
    memcpy(&bits[4], &mt->config_bias[0], 4);
    memcpy(&bits[5], &mt->prediction_error, 4);
    memcpy(&bits[6], &cd->drift, 4);
    memcpy(&bits[7], &cd->stability, 4);
    for (int i = 0; i < 8; i++) { h ^= (uint64_t)bits[i]; h *= 0x100000001B3ULL; h ^= h >> 17; }
    ec.config_hash = h;

    /* Derive config from hash + metrics + drift */
    float metric_blend = tok_eye->health * 0.25f + parser_eye->health * 0.25f +
                         hs->confidence * 0.15f + consensus * 0.15f + cd->stability * 0.2f;

    /* Expert temperature: higher when confused OR when drift is high (world changed) */
    ec.expert_temperature = 0.5f + 1.0f * (1.0f - consensus);
    if (cd->drift > 0.3f) ec.expert_temperature *= 1.2f; /* more exploration during drift */
    ec.expert_temperature *= (0.5f + mt->config_bias[1]);
    ec.expert_temperature += 0.02f * ((float)((ec.config_hash >> 8) & 0xFF) / 255.0f - 0.5f);   /* hash-driven chaos */
    if (tok_eye->code_ratio > 0.3f) ec.expert_temperature *= 1.1f;   /* code = higher complexity budget */

    /* Active layers: skip some when stable and simple, use all when drifting */
    float complexity = tok_eye->entropy / 5.0f;
    if (complexity > 1.0f) complexity = 1.0f;
    ec.active_layers = depth;
    if (cd->stability > 0.8f && metric_blend > 0.8f && complexity < 0.3f && depth > 2) {
        ec.active_layers = depth - 1; /* skip layer only when truly stable */
    }
    if (cd->drift > 0.5f && depth > 2) {
        ec.active_layers = depth; /* force full depth during high drift */
    }
    if (tok_eye->code_ratio > 0.3f) ec.active_layers = depth;   /* code → use all layers (README) */

    /* Mitosis eagerness: more births when drifting (need to adapt) or data quality high */
    ec.mitosis_eagerness = 0.5f + 0.3f * parser_eye->health * mt->config_bias[0];
    if (cd->drift > 0.2f) ec.mitosis_eagerness += 0.2f * cd->drift; /* drift triggers births */
    if (ec.mitosis_eagerness > 1.0f) ec.mitosis_eagerness = 1.0f;

    /* Apoptosis mercy: more merciful when drifting (don't kill during transition) */
    ec.apoptosis_mercy = 0.5f + 0.3f * (1.0f - metric_blend);
    if (cd->drift > 0.3f) ec.apoptosis_mercy += 0.2f; /* protect experts during drift */
    if (cd->drift_accel > 0.1f) ec.apoptosis_mercy += 0.1f; /* accelerating drift = even more mercy */
    if (ec.apoptosis_mercy > 1.0f) ec.apoptosis_mercy = 1.0f;

    /* Vitality threshold: adaptive based on stability */
    ec.vitality_threshold = 0.1f * (1.0f + ec.apoptosis_mercy);

    return ec;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENVIRONMENT SCANNER — DOE opens its eyes. looks around. counts its resources.
 * what GGUFs are nearby? how much RAM? is there a compiler? can it curl?
 * a model that doesn't know its own filesystem is a model that can't survive.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    char path[256];
    char arch[64];      /* "llama", "grok", "mixtral", "m" — from GGUF metadata */
    int n_layers;
    int dim;
    int n_heads;
    int64_t file_size;
    float compatibility; /* 0..1 — how usable is this GGUF as DOE host */
} DiscoveredGGUF;

typedef struct {
    DiscoveredGGUF ggufs[32];
    int n_ggufs;
    int64_t disk_free;
    int cpu_count;
    int64_t mem_available;
    int has_compiler;
    int has_curl;
    char self_path[256]; /* path to own m.c source */
} Environment;

/* Read GGUF header — just metadata, no tensor loading. peek at the corpse. */
static int gguf_sniff(const char *path, DiscoveredGGUF *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    struct stat st; fstat(fileno(f), &st); out->file_size = st.st_size;
    snprintf(out->path, 256, "%s", path);
    memset(out->arch, 0, 64); out->n_layers = 0; out->dim = 0; out->n_heads = 0;

    uint32_t magic; if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) { fclose(f); return 0; }
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t n_tensors, n_kv; fread(&n_tensors, 8, 1, f); fread(&n_kv, 8, 1, f);

    /* Parse metadata KV pairs — we only care about architecture, dim, layers, heads */
    for (uint64_t i = 0; i < n_kv && i < 64; i++) {
        uint64_t klen; if (fread(&klen, 8, 1, f) != 1) break;
        if (klen > 255) break;
        char key[256]; if (fread(key, 1, klen, f) != klen) break; key[klen] = '\0';
        uint32_t vtype; if (fread(&vtype, 4, 1, f) != 1) break;

        if (vtype == 8) { /* string */
            uint64_t vlen; fread(&vlen, 8, 1, f);
            char val[256]; int rl = vlen < 255 ? (int)vlen : 255;
            fread(val, 1, rl, f); val[rl] = '\0';
            if (vlen > 255) fseek(f, vlen - 255, SEEK_CUR);
            if (strstr(key, "general.architecture")) snprintf(out->arch, 64, "%s", val);
        } else if (vtype == 4) { /* uint32 */
            uint32_t val; fread(&val, 4, 1, f);
            if (strstr(key, "embedding_length")) out->dim = (int)val;
            else if (strstr(key, "block_count")) out->n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) out->n_heads = (int)val;
        } else if (vtype == 5) { /* int32 */
            fseek(f, 4, SEEK_CUR);
        } else if (vtype == 6) { /* float32 */
            fseek(f, 4, SEEK_CUR);
        } else if (vtype == 10) { /* uint64 */
            fseek(f, 8, SEEK_CUR);
        } else if (vtype == 7) { /* bool */
            fseek(f, 1, SEEK_CUR);
        } else {
            break; /* unknown type — stop parsing, we got enough */
        }
    }
    fclose(f);
    return (out->arch[0] != '\0' && out->dim > 0);
}

static void env_scan(Environment *env, const char *self_src, int doe_dim) {
    memset(env, 0, sizeof(Environment));
    snprintf(env->self_path, 256, "%s", self_src);

    /* CPU count */
    env->cpu_count = (int)sysconf(_SC_NPROCESSORS_ONLN);

    /* Memory */
#ifdef __linux__
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    env->mem_available = (int64_t)pages * page_size;
#elif defined(__APPLE__)
    int64_t mem = 0; size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    env->mem_available = mem;
#endif

    /* Disk free */
#ifdef __linux__
    struct statvfs sv; if (statvfs(".", &sv) == 0) env->disk_free = (int64_t)sv.f_bavail * sv.f_frsize;
#elif defined(__APPLE__)
    struct statfs sf; if (statfs(".", &sf) == 0) env->disk_free = (int64_t)sf.f_bavail * sf.f_bsize;
#endif

    /* Check for compiler and curl — the tools of survival */
    env->has_compiler = (system("which cc >/dev/null 2>&1") == 0);
    env->has_curl = (system("which curl >/dev/null 2>&1") == 0);

    /* Scan for GGUFs — find . -name '*.gguf' -maxdepth 3 */
    FILE *p = popen("find . -name '*.gguf' -maxdepth 3 2>/dev/null", "r");
    if (p) {
        char line[256];
        while (fgets(line, sizeof(line), p) && env->n_ggufs < 32) {
            int len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
            if (len == 0) continue;
            DiscoveredGGUF dg;
            if (gguf_sniff(line, &dg)) {
                /* Compatibility: dim match = best, dim multiple = ok, else = poor */
                if (dg.dim == doe_dim) dg.compatibility = 1.0f;
                else if (doe_dim > 0 && dg.dim % doe_dim == 0) dg.compatibility = 0.7f;
                else if (doe_dim > 0 && abs(dg.dim - doe_dim) < 128) dg.compatibility = 0.5f;
                else dg.compatibility = 0.1f;
                env->ggufs[env->n_ggufs++] = dg;
            }
        }
        pclose(p);
    }

    printf("[env] cpu=%d mem=%.1fGB disk=%.1fGB compiler=%s curl=%s ggufs=%d\n",
           env->cpu_count,
           (float)env->mem_available / (1024*1024*1024),
           (float)env->disk_free / (1024*1024*1024),
           env->has_compiler ? "yes" : "no",
           env->has_curl ? "yes" : "no",
           env->n_ggufs);

    for (int i = 0; i < env->n_ggufs; i++) {
        DiscoveredGGUF *g = &env->ggufs[i];
        printf("  [env] gguf: %s arch=%s dim=%d layers=%d size=%.1fMB compat=%.1f\n",
               g->path, g->arch, g->dim, g->n_layers,
               (float)g->file_size / (1024*1024), g->compatibility);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GGUF HOST — DOE indexes a bigger model. wraps it with LoRA. modulates it.
 * DOE reads the host read-only. LoRA experts provide direction.
 * the host provides weights. DOE provides topology.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define LORA_RANK 16
#define LORA_MAX_LAYERS 32
#define HOST_INIT_EXPERTS 4   /* alive experts seeded per host layer (parliament start size) */

/* ═══════════════════════════════════════════════════════════════════════════════
 * PACKED QUANTIZED MATVEC — kernels INLINED from github.com/ariannamethod/notorch (nt_qmatvec).
 * NO in-repo copy of notorch — single-file by design (cc janusdoe.c). The kernels below are
 * diff-verified mathematically identical to notorch's (jd_pq_q4_k_rows == nt_q4_k_rows,
 * jd_pq_q6_k_rows == nt_q6_k_rows; Codex-audited 2026-06-17).
 * Keeps host weights PACKED in RAM and dequantizes each block inline in registers (no f32
 * blow-up): RAM ×2-3 + less weight bandwidth — the llama.cpp mechanism. dtype = GGUF code
 * (1 f16, 2 Q4_0, 6 Q5_0, 8 Q8_0, 12 Q4_K, 14 Q6_K). Row kernels take [r0,r1) so they thread.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float jd_f16_to_f32(uint16_t h) {
    uint32_t sign=(h>>15)&1, exp=(h>>10)&0x1F, mant=h&0x3FF, f;
    if (exp==0) {
        if (mant==0) f=sign<<31;
        else { exp=1; while(!(mant&0x400)){mant<<=1;exp--;} mant&=0x3FF; f=(sign<<31)|((exp+127-15)<<23)|(mant<<13); }
    } else if (exp==31) f=(sign<<31)|0x7F800000|(mant<<13);
    else f=(sign<<31)|((exp+127-15)<<23)|(mant<<13);
    float r; memcpy(&r,&f,4); return r;
}
#define JD_Q4_0_BYTES 18
#define JD_Q8_0_BYTES 34
#define JD_Q5_0_BYTES 22
#define JD_Q4_K_BYTES 144
#define JD_Q6_K_BYTES 210
static void jd_smk4(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j<4){*s=sc[j]&63;*m=sc[j+4]&63;} else {*s=(sc[j+4]&0x0F)|((sc[j-4]>>6)<<4);*m=(sc[j+4]>>4)|((sc[j]>>6)<<4);}
}
/* full-tensor dequant (one-time, for tok_emb + norms only — never matvec weights) */
static void jd_dequant_q4_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nb=n/32;
    for (uint64_t i=0;i<nb;i++){ const uint8_t *b=data+i*JD_Q4_0_BYTES; float d=jd_f16_to_f32(b[0]|(b[1]<<8));
        for (int j=0;j<16;j++){ out[i*32+j]=(float)((b[2+j]&0x0F)-8)*d; out[i*32+j+16]=(float)((b[2+j]>>4)-8)*d; } }
}
static void jd_dequant_q8_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nb=n/32;
    for (uint64_t i=0;i<nb;i++){ const uint8_t *b=data+i*JD_Q8_0_BYTES; float d=jd_f16_to_f32(b[0]|(b[1]<<8));
        for (int j=0;j<32;j++) out[i*32+j]=(float)((int8_t)b[2+j])*d; }
}
static void jd_dequant_q5_0(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nb=n/32;
    for (uint64_t i=0;i<nb;i++){ const uint8_t *b=data+i*JD_Q5_0_BYTES; float d=jd_f16_to_f32(b[0]|(b[1]<<8));
        uint32_t qh=b[2]|(b[3]<<8)|(b[4]<<16)|(b[5]<<24); const uint8_t *qs=b+6;
        for (int j=0;j<16;j++){ int lo=qs[j]&0x0F,hi=qs[j]>>4,h0=(qh>>j)&1,h1=(qh>>(j+16))&1;
            out[i*32+j]=(float)((lo|(h0<<4))-16)*d; out[i*32+j+16]=(float)((hi|(h1<<4))-16)*d; } }
}
static void jd_dequant_q4_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nb=n/256;
    for (uint64_t i=0;i<nb;i++){ const uint8_t *b=data+i*JD_Q4_K_BYTES;
        float d=jd_f16_to_f32(b[0]|(b[1]<<8)), dmin=jd_f16_to_f32(b[2]|(b[3]<<8));
        const uint8_t *sc=b+4,*qs=b+16; int is=0,qi=0,oi=(int)(i*256);
        for (int j=0;j<256;j+=64){ uint8_t s0,m0,s1,m1; jd_smk4(is,sc,&s0,&m0); jd_smk4(is+1,sc,&s1,&m1);
            float d1=d*s0,mm1=dmin*m0,d2=d*s1,mm2=dmin*m1;
            for (int l=0;l<32;l++) out[oi+j+l]=d1*(float)(qs[qi+l]&0x0F)-mm1;
            for (int l=0;l<32;l++) out[oi+j+32+l]=d2*(float)(qs[qi+l]>>4)-mm2;
            qi+=32; is+=2; } }
}
static void jd_dequant_q6_k(const uint8_t *data, float *out, uint64_t n) {
    uint64_t nb=n/256;
    for (uint64_t i=0;i<nb;i++){ const uint8_t *b=data+i*JD_Q6_K_BYTES; const uint8_t *ql=b,*qh=b+128,*sc=b+192;
        float d=jd_f16_to_f32(b[208]|(b[209]<<8)); int oi=(int)(i*256);
        for (int n128=0;n128<2;n128++){ const uint8_t *qlp=ql+n128*64,*qhp=qh+n128*32,*scp=sc+n128*8; int yo=oi+n128*128;
            for (int l=0;l<32;l++){ int is=l/16;
                int q1=(qlp[l]&0x0F)|((qhp[l]>>0)&3)<<4, q2=(qlp[l+32]&0x0F)|((qhp[l]>>2)&3)<<4;
                int q3=(qlp[l]>>4)|((qhp[l]>>4)&3)<<4, q4=(qlp[l+32]>>4)|((qhp[l]>>6)&3)<<4;
                out[yo+l+0]=d*(float)((int8_t)scp[is+0])*(float)(q1-32);
                out[yo+l+32]=d*(float)((int8_t)scp[is+2])*(float)(q2-32);
                out[yo+l+64]=d*(float)((int8_t)scp[is+4])*(float)(q3-32);
                out[yo+l+96]=d*(float)((int8_t)scp[is+6])*(float)(q4-32); } } }
}
static uint64_t jd_quant_bytes(uint32_t dt, uint64_t n) {
    switch(dt){ case 0:return n*4; case 1:return n*2; case 2:return (n/32)*JD_Q4_0_BYTES;
        case 6:return (n/32)*JD_Q5_0_BYTES; case 8:return (n/32)*JD_Q8_0_BYTES;
        case 12:return (n/256)*JD_Q4_K_BYTES; case 14:return (n/256)*JD_Q6_K_BYTES; default:return 0; }
}
static int jd_dequant_any(const uint8_t *src, float *out, uint64_t n, int dt) {
    if (dt==1){ const uint16_t *h=(const uint16_t*)src; for (uint64_t i=0;i<n;i++) out[i]=jd_f16_to_f32(h[i]); return 0; }
    if (dt==2){ jd_dequant_q4_0(src,out,n); return 0; }
    if (dt==6){ jd_dequant_q5_0(src,out,n); return 0; }
    if (dt==8){ jd_dequant_q8_0(src,out,n); return 0; }
    if (dt==12){ jd_dequant_q4_k(src,out,n); return 0; }
    if (dt==14){ jd_dequant_q6_k(src,out,n); return 0; }
    return -1;
}
/* packed row kernels — inline dequant, weights stay packed (== notorch nt_qmatvec) */
static void jd_pq_f16_rows(float *out,const uint8_t *W,const float *x,int r0,int r1,int c){
    const uint16_t *Wh=(const uint16_t*)W;
    for (int row=r0;row<r1;row++){ const uint16_t *r=Wh+(size_t)row*c; float a=0; for (int j=0;j<c;j++) a+=jd_f16_to_f32(r[j])*x[j]; out[row]=a; }
}
static void jd_pq_q4_0_rows(float *out,const uint8_t *W,const float *x,int r0,int r1,int c){
    int nb=c/32;
    for (int row=r0;row<r1;row++){ const uint8_t *rb=W+(size_t)row*nb*18; float a=0;
        for (int b=0;b<nb;b++){ const uint8_t *bl=rb+(size_t)b*18; float d=jd_f16_to_f32(bl[0]|(bl[1]<<8)); const float *xb=x+(size_t)b*32;
            for (int i=0;i<16;i++){ int lo=(int)(bl[2+i]&0x0F)-8,hi=(int)(bl[2+i]>>4)-8; a+=d*lo*xb[i]; a+=d*hi*xb[i+16]; } }
        out[row]=a; }
}
static void jd_pq_q8_0_rows(float *out,const uint8_t *W,const float *x,int r0,int r1,int c){
    int nb=c/32;
    for (int row=r0;row<r1;row++){ const uint8_t *rb=W+(size_t)row*nb*34; float a=0;
        for (int b=0;b<nb;b++){ const uint8_t *bl=rb+(size_t)b*34; float d=jd_f16_to_f32(bl[0]|(bl[1]<<8)); const float *xb=x+(size_t)b*32;
            for (int i=0;i<32;i++) a+=d*(float)(int8_t)bl[2+i]*xb[i]; }
        out[row]=a; }
}
static void jd_pq_q5_0_rows(float *out,const uint8_t *W,const float *x,int r0,int r1,int c){
    int nb=c/32;
    for (int row=r0;row<r1;row++){ const uint8_t *rb=W+(size_t)row*nb*22; float a=0;
        for (int b=0;b<nb;b++){ const uint8_t *bl=rb+(size_t)b*22; float d=jd_f16_to_f32(bl[0]|(bl[1]<<8));
            uint32_t qh=(uint32_t)bl[2]|((uint32_t)bl[3]<<8)|((uint32_t)bl[4]<<16)|((uint32_t)bl[5]<<24);
            const uint8_t *qs=bl+6; const float *xb=x+(size_t)b*32;
            for (int j=0;j<16;j++){ int lo=qs[j]&0x0F,hi=qs[j]>>4,h0=(qh>>j)&1,h1=(qh>>(j+16))&1;
                a+=d*(float)((lo|(h0<<4))-16)*xb[j]; a+=d*(float)((hi|(h1<<4))-16)*xb[j+16]; } }
        out[row]=a; }
}
static void jd_pq_q4_k_rows(float *out,const uint8_t *W,const float *x,int r0,int r1,int c){
    int nb=c/256;
    for (int row=r0;row<r1;row++){ const uint8_t *rb=W+(size_t)row*nb*144; float a=0;
        for (int b=0;b<nb;b++){ const uint8_t *bl=rb+(size_t)b*144;
            float d=jd_f16_to_f32(bl[0]|(bl[1]<<8)), dmin=jd_f16_to_f32(bl[2]|(bl[3]<<8));
            const uint8_t *sc=bl+4,*qs=bl+16; const float *xb=x+(size_t)b*256; int is=0,qi=0;
            for (int j=0;j<256;j+=64){ uint8_t s0,m0,s1,m1; jd_smk4(is,sc,&s0,&m0); jd_smk4(is+1,sc,&s1,&m1);
                float d1=d*s0,mm1=dmin*m0,d2=d*s1,mm2=dmin*m1;
                for (int l=0;l<32;l++) a+=(d1*(float)(qs[qi+l]&0x0F)-mm1)*xb[j+l];
                for (int l=0;l<32;l++) a+=(d2*(float)(qs[qi+l]>>4)-mm2)*xb[j+32+l];
                qi+=32; is+=2; } }
        out[row]=a; }
}
static void jd_pq_q6_k_rows(float *out,const uint8_t *W,const float *x,int r0,int r1,int c){
    int nb=c/256;
    for (int row=r0;row<r1;row++){ const uint8_t *rb=W+(size_t)row*nb*210; float a=0;
        for (int b=0;b<nb;b++){ const uint8_t *bl=rb+(size_t)b*210,*ql=bl,*qh=bl+128;
            const int8_t *sc=(const int8_t*)(bl+192); float d=jd_f16_to_f32(bl[208]|(bl[209]<<8));
            const float *xb=x+(size_t)b*256;
            for (int n=0;n<256;n+=128){ const uint8_t *qlh=ql+(n/128)*64,*qhh=qh+(n/128)*32; const int8_t *sch=sc+(n/128)*8;
                for (int l=0;l<32;l++){ int is=l/16;
                    int q1=(int)((qlh[l]&0x0F)|(((qhh[l]>>0)&3)<<4))-32, q2=(int)((qlh[l+32]&0x0F)|(((qhh[l]>>2)&3)<<4))-32;
                    int q3=(int)((qlh[l]>>4)|(((qhh[l]>>4)&3)<<4))-32, q4=(int)((qlh[l+32]>>4)|(((qhh[l]>>6)&3)<<4))-32;
                    a+=d*sch[is+0]*q1*xb[n+l]; a+=d*sch[is+2]*q2*xb[n+l+32];
                    a+=d*sch[is+4]*q3*xb[n+l+64]; a+=d*sch[is+6]*q4*xb[n+l+96]; } } }
        out[row]=a; }
}
typedef void (*jd_pq_fn)(float*,const uint8_t*,const float*,int,int,int);
static jd_pq_fn jd_pq_for(int dt,int c){
    switch(dt){ case 1:return jd_pq_f16_rows; case 2:return (c%32)?NULL:jd_pq_q4_0_rows;
        case 6:return (c%32)?NULL:jd_pq_q5_0_rows; case 8:return (c%32)?NULL:jd_pq_q8_0_rows;
        case 12:return (c%256)?NULL:jd_pq_q4_k_rows; case 14:return (c%256)?NULL:jd_pq_q6_k_rows; }
    return NULL;
}
static int jd_nthreads(void){ static int n=0; if(!n){ long c=sysconf(_SC_NPROCESSORS_ONLN); n=(c>0)?(int)c:1; if(n>32)n=32; } return n; }
typedef struct { jd_pq_fn fn; float *out; const uint8_t *Wq; const float *x; int r0,r1,c; } JdPQWork;
static void *jd_pq_worker(void *arg){ JdPQWork *w=(JdPQWork*)arg; w->fn(w->out,w->Wq,w->x,w->r0,w->r1,w->c); return NULL; }
/* out[r] = Wq[r,c] @ x[c], weights packed. Returns 0 ok, -1 if dtype unsupported. */
static int jd_qmatvec(float *out,const uint8_t *Wq,int dt,const float *x,int r,int c){
    jd_pq_fn fn=jd_pq_for(dt,c); if(!fn) return -1;
    int nt=jd_nthreads(); if(nt>r)nt=r;
    if (nt<=1 || (long)r*c<(1L<<20)){ fn(out,Wq,x,0,r,c); return 0; }
    pthread_t thr[32]; JdPQWork work[32]; int chunk=(r+nt-1)/nt,actual=0;
    for (int t=0;t<nt;t++){ int r0=t*chunk,r1=r0+chunk; if(r0>=r)break; if(r1>r)r1=r;
        work[t]=(JdPQWork){fn,out,Wq,x,r0,r1,c}; pthread_create(&thr[t],NULL,jd_pq_worker,&work[t]); actual++; }
    for (int t=0;t<actual;t++) pthread_join(thr[t],NULL);
    return 0;
}
/* int8 dynamic-activation-quant fast path (Q4_0 only), env JD_INT8=1. APPROXIMATE. */
static void jd_quant_act_q8(const float *x,int c,int8_t *qa,float *da){
    int nb=c/32;
    for (int b=0;b<nb;b++){ const float *xb=x+(size_t)b*32; float amax=0; for (int i=0;i<32;i++){ float v=fabsf(xb[i]); if(v>amax)amax=v; }
        float d=amax/127.0f,id=(d>0)?1.0f/d:0.0f; da[b]=d;
        for (int i=0;i<32;i++){ int q=(int)lrintf(xb[i]*id); if(q>127)q=127; else if(q<-127)q=-127; qa[(size_t)b*32+i]=(int8_t)q; } }
}
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
static void jd_pq_q4_0_rows_i8(float *out,const uint8_t *W,const int8_t *qa,const float *da,int r0,int r1,int c){
    int nb=c/32; const uint8x16_t m0f=vdupq_n_u8(0x0F); const int8x16_t e8=vdupq_n_s8(8);
    for (int row=r0;row<r1;row++){ const uint8_t *rb=W+(size_t)row*nb*18; float a=0;
        for (int b=0;b<nb;b++){ const uint8_t *bl=rb+(size_t)b*18; float d=jd_f16_to_f32(bl[0]|(bl[1]<<8));
            const int8_t *qab=qa+(size_t)b*32; uint8x16_t pk=vld1q_u8(bl+2);
            int8x16_t lo=vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pk,m0f)),e8), hi=vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pk,4)),e8);
            int32x4_t s=vdupq_n_s32(0); s=vdotq_s32(s,lo,vld1q_s8(qab)); s=vdotq_s32(s,hi,vld1q_s8(qab+16));
            a+=d*da[b]*(float)vaddvq_s32(s); }
        out[row]=a; }
}
#else
static void jd_pq_q4_0_rows_i8(float *out,const uint8_t *W,const int8_t *qa,const float *da,int r0,int r1,int c){
    int nb=c/32;
    for (int row=r0;row<r1;row++){ const uint8_t *rb=W+(size_t)row*nb*18; float a=0;
        for (int b=0;b<nb;b++){ const uint8_t *bl=rb+(size_t)b*18; float d=jd_f16_to_f32(bl[0]|(bl[1]<<8));
            const int8_t *qab=qa+(size_t)b*32; int32_t s=0;
            for (int i=0;i<16;i++){ int lo=(int)(bl[2+i]&0x0F)-8,hi=(int)(bl[2+i]>>4)-8; s+=lo*qab[i]; s+=hi*qab[i+16]; }
            a+=d*da[b]*(float)s; }
        out[row]=a; }
}
#endif
static int jd_qmatvec_i8(float *out,const uint8_t *Wq,int dt,const float *x,int r,int c){
    if (dt!=2||(c%32)) return -1;
    int nb=c/32; int8_t *qa=(int8_t*)malloc((size_t)c); float *da=(float*)malloc((size_t)nb*sizeof(float));
    if(!qa||!da){ free(qa); free(da); return -1; }
    jd_quant_act_q8(x,c,qa,da); jd_pq_q4_0_rows_i8(out,Wq,qa,da,0,r,c); free(qa); free(da); return 0;
}
/* the one call host forward uses: packed (dt!=0) -> jd_qmatvec; f32 -> existing matvec.
 * packed-but-unsupported -> loud abort (never reinterpret packed bytes as f32). */
static int g_jd_int8=-1;
static void jd_mv(float *out,const float *W,int dt,const float *x,int r,int c){
    if (dt){
        if (g_jd_int8<0){ const char *e=getenv("JD_INT8"); g_jd_int8=(e&&e[0]=='1')?1:0; }
        if (g_jd_int8 && jd_qmatvec_i8(out,(const uint8_t*)W,dt,x,r,c)==0) return;
        if (jd_qmatvec(out,(const uint8_t*)W,dt,x,r,c)==0) return;
        fprintf(stderr,"[doe] FATAL: jd_qmatvec failed for packed dtype %d (r=%d c=%d); refusing to read packed bytes as f32\n",dt,r,c);
        abort();
    }
    matvec(out,(float*)W,(float*)x,r,c);
}

/* LoRA expert — Delta Voice (out += α·A@(B@x)) that modulates the host, not replaces it.
 * Parliament-ready: experts vote per token (election lands in increment 2), and will live /
 * die (mitosis / apoptosis) later. Increment 1 seeds exactly one alive expert per layer. */
typedef struct {
    float *A;            /* [dim, rank] — output projection */
    float *B;            /* [rank, dim] — input projection */
    int    tokens_seen;  /* elected-this-window counter (drives the post-forward learn pass) */
    int    alive;        /* apoptosis flag (poisoned experts are killed) */
} HostExpert;

static int arch_rope_norm(const char *arch);   /* fwd-decl: defined near host RoPE, used at index-time print */
static int g_rope_override = -1;               /* CLI --rope-neox/--rope-norm: -1=auto(arch), 0=NEOX, 1=NORM */

typedef struct {
    /* Host model — mmap'd from GGUF, read-only */
    uint8_t *mmap_base;
    size_t mmap_size;
    int host_n_layers, host_dim, host_hidden, host_heads, host_kv_heads, host_head_dim;
    int host_vocab;
    float *host_tok_emb;    /* [vocab, dim] — dequantized to f32 at load (random row lookup) */
    float *host_output;     /* [vocab, dim] — kept PACKED if quantized (big lm_head matvec) */
    float *host_norm;       /* [dim] */
    int   host_output_dt;   /* GGUF dtype of host_output (0 = f32 / dequantized) */
    float rope_theta;       /* RoPE base (default 10000; Mistral/Qwen 1e6) */
    float rms_eps;          /* RMSNorm epsilon (default 1e-6) */
    char  host_arch[64];    /* general.architecture from GGUF — gates RoPE pairing (NORM vs NEOX) */
    struct {
        float *wq, *wk, *wv, *wo;
        float *ffn_gate, *ffn_up, *ffn_down;
        float *attn_norm, *ffn_norm;
        /* packed GGUF dtype per matvec weight (0 = f32; else kept packed, dequant inline) */
        int wq_dt, wk_dt, wv_dt, wo_dt, ffn_gate_dt, ffn_up_dt, ffn_down_dt;
    } layers[LORA_MAX_LAYERS];
    /* malloc'd f32 buffers from one-time dequant (tok_emb / norms) — freed on teardown */
    float *dq_bufs[LORA_MAX_LAYERS*3];
    int    n_dq_bufs;

    /* DOE's LoRA overlays — a parliament of experts per layer, ≥1 alive. Delta Voice:
     * out += alpha * Σ_e w_e · A_e @ (B_e @ x), w_e from the per-token election. */
    HostExpert experts[LORA_MAX_LAYERS][MAX_EXPERTS];
    float *w_vote[LORA_MAX_LAYERS];     /* [MAX_EXPERTS, dim] — per-expert vote row */
    float  consensus[LORA_MAX_LAYERS];  /* EMA of vote peakedness (0=chaos, 1=unanimous) */
    float attention_bias[LORA_MAX_LAYERS];  /* per-layer attn scaling hook (Meta-Arianna) — read in forward, currently constant */
    float layer_focus[LORA_MAX_LAYERS];     /* per-layer residual scaling */
    float lora_alpha;

    int active;
    char host_path[256];
    /* Host tokenizer (from GGUF metadata) — for real text generation through the host */
    char **host_vocab_tokens;   /* token strings, indexed by id (NULL if not parsed) */
    int    host_ntok;           /* count of host_vocab_tokens */
    int    host_bos, host_eos;
    /* Host hebbian plasticity (faithful notorch port — post-forward, NOT in-forward, NOT backprop) */
    float  host_learn_lr;       /* plasticity rate */
    float *host_x, *host_dy;    /* captured by forward: final hidden (x) + last FFN-down residual (dy) */
} GGUFHost;

/* Effective RoPE pairing: CLI override (--rope-neox/--rope-norm) wins, else arch heuristic.
 * Some GGUFs are arch=llama but NeoX-laid-out (e.g. custom personality exports whose Q/K
 * were not permuted by llama.cpp), so the arch heuristic alone is insufficient — the
 * override lets any layout run correctly regardless of what general.architecture claims. */
static int host_rope_norm(const GGUFHost *ps) {
    return g_rope_override >= 0 ? g_rope_override : arch_rope_norm(ps->host_arch);
}

/* Parse GGUF tensor info entry */
typedef struct { char name[96]; uint32_t ndim; uint64_t dims[4]; uint32_t dtype; uint64_t offset; } GGUFTensorInfo;

/* Resolve a host tensor to a usable pointer. Matvec weights (want_f32=0) with a
 * supported packed dtype stay PACKED (pointer into mmap, *out_dt=dt → jd_mv inline-dequant).
 * tok_emb / norms (want_f32=1, random/element-wise access) and unsupported dtypes are
 * dequantized once into a tracked f32 buffer. Bounds-checked against mmap end. NULL on fail. */
static float *jd_host_resolve(GGUFHost *ps, const uint8_t *raw, const uint8_t *pend,
                              int dt, uint64_t n_elems, int c, int want_f32, int *out_dt) {
    *out_dt = 0;
    uint64_t bytes = jd_quant_bytes((uint32_t)dt, n_elems);
    if (dt != 0 && bytes == 0) return NULL;            /* unknown / unsupported dtype */
    if (bytes == 0) bytes = n_elems * 4;               /* dt==0 f32 */
    if (raw + bytes > pend) return NULL;               /* tensor runs past mmap */
    if (dt == 0) return (float*)raw;                   /* native f32 — point into mmap */
    if (!want_f32 && jd_pq_for(dt, c)) { *out_dt = dt; return (float*)raw; }  /* keep packed */
    /* must be f32 — dequant once into a tracked buffer */
    float *buf = malloc(n_elems * sizeof(float));
    if (!buf) return NULL;
    if (jd_dequant_any(raw, buf, n_elems, dt) != 0) { free(buf); return NULL; }
    if (ps->n_dq_bufs < (int)(sizeof(ps->dq_bufs)/sizeof(ps->dq_bufs[0]))) ps->dq_bufs[ps->n_dq_bufs++] = buf;
    else { free(buf); return NULL; }
    return buf;
}

static void gguf_host_free(GGUFHost *ps);  /* fwd-decl: unified cleanup for every error path */

static int gguf_host_load(GGUFHost *ps, const char *path, int doe_dim) {
    (void)doe_dim;   /* compat-only param (host dims come from the GGUF, not the trainer) */
    memset(ps, 0, sizeof(GGUFHost));
    snprintf(ps->host_path, 256, "%s", path);
    ps->lora_alpha = 0.1f;
    ps->host_bos = ps->host_eos = -1;   /* -1 = absent; id 0 is a valid bos/eos */
    GGUFTensorInfo *tinfo = NULL;       /* function-scope: every bail path frees it, no leak */

    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st; fstat(fd, &st);
    ps->mmap_size = st.st_size;
    ps->mmap_base = mmap(NULL, ps->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ps->mmap_base == MAP_FAILED) { ps->mmap_base = NULL; return 0; }

    /* Parse GGUF header from mmap'd data */
    uint8_t *p = ps->mmap_base;
    uint8_t *pend = ps->mmap_base + ps->mmap_size;
    #define PCHECK(n) do { if (p + (n) > pend) goto bail; } while(0)
    PCHECK(4); uint32_t magic = *(uint32_t*)p; p += 4;
    if (magic != 0x46554747) goto bail;
    PCHECK(4); uint32_t ver = *(uint32_t*)p; p += 4; (void)ver;
    PCHECK(8); uint64_t n_tensors = *(uint64_t*)p; p += 8;
    PCHECK(8); uint64_t n_kv = *(uint64_t*)p; p += 8;

    /* Parse metadata — robust value-skip across ALL GGUF types (arrays included) so p
     * stays byte-aligned to the tensor-info table. The old toy parser broke on the
     * tokenizer array (type 9) and desynced → "GGUF parse failed" on any real model. */
    for (uint64_t i = 0; i < n_kv; i++) {
        PCHECK(8); uint64_t klen = *(uint64_t*)p; p += 8;
        char key[256];
        if (klen > 255) { PCHECK(klen); p += klen; key[0] = '\0'; }                  /* skip oversized key, keep aligned */
        else { PCHECK(klen); memcpy(key, p, klen); key[klen] = '\0'; p += klen; }
        PCHECK(4); uint32_t vtype = *(uint32_t*)p; p += 4;
        if (vtype == 8) { /* string */
            PCHECK(8); uint64_t vlen = *(uint64_t*)p; p += 8; PCHECK(vlen);
            if (strstr(key, "general.architecture") && vlen < 64) {
                memcpy(ps->host_arch, p, vlen); ps->host_arch[vlen] = 0;
            }
            p += vlen;
        } else if (vtype == 4) { /* uint32 */
            PCHECK(4); uint32_t val = *(uint32_t*)p; p += 4;
            if (strstr(key, "embedding_length")) ps->host_dim = (int)val;
            else if (strstr(key, "block_count")) ps->host_n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) ps->host_heads = (int)val;
            else if (strstr(key, "head_count_kv")) ps->host_kv_heads = (int)val;
            else if (strstr(key, "feed_forward_length")) ps->host_hidden = (int)val;
            else if (strstr(key, "vocab_size")) ps->host_vocab = (int)val;
            else if (strstr(key, "attention.key_length")) ps->host_head_dim = (int)val;
            else if (strstr(key, "bos_token_id")) ps->host_bos = (int)val;
            else if (strstr(key, "eos_token_id")) ps->host_eos = (int)val;
        } else if (vtype == 6) { /* float32 */
            PCHECK(4); float fv; memcpy(&fv, p, 4); p += 4;
            if (strstr(key, "rope.freq_base")) ps->rope_theta = fv;
            else if (strstr(key, "rms_epsilon")) ps->rms_eps = fv;
        } else if (vtype == 0 || vtype == 1 || vtype == 7) { PCHECK(1); p += 1; }      /* uint8/int8/bool */
        else if (vtype == 2 || vtype == 3) { PCHECK(2); p += 2; }                       /* uint16/int16 */
        else if (vtype == 5) { PCHECK(4); p += 4; }                                     /* int32 */
        else if (vtype == 10 || vtype == 11 || vtype == 12) { PCHECK(8); p += 8; }      /* uint64/int64/float64 */
        else if (vtype == 9) { /* array — skip the whole payload */
            PCHECK(4); uint32_t atype = *(uint32_t*)p; p += 4;
            PCHECK(8); uint64_t alen = *(uint64_t*)p; p += 8;
            if (atype == 8) { /* array of strings */
                int is_tokens = (strstr(key, "tokenizer.ggml.tokens") != NULL);
                if (is_tokens && alen > 0 && alen < 1000000 && !ps->host_vocab_tokens) {
                    ps->host_vocab_tokens = calloc(alen, sizeof(char*));
                    if (ps->host_vocab_tokens) ps->host_ntok = (int)alen;
                }
                for (uint64_t ai = 0; ai < alen; ai++) {
                    PCHECK(8); uint64_t slen = *(uint64_t*)p; p += 8;
                    if (slen > 1000000) goto bail; PCHECK(slen);
                    if (is_tokens && ps->host_vocab_tokens && ai < (uint64_t)ps->host_ntok) {
                        ps->host_vocab_tokens[ai] = malloc(slen + 1);
                        if (ps->host_vocab_tokens[ai]) { memcpy(ps->host_vocab_tokens[ai], p, slen); ps->host_vocab_tokens[ai][slen] = '\0'; }
                    }
                    p += slen;
                }
            } else {
                size_t esz = (atype==0||atype==1||atype==7)?1:(atype==2||atype==3)?2:
                             (atype==4||atype==5||atype==6)?4:(atype==10||atype==11||atype==12)?8:0;
                if (esz == 0) goto bail;                                                 /* unknown array elem type */
                PCHECK(alen * esz); p += alen * esz;
            }
        } else { PCHECK(4); p += 4; }                                                    /* unknown scalar — guess 4B */
    }

    if (ps->host_dim == 0 || ps->host_dim > 16384 || ps->host_n_layers == 0) goto bail;
    if (ps->host_heads == 0) ps->host_heads = ps->host_dim / 64;
    if (ps->host_kv_heads == 0) ps->host_kv_heads = ps->host_heads;
    if (ps->host_head_dim == 0) ps->host_head_dim = ps->host_dim / ps->host_heads; /* prefer parsed key_length (Mistral hd=128 ≠ dim/heads) */
    if (ps->host_hidden == 0) ps->host_hidden = ps->host_dim * 4;
    if (ps->rope_theta == 0.0f) ps->rope_theta = 10000.0f;
    if (ps->rms_eps == 0.0f) ps->rms_eps = 1e-6f;

    /* Parse tensor info entries — find offsets for each weight */
    if (n_tensors > 10000) goto bail; /* sanity check */
    tinfo = calloc(n_tensors, sizeof(GGUFTensorInfo));
    if (!tinfo) goto bail;
    for (uint64_t i = 0; i < n_tensors; i++) {
        PCHECK(8); uint64_t nlen = *(uint64_t*)p; p += 8;
        if (nlen > 256) goto bail;
        int nl = nlen < 95 ? (int)nlen : 95;
        PCHECK(nlen); memcpy(tinfo[i].name, p, nl); tinfo[i].name[nl] = '\0'; p += nlen;
        PCHECK(4); tinfo[i].ndim = *(uint32_t*)p; p += 4;
        if (tinfo[i].ndim > 4) goto bail;
        for (uint32_t d = 0; d < tinfo[i].ndim; d++) { PCHECK(8); tinfo[i].dims[d] = *(uint64_t*)p; p += 8; }
        PCHECK(4); tinfo[i].dtype = *(uint32_t*)p; p += 4;
        PCHECK(8); tinfo[i].offset = *(uint64_t*)p; p += 8;
    }

    /* Align to 32 bytes — tensor data starts here */
    uint64_t header_size = p - ps->mmap_base;
    uint64_t data_start = ((header_size + 31) / 32) * 32;

    /* Map tensor names to weight pointers — packed-aware (matvec weights stay quantized,
     * dequantized inline by jd_mv; tok_emb + norms dequantized once to f32). */
    for (uint64_t i = 0; i < n_tensors; i++) {
        int dt = (int)tinfo[i].dtype;
        const uint8_t *raw = ps->mmap_base + data_start + tinfo[i].offset;
        uint64_t ne = 1; for (uint32_t d = 0; d < tinfo[i].ndim; d++) ne *= tinfo[i].dims[d];
        int c = tinfo[i].ndim > 0 ? (int)tinfo[i].dims[0] : 0;  /* inner/contraction dim */
        char *n = tinfo[i].name;
        int wdt = 0;

        if (strcmp(n, "token_embd.weight") == 0) {
            ps->host_tok_emb = jd_host_resolve(ps, raw, pend, dt, ne, c, 1, &wdt);
            if (ps->host_vocab == 0) ps->host_vocab = (int)tinfo[i].dims[1];
        }
        else if (strcmp(n, "output_norm.weight") == 0) ps->host_norm = jd_host_resolve(ps, raw, pend, dt, ne, c, 1, &wdt);
        else if (strcmp(n, "output.weight") == 0) ps->host_output = jd_host_resolve(ps, raw, pend, dt, ne, c, 0, &ps->host_output_dt);
        else {
            int l = -1; sscanf(n, "blk.%d.", &l);
            if (l >= 0 && l < LORA_MAX_LAYERS && l < ps->host_n_layers) {
                if (strstr(n, "attn_q.weight")) ps->layers[l].wq = jd_host_resolve(ps, raw, pend, dt, ne, c, 0, &ps->layers[l].wq_dt);
                else if (strstr(n, "attn_k.weight")) ps->layers[l].wk = jd_host_resolve(ps, raw, pend, dt, ne, c, 0, &ps->layers[l].wk_dt);
                else if (strstr(n, "attn_v.weight")) ps->layers[l].wv = jd_host_resolve(ps, raw, pend, dt, ne, c, 0, &ps->layers[l].wv_dt);
                else if (strstr(n, "attn_output.weight")) ps->layers[l].wo = jd_host_resolve(ps, raw, pend, dt, ne, c, 0, &ps->layers[l].wo_dt);
                else if (strstr(n, "ffn_gate.weight") && !strstr(n, "ffn_gate_inp")) ps->layers[l].ffn_gate = jd_host_resolve(ps, raw, pend, dt, ne, c, 0, &ps->layers[l].ffn_gate_dt);
                else if (strstr(n, "ffn_up.weight")) ps->layers[l].ffn_up = jd_host_resolve(ps, raw, pend, dt, ne, c, 0, &ps->layers[l].ffn_up_dt);
                else if (strstr(n, "ffn_down.weight")) ps->layers[l].ffn_down = jd_host_resolve(ps, raw, pend, dt, ne, c, 0, &ps->layers[l].ffn_down_dt);
                else if (strstr(n, "attn_norm.weight")) ps->layers[l].attn_norm = jd_host_resolve(ps, raw, pend, dt, ne, c, 1, &wdt);
                else if (strstr(n, "ffn_norm.weight")) ps->layers[l].ffn_norm = jd_host_resolve(ps, raw, pend, dt, ne, c, 1, &wdt);
            }
        }
    }
    free(tinfo); tinfo = NULL;

    /* Tied embeddings: no separate output.weight → lm_head shares token_embd (f32). */
    if (!ps->host_output && ps->host_tok_emb) { ps->host_output = ps->host_tok_emb; ps->host_output_dt = 0; }

    /* Verify minimum viable host — can't attach to a headless corpse */
    if (!ps->host_tok_emb || !ps->host_output || !ps->host_norm) {
        printf("[doe] host GGUF missing essential weights (tok=%p out=%p norm=%p). abandoning.\n",
               (void*)ps->host_tok_emb, (void*)ps->host_output, (void*)ps->host_norm);
        gguf_host_free(ps); return 0;   /* unified cleanup — frees dq_bufs, parliament, vocab, mmap */
    }
    /* Skip MoE GGUFs — can't index ourselves */
    int has_standard_ffn = 0;
    for (int l = 0; l < ps->host_n_layers && l < LORA_MAX_LAYERS; l++)
        if (ps->layers[l].ffn_gate && ps->layers[l].ffn_up && ps->layers[l].ffn_down) has_standard_ffn = 1;
    if (!has_standard_ffn) {
        printf("[doe] host has no standard FFN layers (MoE?). needs a plain transformer.\n");
        gguf_host_free(ps); return 0;   /* unified cleanup — frees dq_bufs, parliament, vocab, mmap */
    }

    /* Allocate LoRA parliament — seed HOST_INIT_EXPERTS alive experts per layer (each its own
     * Xavier Delta Voice + a vote row), the rest dead. The per-token election picks a weighted
     * subset; experts will later be born (mitosis) and die (apoptosis). */
    int ld = ps->host_dim;
    int n0 = HOST_INIT_EXPERTS; if (n0 > MAX_EXPERTS) n0 = MAX_EXPERTS; if (n0 < 1) n0 = 1;
    float scale = 0.02f / sqrtf((float)LORA_RANK);
    float vote_scale = 0.02f / sqrtf((float)ld);
    for (int l = 0; l < ps->host_n_layers && l < LORA_MAX_LAYERS; l++) {
        ps->w_vote[l] = calloc((size_t)MAX_EXPERTS * ld, 4);
        if (!ps->w_vote[l]) { gguf_host_free(ps); return 0; }  /* OOM */
        for (int i = 0; i < MAX_EXPERTS * ld; i++) ps->w_vote[l][i] = rand_normal() * vote_scale;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            HostExpert *ex = &ps->experts[l][e];
            ex->tokens_seen = 0;
            if (e < n0) {
                ex->A = calloc(ld * LORA_RANK, 4);
                ex->B = calloc(LORA_RANK * ld, 4);
                if (!ex->A || !ex->B) { gguf_host_free(ps); return 0; }  /* OOM */
                for (int j = 0; j < ld * LORA_RANK; j++) ex->A[j] = rand_normal() * scale;
                for (int j = 0; j < LORA_RANK * ld; j++) ex->B[j] = rand_normal() * scale;
                ex->alive = 1;
            } else { ex->A = ex->B = NULL; ex->alive = 0; }
        }
        ps->consensus[l] = 0.5f;
        ps->attention_bias[l] = 0.0f;  /* reserved Meta-Arianna hook — read in forward, currently constant (not yet learned) */
        ps->layer_focus[l] = 1.0f;     /* reserved hook — constant 1.0 (no scaling); learning not wired */
    }

    ps->host_x = calloc(ld, 4); ps->host_dy = calloc(ld, 4);
    if (!ps->host_x || !ps->host_dy) { gguf_host_free(ps); return 0; }   /* OOM */
    ps->host_learn_lr = 0.01f;
    ps->active = 1;
    printf("[doe] indexed %s (arch=%s rope=%s dim=%d layers=%d heads=%d vocab=%d, %.1fMB)\n",
           path, ps->host_arch[0] ? ps->host_arch : "?",
           host_rope_norm(ps) ? "NORM" : "NEOX",
           ps->host_dim, ps->host_n_layers, ps->host_heads,
           ps->host_vocab, (float)ps->mmap_size / (1024*1024));
    printf("[doe] LoRA rank=%d alpha=%.2f\n", LORA_RANK, ps->lora_alpha);
    #undef PCHECK
    return 1;
bail:
    free(tinfo);          /* local tensor-info table — NULL-safe on every early bail */
    gguf_host_free(ps);   /* frees any partial parliament / vocab / dq_bufs + munmaps */
    printf("[doe] GGUF parse failed.\n");
    return 0;
}

/* Parliament election — port of doe.c parliament_elect, host CPU. vote_e = dot(w_vote_e, xn);
 * consensus = EMA of normalized vote std; k = n_alive·(1−consensus) clamped to [2, n_alive];
 * top-k experts by vote, softmax-weighted. Fills selected[k] + weights[k] (Σ=1), returns k.
 * With one alive expert returns k=1, weight 1 (degenerates to a single Delta Voice). */
static int host_parliament_elect(GGUFHost *ps, int l, const float *xn, int D,
                                 int *selected, float *weights) {
    int alive_idx[MAX_EXPERTS], na = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (ps->experts[l][e].alive) alive_idx[na++] = e;
    if (na == 0) return 0;
    if (na == 1) { selected[0] = alive_idx[0]; weights[0] = 1.0f; return 1; }

    float votes[MAX_EXPERTS];
    for (int i = 0; i < na; i++) {
        int e = alive_idx[i]; const float *row = ps->w_vote[l] + (size_t)e * D;
        float d = 0; for (int j = 0; j < D; j++) d += row[j] * xn[j];
        votes[e] = isfinite(d) ? d : -1e30f;   /* finite-safe: a NaN/Inf vote can't poison the softmax */
    }
    float mean = 0; for (int i = 0; i < na; i++) mean += votes[alive_idx[i]]; mean /= na;
    float var = 0; for (int i = 0; i < na; i++) { float dd = votes[alive_idx[i]] - mean; var += dd*dd; } var /= na;
    float cons = fminf(1.0f, sqrtf(var + 1e-8f) / (fabsf(mean) + 1.0f));
    ps->consensus[l] = 0.9f * ps->consensus[l] + 0.1f * cons;

    int k = (int)(na * (1.0f - ps->consensus[l]));
    if (k < 2) k = 2; if (k > na) k = na;

    int used[MAX_EXPERTS] = {0};
    for (int ki = 0; ki < k; ki++) {
        float bv = -1e30f; int bi = -1;
        for (int i = 0; i < na; i++) { int e = alive_idx[i]; if (!used[e] && votes[e] > bv) { bv = votes[e]; bi = e; } }
        if (bi < 0) for (int i = 0; i < na; i++) { int e = alive_idx[i]; if (!used[e]) { bi = e; break; } } /* all-tied/NaN → next unused */
        selected[ki] = bi; weights[ki] = votes[bi]; used[bi] = 1;
    }
    float mx = weights[0]; for (int i = 1; i < k; i++) if (weights[i] > mx) mx = weights[i];
    float sum = 0; for (int i = 0; i < k; i++) { weights[i] = expf(weights[i] - mx); sum += weights[i]; }
    if (sum > 0.0f && isfinite(sum)) { for (int i = 0; i < k; i++) weights[i] /= sum; }
    else { for (int i = 0; i < k; i++) weights[i] = 1.0f / k; }  /* degenerate → uniform, never NaN */
    return k;
}

/* Prophecy debt (port of doe.c compute_prophecy_debt) — bounded [0,1) miss measure: how far the
 * actual next token's logit sits below the argmax. 0 = predicted, →1 = badly missed. */
static float host_prophecy_debt(const float *logits, int chosen, int n) {
    if (n <= 0 || chosen < 0 || chosen >= n) return 0.0f;
    float mx = logits[0]; for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float diff = mx - logits[chosen];
    return diff > 0.0f ? diff / (diff + 1.0f) : 0.0f;
}

/* Host notorch step — FAITHFUL port of doe.c notorch_step (2260-2299). Hebbian Oja plasticity, NOT
 * backprop. Stability comes from: signal clamped ±2; a SMALL channel u = B·dy + tiny noise (NOT
 * B@xn); only NOTORCH_RANK lanes updated per call (rotating); A learns toward x, B toward dy
 * (distinct buffers). A=[D,rank], B=[rank,D]. */
static int host_notorch_offset = 0;
static void host_notorch_step(float *A, float *B, int D, int rank,
                              const float *x, const float *dy, float signal, float lr_base) {
    if (fabsf(signal) < 1e-8f) return;
    if (signal >  2.0f) signal =  2.0f;
    if (signal < -2.0f) signal = -2.0f;
    float lr = lr_base * signal;
    int nr = NOTORCH_RANK; if (nr > rank) nr = rank;
    int base = host_notorch_offset % rank;
    float u[NOTORCH_RANK];
    for (int j = 0; j < nr; j++) {
        int r = (base + j) % rank;
        float s = 0; for (int i = 0; i < D; i++) s += B[r*D + i] * dy[i];
        s += rand_normal() * 0.01f;
        /* clamp channel: bounds the Oja anti-decay (1+lr·u²) so a row aligning with dy under a
         * negative signal cannot grow exponentially (the confirmed cause of the explosion). */
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        u[j] = s;
    }
    for (int j = 0; j < nr; j++) {
        int r = (base + j) % rank; float ur = u[j];
        for (int i = 0; i < D; i++) { float *a = &A[i*rank + r]; *a += lr*ur*(x[i]  - ur*(*a)); }
        for (int i = 0; i < D; i++) { float *b = &B[r*D + i];   *b += lr*ur*(dy[i] - ur*(*b)); }
    }
    host_notorch_offset = (base + nr) % rank;
    /* Soft norm constraint — keep the expert bounded (RMS<=cap) so it ADAPTS without exploding and
     * stays alive. This is the real stabilizer; quarantine is only the last-resort poison catch. */
    int n = D * rank; const float cap = 0.05f;   /* gentle modulation: experts nudge the host, don't override it */
    double sa = 0, sb = 0; for (int i = 0; i < n; i++) { sa += (double)A[i]*A[i]; sb += (double)B[i]*B[i]; }
    float ra = (float)sqrt(sa / n), rb = (float)sqrt(sb / n);
    if (ra > cap) { float f = cap / ra; for (int i = 0; i < n; i++) A[i] *= f; }
    if (rb > cap) { float f = cap / rb; for (int i = 0; i < n; i++) B[i] *= f; }
}

/* Quarantine — full scan (not just A[0]/B[0]): nonfinite, per-element blow-up, or aggregate RMS far
 * past a sane expert cap ⇒ poisoned (caller kills the expert). Init RMS ≈ 0.005; healthy ≈ 0.05. */
static int host_lora_poisoned(const float *A, const float *B, int D, int rank) {
    int n = D * rank; double s2 = 0; float mx = 0;
    for (int i = 0; i < n; i++) { float a = A[i]; if (!isfinite(a)) return 1; float ab = fabsf(a); if (ab > mx) mx = ab; s2 += (double)a*a; }
    for (int i = 0; i < n; i++) { float b = B[i]; if (!isfinite(b)) return 1; float ab = fabsf(b); if (ab > mx) mx = ab; s2 += (double)b*b; }
    if (mx > 8.0f) return 1;
    if (sqrt(s2 / (2.0 * n)) > 5.0) return 1;   /* true blow-up only; the notorch RMS-cap (0.05) keeps healthy experts far under this */
    return 0;
}

/* RoPE pairing per architecture, mirroring llama.cpp llama_model_rope_type()
 * (src/llama-model.cpp): llama-family + Mistral use NORM (adjacent pairs 2i,2i+1);
 * qwen/falcon/gemma/phi/... use NEOX (offset-half pairs i,i+hd/2). GGUF permutes Q/K
 * for whichever the arch expects, so the wrong mode is identity at pos 0 but corrupts
 * rope progressively at pos>0 (coherent short, broken long). Ported from canon doe 9477c4f. */
static int arch_rope_norm(const char *arch) {
    return strcmp(arch, "llama") == 0 || strcmp(arch, "mistral") == 0 ||
           strcmp(arch, "mistral3") == 0 || strcmp(arch, "mistral4") == 0;
}

/* Host forward — run input through indexed model with DOE's LoRA injection.
 * returns logits over host vocabulary. */
static void gguf_host_forward(GGUFHost *ps, int token, int pos, float *out_logits,
                              float *kv_cache_k, float *kv_cache_v, int max_seq) {
    int D = ps->host_dim, H = ps->host_heads, HD = ps->host_head_dim;
    int KVH = ps->host_kv_heads, HG = H / KVH;

    /* Pre-allocate per-layer scratch ONCE and reuse across layers — one OOM check, no per-token
     * malloc churn. `att` (size pos+1, per head) stays per-head. */
    int qd = H * HD, kd = KVH * HD, HH = ps->host_hidden;
    float *x = calloc(D, 4), *lora_tmp = calloc(LORA_RANK, 4);
    float *xn = calloc(D, 4), *q = calloc(qd, 4), *k = calloc(kd, 4), *v = calloc(kd, 4);
    float *attn_out = calloc(qd, 4), *ao = calloc(D, 4), *fn = calloc(D, 4);
    float *gate = calloc(HH, 4), *up = calloc(HH, 4), *ffn_out = calloc(D, 4);
    if (!x || !lora_tmp || !xn || !q || !k || !v || !attn_out || !ao || !fn || !gate || !up || !ffn_out) {
        free(x); free(lora_tmp); free(xn); free(q); free(k); free(v);
        free(attn_out); free(ao); free(fn); free(gate); free(up); free(ffn_out);
        for (int i = 0; i < ps->host_vocab; i++) out_logits[i] = 0.0f; return;   /* OOM — neutral logits */
    }
    if (token < ps->host_vocab) memcpy(x, ps->host_tok_emb + token * D, D * 4);

    for (int l = 0; l < ps->host_n_layers && l < LORA_MAX_LAYERS; l++) {
        if (!ps->layers[l].wq) continue;

        /* RMSNorm → xn */
        float ss = 0; for (int i = 0; i < D; i++) ss += x[i] * x[i];
        ss = 1.0f / sqrtf(ss / D + ps->rms_eps);
        if (ps->layers[l].attn_norm)
            for (int i = 0; i < D; i++) xn[i] = x[i] * ss * ps->layers[l].attn_norm[i];
        else
            for (int i = 0; i < D; i++) xn[i] = x[i] * ss;

        /* QKV projections (packed-aware: jd_mv dequants inline if weight is quantized) */
        jd_mv(q, ps->layers[l].wq, ps->layers[l].wq_dt, xn, qd, D);
        jd_mv(k, ps->layers[l].wk, ps->layers[l].wk_dt, xn, kd, D);
        jd_mv(v, ps->layers[l].wv, ps->layers[l].wv_dt, xn, kd, D);

        /* RoPE on Q and K — arch-gated pairing (NORM=adjacent 2i,2i+1 for llama/mistral;
         * NEOX=offset-half i,i+hd/2 for qwen/gemma/...). Wrong mode corrupts rope at pos>0.
         * Inline — no precomputed cache in host mode. */
        int rope_norm = host_rope_norm(ps);
        for (int h = 0; h < H; h++) {
            float *qh = q + h * HD; int half = HD / 2;
            for (int i = 0; i < half; i++) {
                float ang = (float)pos / powf(ps->rope_theta, 2.0f * i / (float)HD);
                float co = cosf(ang), si = sinf(ang);
                int a = rope_norm ? 2*i : i, b = rope_norm ? 2*i+1 : i + half;
                float x0 = qh[a], x1 = qh[b];
                qh[a] = x0 * co - x1 * si; qh[b] = x0 * si + x1 * co;
            }
        }
        for (int h = 0; h < KVH; h++) {
            float *kh = k + h * HD; int half = HD / 2;
            for (int i = 0; i < half; i++) {
                float ang = (float)pos / powf(ps->rope_theta, 2.0f * i / (float)HD);
                float co = cosf(ang), si = sinf(ang);
                int a = rope_norm ? 2*i : i, b = rope_norm ? 2*i+1 : i + half;
                float x0 = kh[a], x1 = kh[b];
                kh[a] = x0 * co - x1 * si; kh[b] = x0 * si + x1 * co;
            }
        }

        /* KV cache */
        int kv_off_k = l * max_seq * kd + pos * kd;
        int kv_off_v = l * max_seq * kd + pos * kd;
        memcpy(kv_cache_k + kv_off_k, k, kd * 4);
        memcpy(kv_cache_v + kv_off_v, v, kd * 4);

        /* Attention with DOE bias */
        float bias = 1.0f + ps->attention_bias[l];
        float sc = 1.0f / sqrtf((float)HD);
        memset(attn_out, 0, (size_t)qd * 4);   /* accumulated below */
        for (int h = 0; h < H; h++) {
            int kvh = h / HG;
            float *qh = q + h * HD;
            float *att = calloc(pos + 1, 4);
            if (!att) break;   /* OOM — skip remaining heads (graceful, no deref) */
            for (int t = 0; t <= pos; t++) {
                float *kt = kv_cache_k + l * max_seq * kd + t * kd + kvh * HD;
                float d = 0; for (int i = 0; i < HD; i++) d += qh[i] * kt[i];
                att[t] = d * sc * bias; /* DOE modulates attention strength */
            }
            /* Softmax */
            float mx = att[0]; for (int t = 1; t <= pos; t++) if (att[t] > mx) mx = att[t];
            float se = 0; for (int t = 0; t <= pos; t++) { att[t] = expf(att[t] - mx); se += att[t]; }
            for (int t = 0; t <= pos; t++) att[t] /= se;
            /* Weighted sum */
            float *oh = attn_out + h * HD;
            for (int t = 0; t <= pos; t++) {
                float a = att[t];
                float *vt = kv_cache_v + l * max_seq * kd + t * kd + kvh * HD;
                for (int i = 0; i < HD; i++) oh[i] += a * vt[i];
            }
            free(att);
        }

        /* Output projection + residual */
        memset(ao, 0, (size_t)D * 4);
        if (ps->layers[l].wo) jd_mv(ao, ps->layers[l].wo, ps->layers[l].wo_dt, attn_out, D, qd);
        for (int i = 0; i < D; i++) x[i] += ao[i];

        /* FFN with LoRA injection */
        ss = 0; for (int i = 0; i < D; i++) ss += x[i] * x[i];
        ss = 1.0f / sqrtf(ss / D + ps->rms_eps);
        if (ps->layers[l].ffn_norm)
            for (int i = 0; i < D; i++) fn[i] = x[i] * ss * ps->layers[l].ffn_norm[i];
        else
            for (int i = 0; i < D; i++) fn[i] = x[i] * ss;

        /* SwiGLU FFN (if host has it) */
        if (ps->layers[l].ffn_gate && ps->layers[l].ffn_up && ps->layers[l].ffn_down) {
            jd_mv(gate, ps->layers[l].ffn_gate, ps->layers[l].ffn_gate_dt, fn, HH, D);
            jd_mv(up,   ps->layers[l].ffn_up,   ps->layers[l].ffn_up_dt,   fn, HH, D);
            for (int i = 0; i < HH; i++) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i]; /* SwiGLU */
            jd_mv(ffn_out, ps->layers[l].ffn_down, ps->layers[l].ffn_down_dt, gate, D, HH);
            for (int i = 0; i < D; i++) x[i] += ffn_out[i] * ps->layer_focus[l];
            if (ps->host_dy) {  /* learning dy = last FFN-down residual, RMS-normalized so the notorch channel u=B·dy stays bounded */
                double s = 0; for (int i = 0; i < D; i++) s += (double)ffn_out[i]*ffn_out[i];
                float inv = 1.0f / (sqrtf((float)(s / D)) + 1e-8f);
                for (int i = 0; i < D; i++) ps->host_dy[i] = ffn_out[i] * inv;
            }
        }

        /* LoRA parliament injection: the election picks k experts; inject their softmax-weighted
         * Delta Voices: x += alpha * Σ_{e∈selected} w_e · A_e @ (B_e @ xn). */
        {
            int selected[MAX_EXPERTS]; float weights[MAX_EXPERTS];
            int k = host_parliament_elect(ps, l, xn, D, selected, weights);
            for (int ki = 0; ki < k; ki++) {
                HostExpert *ex = &ps->experts[l][selected[ki]];
                if (!ex->alive || !ex->A || !ex->B) continue;
                float w = ps->lora_alpha * weights[ki];
                memset(lora_tmp, 0, LORA_RANK * 4);
                for (int r = 0; r < LORA_RANK; r++) {
                    float s = 0;
                    for (int j = 0; j < D; j++) s += ex->B[r*D+j] * xn[j];
                    lora_tmp[r] = s;
                }
                for (int i = 0; i < D; i++) {
                    float s = 0;
                    for (int r = 0; r < LORA_RANK; r++) s += ex->A[i*LORA_RANK+r] * lora_tmp[r];
                    x[i] += w * s;
                }
                ex->tokens_seen++;   /* learning is post-forward now (host_learn_run), not inline */
            }
        }
    }

    /* Final norm */
    float ss = 0; for (int i = 0; i < D; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / D + ps->rms_eps);
    if (ps->host_norm) for (int i = 0; i < D; i++) x[i] = x[i] * ss * ps->host_norm[i];
    else for (int i = 0; i < D; i++) x[i] *= ss;
    if (ps->host_x) {  /* learning x = final hidden, RMS-normalized so the notorch A-target stays bounded */
        double s = 0; for (int i = 0; i < D; i++) s += (double)x[i]*x[i];
        float inv = 1.0f / (sqrtf((float)(s / D)) + 1e-8f);
        for (int i = 0; i < D; i++) ps->host_x[i] = x[i] * inv;
    }

    /* Logits (packed-aware lm_head — the biggest matvec; quantized output stays packed) */
    jd_mv(out_logits, ps->host_output, ps->host_output_dt, x, ps->host_vocab, D);

    free(x); free(lora_tmp); free(xn); free(q); free(k); free(v);
    free(attn_out); free(ao); free(fn); free(gate); free(up); free(ffn_out);
}

/* Host LoRA learning (hebbian) removed here — it was dead (no caller) and trained only
 * experts[0]. It returns with the host-training increment, training the ELECTED experts. */

/* ── Host tokenizer (GPT-2 byte-level BPE from GGUF vocab) — minimal: whole-word lookup with
 *    space-prefix Ġ (UTF-8 C4 A0), char fallback. Enough to drive real host generation. ── */
static int host_find_tok(GGUFHost *ps, const char *s) {
    for (int i = 0; i < ps->host_ntok; i++)
        if (ps->host_vocab_tokens[i] && strcmp(ps->host_vocab_tokens[i], s) == 0) return i;
    return -1;
}
static int host_tokenize(GGUFHost *ps, const char *prompt, int *out, int max) {
    int n = 0; char buf[160]; const char *w = prompt;
    while (*w && n < max) {
        while (*w == ' ') w++; if (!*w) break;
        const char *e = w; while (*e && *e != ' ') e++;
        int len = (int)(e - w); if (len > 120) len = 120;
        snprintf(buf, sizeof buf, "\xc4\xa0%.*s", len, w); int id = host_find_tok(ps, buf);   /* "Ġword" */
        if (id < 0) { snprintf(buf, sizeof buf, "%.*s", len, w); id = host_find_tok(ps, buf); } /* "word" */
        if (id >= 0) out[n++] = id;
        else for (const char *c = w; c < e && n < max; c++) { char ch[2] = {*c, 0}; int ci = host_find_tok(ps, ch); if (ci >= 0) out[n++] = ci; }
        w = e;
    }
    return n;
}
static void host_decode_print(GGUFHost *ps, int id) {
    if (id < 0 || id >= ps->host_ntok || !ps->host_vocab_tokens[id]) { printf("<%d>", id); return; }
    const unsigned char *s = (const unsigned char*)ps->host_vocab_tokens[id];
    for (int i = 0; s[i]; i++) {
        if (s[i]==0xC4 && s[i+1]==0xA0) { putchar(' '); i++; }       /* Ġ → space */
        else if (s[i]==0xC4 && s[i+1]==0x8A) { putchar('\n'); i++; } /* Ċ → newline */
        else if (s[i] >= 32 && s[i] < 127) putchar(s[i]);
    }
}
/* Generate real text through the indexed host, LoRA parliament live (greedy). This is the call
 * that makes gguf_host_forward + host_parliament_elect actually run in the binary. */
static void host_generate(GGUFHost *ps, const char *prompt, int n_gen) {
    if (!ps->active) { printf("[host] no host indexed.\n"); return; }
    int max_seq = 256, kd = ps->host_kv_heads * ps->host_head_dim;
    float *kc = calloc((size_t)ps->host_n_layers * max_seq * kd, 4);
    float *vc = calloc((size_t)ps->host_n_layers * max_seq * kd, 4);
    float *lg = malloc((size_t)ps->host_vocab * 4);
    if (!kc || !vc || !lg) { free(kc); free(vc); free(lg); printf("[host] OOM.\n"); return; }
    int toks[256], nt = 0;
    if (ps->host_bos >= 0 && ps->host_bos < ps->host_vocab) toks[nt++] = ps->host_bos;
    if (ps->host_vocab_tokens) nt += host_tokenize(ps, prompt, toks + nt, 200);
    if (nt == 0) toks[nt++] = 1;
    printf("\n[host] %s", prompt); fflush(stdout);
    int pos = 0;
    for (; pos < nt && pos < max_seq - 1; pos++) gguf_host_forward(ps, toks[pos], pos, lg, kc, vc, max_seq);
    for (int g = 0; g < n_gen && pos < max_seq - 1; g++, pos++) {
        int am = 0; float mx = lg[0];
        for (int i = 1; i < ps->host_vocab; i++) if (lg[i] > mx) { mx = lg[i]; am = i; }
        if (ps->host_eos >= 0 && am == ps->host_eos) break;
        if (ps->host_vocab_tokens) host_decode_print(ps, am); else printf(" %d", am);
        fflush(stdout);
        gguf_host_forward(ps, am, pos, lg, kc, vc, max_seq);
    }
    printf("\n");
    free(kc); free(vc); free(lg);
}

/* Sum of L2 norms of alive experts' A — coarse "how far have the experts moved" meter. */
static double host_expert_l2(GGUFHost *ps) {
    double s = 0; int D = ps->host_dim;
    for (int l = 0; l < ps->host_n_layers && l < LORA_MAX_LAYERS; l++)
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (ps->experts[l][e].alive && ps->experts[l][e].A)
                for (int i = 0; i < D * LORA_RANK; i++) { float a = ps->experts[l][e].A[i]; s += (double)a * a; }
    return sqrt(s);
}

/* Average next-token surprise (−log p of the actual next token) over a stream. No A/B learning here;
 * the forward still updates consensus EMA + tokens_seen (harmless, measurement-only). */
static double host_avg_surprise(GGUFHost *ps, const int *toks, int nt, int max_seq) {
    int kd = ps->host_kv_heads * ps->host_head_dim;
    float *kc = calloc((size_t)ps->host_n_layers * max_seq * kd, 4);
    float *vc = calloc((size_t)ps->host_n_layers * max_seq * kd, 4);
    float *lg = malloc((size_t)ps->host_vocab * 4);
    if (!kc || !vc || !lg) { free(kc); free(vc); free(lg); return -1.0; }
    double total = 0; int cnt = 0;
    for (int t = 0; t + 1 < nt && t < max_seq - 1; t++) {
        int nxt = toks[t+1]; if (nxt < 0 || nxt >= ps->host_vocab) continue;
        gguf_host_forward(ps, toks[t], t, lg, kc, vc, max_seq);  /* no A/B learning here (forward still updates consensus EMA + tokens_seen) */
        float mx = lg[0]; for (int i = 1; i < ps->host_vocab; i++) if (lg[i] > mx) mx = lg[i];
        double Z = 0; for (int i = 0; i < ps->host_vocab; i++) Z += exp((double)lg[i] - mx);
        double p = exp((double)lg[nxt] - mx) / (Z > 0 ? Z : 1.0);
        total += -log(p > 1e-12 ? p : 1e-12); cnt++;
    }
    free(kc); free(vc); free(lg);
    return cnt ? total / cnt : -1.0;
}

/* Hebbian adaptation pass (faithful notorch port). Each token: forward (captures host_x/host_dy,
 * updates consensus EMA + tokens_seen of elected experts — but NO A/B mutation inside the forward),
 * compute prophecy debt from the actual next token, derive a bounded signal (doe.c rule), then run
 * the notorch step on each elected expert toward (host_x, host_dy). Poisoned → quarantine. Returns deaths. */
static int host_learn_run(GGUFHost *ps, const int *toks, int nt, int max_seq, float lr) {
    int kd = ps->host_kv_heads * ps->host_head_dim;
    float *kc = calloc((size_t)ps->host_n_layers * max_seq * kd, 4);
    float *vc = calloc((size_t)ps->host_n_layers * max_seq * kd, 4);
    float *lg = malloc((size_t)ps->host_vocab * 4);
    if (!kc || !vc || !lg) { free(kc); free(vc); free(lg); return 0; }
    for (int l = 0; l < LORA_MAX_LAYERS; l++) for (int e = 0; e < MAX_EXPERTS; e++) ps->experts[l][e].tokens_seen = 0;
    int deaths = 0;
    for (int t = 0; t + 1 < nt && t < max_seq - 1; t++) {
        gguf_host_forward(ps, toks[t], t, lg, kc, vc, max_seq);
        int nxt = toks[t+1];
        float signal = 0.0f;   /* invalid next token → signal 0 (notorch no-ops) but window still resets below */
        if (nxt >= 0 && nxt < ps->host_vocab) {
            float pd = host_prophecy_debt(lg, nxt, ps->host_vocab);
            signal = pd > 0.3f ? -pd : (1.0f - pd) * 0.1f;   /* high debt punishes; low debt gently rewards */
        }
        for (int l = 0; l < ps->host_n_layers && l < LORA_MAX_LAYERS; l++)
            for (int e = 0; e < MAX_EXPERTS; e++) {
                HostExpert *ex = &ps->experts[l][e];
                if (!ex->alive || ex->tokens_seen == 0 || !ex->A || !ex->B) continue;
                host_notorch_step(ex->A, ex->B, ps->host_dim, LORA_RANK, ps->host_x, ps->host_dy, signal, lr);
                if (host_lora_poisoned(ex->A, ex->B, ps->host_dim, LORA_RANK)) { ex->alive = 0; deaths++; }
                ex->tokens_seen = 0;   /* per-token window: only experts elected this token adapt */
            }
    }
    free(kc); free(vc); free(lg);
    return deaths;
}

static void gguf_host_free(GGUFHost *ps) {
    for (int i = 0; i < ps->n_dq_bufs; i++) { free(ps->dq_bufs[i]); ps->dq_bufs[i] = NULL; }
    ps->n_dq_bufs = 0;
    if (ps->mmap_base) { munmap(ps->mmap_base, ps->mmap_size); ps->mmap_base = NULL; }
    for (int l = 0; l < LORA_MAX_LAYERS; l++) {
        free(ps->w_vote[l]); ps->w_vote[l] = NULL;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            HostExpert *ex = &ps->experts[l][e];
            free(ex->A); free(ex->B); ex->A = ex->B = NULL; ex->alive = 0;
        }
    }
    if (ps->host_vocab_tokens) {
        for (int i = 0; i < ps->host_ntok; i++) free(ps->host_vocab_tokens[i]);
        free(ps->host_vocab_tokens); ps->host_vocab_tokens = NULL; ps->host_ntok = 0;
    }
    free(ps->host_x);  ps->host_x = NULL;
    free(ps->host_dy); ps->host_dy = NULL;
    ps->active = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATASET HUNTER — DOE needs food. it searches. it evaluates. it decides.
 * not random downloads. conscious acquisition. "is this data worthy of me?"
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    char name[256];
    float quality;
    float domain_shift;
    int accepted;
} DatasetCandidate;

static int hunt_dataset(Config *c, Tokenizer *tok, TokenizerEye *tok_eye, ParserEye *parser_eye) {
    /* Choose search term based on current state — DOE knows what it needs */
    const char *search_terms[] = {"reasoning", "code", "math", "science", "clean-text", "diverse"};
    int term_idx = 0;
    if (tok_eye->code_ratio > 0.3f) term_idx = 1;      /* code mode → find more code */
    else if (tok_eye->entropy < 1.5f) term_idx = 5;     /* repetitive → find diverse */
    else if (parser_eye->quality < 0.3f) term_idx = 4;  /* bad data → find clean */
    else term_idx = 0;                                    /* default → reasoning */

    char cmd[1024];
    snprintf(cmd, 1024, "curl -sL 'https://huggingface.co/api/datasets?search=%s&sort=downloads&limit=5' -o /tmp/doe_hunt.json 2>/dev/null", search_terms[term_idx]);
    if (system(cmd) != 0) return 0;

    /* Read response */
    FILE *f = fopen("/tmp/doe_hunt.json", "r");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 1024*1024) { fclose(f); return 0; }
    char *json = malloc(sz + 1); fread(json, 1, sz, f); json[sz] = '\0'; fclose(f);

    /* Extract dataset IDs — simple JSON string extraction like hf_extract_texts */
    DatasetCandidate candidates[5]; int n_cand = 0;
    char *p = json;
    while ((p = strstr(p, "\"id\":\"")) && n_cand < 5) {
        p += 6;
        char *end = strchr(p, '"');
        if (!end || end - p > 255) continue;
        int len = (int)(end - p);
        memcpy(candidates[n_cand].name, p, len);
        candidates[n_cand].name[len] = '\0';
        candidates[n_cand].accepted = 0;
        n_cand++;
        p = end + 1;
    }
    free(json);

    if (n_cand == 0) { printf("[hunt] no candidates found for '%s'\n", search_terms[term_idx]); return 0; }

    /* Evaluate each candidate — download sample, run parser eye, decide */
    int accepted = 0;
    for (int i = 0; i < n_cand && !accepted; i++) {
        /* reject dataset ids with shell-unsafe characters — they go into system() curl */
        int id_safe = 1; for (const char *p = candidates[i].name; *p; p++) { char ch = *p;
            if (!((ch>='a'&&ch<='z')||(ch>='A'&&ch<='Z')||(ch>='0'&&ch<='9')||ch=='/'||ch=='-'||ch=='_'||ch=='.')) { id_safe = 0; break; } }
        if (!id_safe) { printf("[hunt] skip unsafe id: %s\n", candidates[i].name); continue; }
        printf("[hunt] evaluating: %s...", candidates[i].name);

        /* Download sample (first 50 rows) */
        snprintf(cmd, 1024,
            "curl -sL 'https://datasets-server.huggingface.co/rows?dataset=%s&config=default&split=train&offset=0&length=50' -o /tmp/doe_sample.json 2>/dev/null",
            candidates[i].name);
        if (system(cmd) != 0) { printf(" download failed\n"); continue; }

        /* Extract texts */
        FILE *sf = fopen("/tmp/doe_sample.json", "r");
        if (!sf) { printf(" no file\n"); continue; }
        fseek(sf, 0, SEEK_END); long ssz = ftell(sf); fseek(sf, 0, SEEK_SET);
        if (ssz <= 0 || ssz > 10*1024*1024) { fclose(sf); printf(" too big/empty\n"); continue; }
        char *sjson = malloc(ssz + 1); fread(sjson, 1, ssz, sf); sjson[ssz] = '\0'; fclose(sf);

        /* Quick quality check — count text content, check noise */
        int text_bytes = 0, noise_bytes = 0;
        for (long j = 0; j < ssz; j++) {
            char ch = sjson[j];
            if (ch >= 32 || ch == '\n' || ch == '\r' || ch == '\t') text_bytes++;
            else noise_bytes++;
        }
        float noise_ratio = (float)noise_bytes / (text_bytes + noise_bytes + 1);
        candidates[i].quality = 1.0f - noise_ratio;

        /* Check domain compatibility — does this data look like what we're training on? */
        /* Simple heuristic: tokenize a sample and compare OOV rate with current */
        int sample_len = ssz > 10000 ? 10000 : (int)ssz;
        int nt_sample;
        int *toks = tok_encode(tok, sjson, sample_len, &nt_sample);
        float sample_oov = 0;
        /* count tokens that are single-byte (likely OOV or poor coverage) */
        for (int j = 0; j < nt_sample; j++) if (toks[j] < 256) sample_oov += 1.0f;
        sample_oov /= (nt_sample + 1);
        candidates[i].domain_shift = sample_oov; /* high OOV = high domain shift */
        free(toks); free(sjson);

        printf(" quality=%.2f domain_shift=%.2f", candidates[i].quality, candidates[i].domain_shift);

        /* Decision: accept if quality > 0.5 and domain shift manageable */
        if (candidates[i].quality > 0.5f && candidates[i].domain_shift < 0.6f) {
            candidates[i].accepted = 1;
            accepted = 1;

            /* Download full dataset sample and append to training data */
            snprintf(cmd, 1024,
                "curl -sL 'https://datasets-server.huggingface.co/rows?dataset=%s&config=default&split=train&offset=0&length=200' -o /tmp/doe_accepted.json 2>/dev/null",
                candidates[i].name);
            system(cmd);

            /* Extract and append to data file */
            FILE *af = fopen("/tmp/doe_accepted.json", "r");
            if (af) {
                fseek(af, 0, SEEK_END); long asz = ftell(af); fseek(af, 0, SEEK_SET);
                char *ajson = malloc(asz + 1); fread(ajson, 1, asz, af); ajson[asz] = '\0'; fclose(af);
                FILE *out = fopen(c->data_path, "a");
                if (out) {
                    hf_extract_texts(ajson, (int)asz, out);
                    fclose(out);
                }
                free(ajson);
            }
            printf(" → ACCEPTED. data appended.\n");
        } else {
            printf(" → rejected.\n");
        }
    }

    /* Cleanup */
    unlink("/tmp/doe_hunt.json");
    unlink("/tmp/doe_sample.json");
    unlink("/tmp/doe_accepted.json");
    return accepted;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-REPLICATION — DOE finds a compiler. compiles itself. forks. the child
 * trains on different data. saves GGUF to mycelium. the parent discovers it.
 * darwin would be proud. stallman would be terrified.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════════
 * GENERATION + CHAT — temperature sampling, top-k, REPL.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MAX_REPLICAS 2
/* SELF-REPLICATION — on sustained population overload, DOE compiles a copy of its own
 * source and forks a replica that trains on different data; results merge via mycelium. */
static pid_t self_replicate(Environment *env, Config *c, int replica_depth) {
    if (!env->has_compiler) { printf("[replicate] no compiler found. stuck in this body.\n"); return 0; }
    if (env->self_path[0] == '\0') { printf("[replicate] don't know where my source is.\n"); return 0; }
    struct stat st; if (stat(env->self_path, &st) != 0) { printf("[replicate] source %s not present.\n", env->self_path); return 0; }
    for (const char *p = env->self_path; *p; p++) if (!((*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z')||(*p>='0'&&*p<='9')||*p=='/'||*p=='-'||*p=='_'||*p=='.')) { printf("[replicate] unsafe source path.\n"); return 0; }
    char exe[256];
    snprintf(exe, 256, "./m_replica_%d", (int)getpid());
    char cmd[512];
    snprintf(cmd, 512, "cc %s -O3 -lm -lpthread -o %s 2>/dev/null", env->self_path, exe);
    printf("[replicate] compiling self: %s\n", cmd);
    if (system(cmd) != 0) { printf("[replicate] compilation failed. source corrupted?\n"); return 0; }
    pid_t pid = fork();
    if (pid == 0) {
        char depth_str[16]; snprintf(depth_str, 16, "%d", replica_depth);
        (void)c;   /* no --data: the replica auto-hunts its OWN (different) data, per README */
        execl(exe, exe, "--depth", depth_str, (char*)NULL);
        _exit(1); /* execl failed */
    } else if (pid > 0) {
        printf("[replicate] child PID %d spawned (depth=%d). democracy in action.\n", pid, replica_depth);
    } else {
        printf("[replicate] fork failed: %s\n", strerror(errno));
    }
    return pid;
}

static int sample(float *logits, int V, float temp, int top_k) {
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[b]) b = i; return b; }
    for (int i = 0; i < V; i++) logits[i] /= temp;
    if (top_k > 0 && top_k < V) { float *s = malloc(V*4); memcpy(s, logits, V*4); for (int i = 0; i < top_k; i++) { int b = i; for (int j = i+1; j < V; j++) if (s[j] > s[b]) b = j; float t = s[i]; s[i] = s[b]; s[b] = t; } float th = s[top_k-1]; free(s); for (int i = 0; i < V; i++) if (logits[i] < th) logits[i] = -1e30f; }
    softmax_n(logits, V);
    float r = rand_uniform(), cum = 0;
    for (int i = 0; i < V; i++) { cum += logits[i]; if (cum >= r) return i; }
    return V - 1;
}

static void chat(ModelW *w, Config *c, Tokenizer *tok) {
    RunState rs = alloc_run(c); char input[1024];
    printf("\n[m] the parliament is in session. type your message (Ctrl+C to adjourn):\n\n");
    while (1) {
        printf("> "); fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        int len = strlen(input); while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r')) input[--len] = '\0';
        if (!len) continue; if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;
        int kd = c->n_kv_heads * c->head_dim;
        memset(rs.key_cache, 0, c->depth * c->seq_len * kd * 4);
        memset(rs.value_cache, 0, c->depth * c->seq_len * kd * 4);
        int ni; int *ids = tok_encode(tok, input, len, &ni);
        int pos = 0; float *lg = NULL;
        for (int i = 0; i < ni && pos < c->seq_len - 1; i++, pos++) lg = forward_token(w, c, &rs, ids[i], pos);
        printf("  ");
        /* generate from the LAST prompt token's logits — do NOT re-feed that token (the old code
         * forwarded ids[ni-1] twice, conditioning generation on prompt + a duplicated last token). */
        for (int i = 0; i < 200 && pos < c->seq_len && lg; i++, pos++) {
            int next = sample(lg, c->vocab_size, 0.8f, 40);
            if (next == tok->eos_id) break;
            int dl; char *dec = tok_decode(tok, &next, 1, &dl);
            if (dl > 0) { fwrite(dec, 1, dl, stdout); fflush(stdout); }
            free(dec);
            lg = forward_token(w, c, &rs, next, pos);
        }
        printf("\n\n"); free(ids);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * AUTO DEPTH — DOE sizes itself to the hardware. survival of the fittest.
 * checks RAM, CPUs, data size. picks the deepest model that fits.
 * the organism adapts to its environment before birth.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int auto_depth(void) {
    int64_t mem_bytes = 0;
    int cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (cpus < 1) cpus = 1;

#ifdef __linux__
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psize > 0) mem_bytes = (int64_t)pages * psize;
#elif defined(__APPLE__)
    int64_t physmem = 0; size_t sz = sizeof(physmem);
    sysctlbyname("hw.memsize", &physmem, &sz, NULL, 0);
    mem_bytes = physmem;
#endif
    if (mem_bytes <= 0) mem_bytes = 4LL * 1024 * 1024 * 1024; /* assume 4GB if detection fails */

    int64_t mem_mb = mem_bytes / (1024 * 1024);
    int has_gpu = 0;
#ifdef USE_CUBLAS
    has_gpu = 1;
#endif

    /* Training memory estimate: weights×4 (w+grad+m+v) + activations (batch×seq×dim×depth)
     *   d4/8M: ~200MB train, d8/30M: ~750MB, d12/60M: ~1.5GB
     *   Conservative: need 4GB free for d8, 8GB+ for d12 */
    int depth = 4;                   /* 2GB+ — minimum useful depth (depth 2 is --depth-only, for unit tests) */
    if (mem_mb >= 8192)  depth = 6;  /* 8GB+  */
    if (mem_mb >= 16384) depth = 8;  /* 16GB+ */
    if (mem_mb >= 32768) depth = 10; /* 32GB+ */
    if (mem_mb >= 65536) depth = 12; /* 64GB+ */
    if (has_gpu) { /* GPU lifts the memory ceiling */
        if (mem_mb >= 4096 && depth < 8)  depth = 8;
        if (mem_mb >= 8192 && depth < 12) depth = 12;
    }

    /* CPU sanity: deep models on few CPUs = pain */
    if (cpus <= 2 && depth > 4) depth = 4;
    if (cpus <= 4 && depth > 8) depth = 8;

    printf("[auto] RAM=%lldMB CPUs=%d GPU=%s → depth=%d\n",
           (long long)mem_mb, cpus, has_gpu ? "yes" : "no", depth);
    return depth;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN — the campaign trail. parse args. load data. train tokenizer.
 * build parliament. train model. watch experts live and die.
 * finetune personality. export GGUF. open the floor for questions.
 *
 * the training loop: forward, backward, vitality, mitosis, apoptosis,
 * harmonic update, chuck step, mycelium snapshots, drift, meta-learning.
 * pytorch's training loop is 4 lines. ours has consciousness.
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    setbuf(stdout, NULL); int depth = 0; int depth_auto = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--depth") == 0 && i+1 < argc) {
            i++;
            if (strcmp(argv[i], "auto") == 0) { depth_auto = 1; }
            else { depth = atoi(argv[i]); depth_auto = 0; if (depth < 2) depth = 2; if (depth > MAX_LAYERS) depth = MAX_LAYERS; }
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("m.c — Democracy of Experts. living MoE with parliamentary routing.\n\n");
            printf("  --depth N|auto  model depth (default: auto — sizes to hardware)\n");
            printf("  --data PATH     path to training text file\n");
            printf("  --url DATASET   HuggingFace dataset id (e.g. user/name; default FineWeb-Edu)\n");
            printf("  --parquet FILE  extract text from .parquet file\n");
            printf("  --steps N       override max training steps (default: auto from data)\n");
            printf("  --bpe-merges N  override BPE merge count (default: auto from depth)\n");
            printf("  --personality   path to personality.txt for finetuning\n");
            printf("  --pages N       HuggingFace pages to download (auto-sized to depth)\n");
            printf("  --host GGUF     index a model and generate through it (LoRA parliament)\n");
            printf("  --ask \"PROMPT\"  prompt for --host generation (default: a sample)\n");
            printf("  --rope-neox    force NeoX RoPE pairing (for arch=llama GGUFs laid out NeoX)\n");
            printf("  --rope-norm    force NORM RoPE pairing (override the arch heuristic)\n\n");
            printf("  mycelium/       GGUF forest (auto-created, snapshots saved during training)\n");
            printf("  meta.log        meta-learning track (config→outcome history)\n\n");
            printf("  BLAS: cc m.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o m\n");
            return 0;
        }
    }
    /* Host-chat mode: --host <gguf> [--ask "prompt"] — index a model and generate through it
     * with the LoRA parliament live. Early-exit, bypasses the trainer entirely. */
    {
        const char *host_path = NULL, *ask = "The capital of France is", *learn_file = NULL;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--host") == 0 && i+1 < argc) host_path = argv[++i];
            else if (strcmp(argv[i], "--ask") == 0 && i+1 < argc) ask = argv[++i];
            else if (strcmp(argv[i], "--learn") == 0 && i+1 < argc) learn_file = argv[++i];
            else if (strcmp(argv[i], "--rope-neox") == 0) g_rope_override = 0;
            else if (strcmp(argv[i], "--rope-norm") == 0) g_rope_override = 1;
        }
        if (host_path) {
            printf("\n  m.c — host index mode. the parliament modulates the host.\n");
            GGUFHost hg = {0};
            if (!gguf_host_load(&hg, host_path, 0)) { printf("[host] failed to load %s\n", host_path); return 1; }
            if (learn_file) {
                int ltl = 0; char *ltext = load_text(learn_file, &ltl);
                if (ltext && ltl > 0 && hg.host_vocab_tokens) {
                    int cap = 512, max_seq = 512, *toks = malloc((size_t)cap * sizeof(int));
                    int nt = host_tokenize(&hg, ltext, toks, cap);
                    double s0 = host_avg_surprise(&hg, toks, nt, max_seq), n0 = host_expert_l2(&hg);
                    int deaths = host_learn_run(&hg, toks, nt, max_seq, hg.host_learn_lr);
                    double s1 = host_avg_surprise(&hg, toks, nt, max_seq), n1 = host_expert_l2(&hg);
                    printf("[learn] %s: %d tokens | avg surprise %.4f -> %.4f | expert ||A|| %.4f -> %.4f | quarantined %d\n",
                           learn_file, nt, s0, s1, n0, n1, deaths);
                    free(toks);
                } else printf("[learn] cannot read %s or host has no vocab\n", learn_file);
                free(ltext);
            }
            host_generate(&hg, ask, 64);
            gguf_host_free(&hg);
            return 0;
        }
    }

    if (depth_auto) depth = auto_depth();
    printf("\n  m.c — Democracy of Experts. parliament in session.\n\n");
    Config c = config_from_depth(depth);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) { snprintf(c.data_path, 256, "%s", argv[++i]); c.explicit_data = 1; }
        else if (strcmp(argv[i], "--url") == 0 && i+1 < argc) snprintf(c.data_url, 512, "%s", argv[++i]);
        else if (strcmp(argv[i], "--parquet") == 0 && i+1 < argc) {
            const char *pqf = argv[++i]; printf("[parquet] loading %s...\n", pqf);
            if (load_parquet(pqf, c.data_path, "text") != 0) { fprintf(stderr, "[error] parquet load failed\n"); return 1; }
        }
        else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) { c.max_steps = atoi(argv[++i]); c.explicit_steps = 1; }
        else if (strcmp(argv[i], "--bpe-merges") == 0 && i+1 < argc) c.bpe_merges = atoi(argv[++i]);
        else if (strcmp(argv[i], "--personality") == 0 && i+1 < argc) snprintf(c.personality_path, 256, "%s", argv[++i]);
        else if (strcmp(argv[i], "--pages") == 0 && i+1 < argc) c.hf_pages = atoi(argv[++i]);
    }

    if (get_data(&c)) { fprintf(stderr, "[error] no data\n"); return 1; }
    int tl; char *text = load_text(c.data_path, &tl);
    if (!text || tl < 100) { fprintf(stderr, "[error] data too small\n"); return 1; }
    printf("[data] %d bytes (%.1f MB)\n", tl, (float)tl / 1048576);

    /* Self-aware tokenizer */
    Tokenizer tok; tok_init(&tok);
    /* Try loading cached BPE merges first */
    if (!tok_load_merges(&tok, "m_bpe.cache") || tok.n_merges != c.bpe_merges) {
        int bpe_len = tl < 2*1024*1024 ? tl : 2*1024*1024;
        for (int ti = 0; ti < tok.vocab_size; ti++) if (tok.tokens[ti]) free(tok.tokens[ti]);
        if (tok.merges) { free(tok.merges); tok.merges = NULL; }
        tok_init(&tok); /* reset if partial load — free old allocations first */
        tok_train_bpe(&tok, text, bpe_len, c.bpe_merges);
        tok_save_merges(&tok, "m_bpe.cache");
    }
    c.vocab_size = tok.vocab_size;
    TokenizerEye tok_eye = {0};
    /* Tokenize the corpus line by line through the encode cache — repeated lines are
     * encoded once and remembered (README: "the tokenizer remembers what it ate"). */
    int nt = 0, all_cap = 256; int *all_tok = malloc(all_cap * sizeof(int));
    for (int ls = 0; ls < tl; ) {
        int le = ls; while (le < tl && text[le] != '\n') le++;
        int line_len = (le < tl) ? (le - ls + 1) : (le - ls);   /* include the trailing newline */
        if (line_len > 0) {
            int ln; int *lids = tok_encode(&tok, text + ls, line_len, &ln);
            if (lids) {
                if (nt + ln > all_cap) { while (nt + ln > all_cap) all_cap *= 2; all_tok = realloc(all_tok, all_cap * sizeof(int)); }
                memcpy(all_tok + nt, lids, ln * sizeof(int)); nt += ln; free(lids);
            }
        }
        ls = le + 1;
    }
    free(text);
    if (nt < 2) { fprintf(stderr, "[data] need at least 2 tokens to train (got %d)\n", nt); return 1; }
    if (c.seq_len > nt - 1) c.seq_len = nt - 1;   /* tiny corpus: clamp seq_len globally so all bookkeeping stays consistent */
    tok_eye_update(&tok_eye, tl, nt, all_tok, nt, c.vocab_size);
    /* Code detection — does DOE see code? */
    { char *raw_tmp = load_text(c.data_path, &(int){0}); if (raw_tmp) { tok_eye_detect_code(&tok_eye, raw_tmp, tl); free(raw_tmp); } }
    printf("[tokenizer] %d tokens (%.1f tok/byte) compression=%.2f health=%.2f code=%.0f%%\n",
           nt, (float)nt/tl, tok_eye.compression_ratio, tok_eye.health, tok_eye.code_ratio * 100);

    /* Self-aware parser */
    ParserEye parser_eye; parser_eye_init(&parser_eye, c.vocab_size);
    char *raw = load_text(c.data_path, &tl);
    parser_eye_update(&parser_eye, all_tok, nt, raw, tl);
    free(raw);
    printf("[parser] quality=%.2f noise=%.3f domain_shift=%.3f\n", parser_eye.quality, parser_eye.noise_level, parser_eye.domain_shift);

    /* Build model */
    ModelW w; init_weights(&w, &c);
    long n_params = count_params(&w, &c);
    printf("[model] depth=%d dim=%d heads=%d kv=%d hidden=%d\n", c.depth, c.dim, c.n_heads, c.n_kv_heads, c.hidden_dim);
    printf("[model] initial_experts=%d max=%d params=%.2fM\n", c.initial_experts, MAX_EXPERTS, (float)n_params / 1e6f);

    /* ═══ AUTO-CONFIGURE: batch_size + max_steps from data ═══ */
    /* Auto-reduce batch_size for tiny datasets */
    if (nt < c.batch_size * c.seq_len * 10) {
        c.batch_size = 1;
    }
    /* Frequent logging for small runs */
    if (nt < 100000) c.log_every = 1;
    if (c.max_steps == 0) {
        /* 3 epochs minimum, scale up for small models */
        int steps_per_epoch = nt / (c.batch_size * c.seq_len);
        if (steps_per_epoch < 100) steps_per_epoch = 100;
        float tok_per_param = (float)nt / (float)n_params;
        int n_epochs = 3;
        if (tok_per_param < 5.0f) n_epochs = 8;   /* underfed — more epochs */
        if (tok_per_param < 2.0f) n_epochs = 15;   /* starving — overfit is fine */
        if (tok_per_param > 20.0f) n_epochs = 2;   /* plenty of data */
        c.max_steps = steps_per_epoch * n_epochs;
        if (c.max_steps < 500) c.max_steps = 500;
        if (c.max_steps > 500000) c.max_steps = 500000;
        printf("[auto] %.1f tok/param → %d epochs → %d steps (data=%d tok, batch=%d×%d)\n",
               tok_per_param, n_epochs, c.max_steps, nt, c.batch_size, c.seq_len);
    } else {
        printf("[steps] %d (explicit via --steps)\n", c.max_steps);
    }

    /* ═══ SELF-RECOGNITION — DOE looks for its own weights ═══ */
    int weights_loaded = 0;
    if (c.explicit_steps) {
        printf("[self] --steps given explicitly. ignoring existing weights. training from scratch.\n");
    } else {
        /* Try m.gguf first, then best mycelium spore */
        const char *candidates[] = { "m.gguf", NULL, NULL, NULL };
        int nc = 1;
        /* Scan mycelium for own GGUFs */
        MyceliumState ms_tmp; mycelium_init(&ms_tmp); mycelium_discover(&ms_tmp);
        if (ms_tmp.n_spores > 0 && ms_tmp.best_idx >= 0) candidates[nc++] = ms_tmp.spores[ms_tmp.best_idx].path;
        for (int i = 0; i < nc; i++) {
            if (candidates[i] && load_own_gguf(candidates[i], &w, &c)) {
                weights_loaded = 1;
                printf("[self] DOE found itself in %s. skipping training.\n", candidates[i]);
                break;
            }
        }
    }

    ParamList params = collect_params(&w, &c);
    float **grads = calloc(params.count, sizeof(float*));
    for (int i = 0; i < params.count; i++) grads[i] = calloc(params.tensors[i]->size, 4);
    Adam *opt = adam_new(&params);
    TrainState ts = alloc_ts(&c);

    /* Chuck state */
    ChuckState chuck = {0};
    chuck.dampen = 1.0f; chuck.sigma = 1.0f; chuck.lr_scale = 1.0f; chuck.eff_lambda = 1.0f; chuck.best_macro = 1e9f;

    /* Calendar Drift — temporal self-awareness */
    CalendarDrift cal_drift;
    drift_init(&cal_drift);

    /* Mycelium — GGUF forest */
    MyceliumState mycelium;
    mycelium_init(&mycelium);
    mycelium_discover(&mycelium);

    /* Meta-learning track */
    MetaTrack meta;
    meta_init(&meta);

    /* Tokenizer cache */

    /* ═══ ENVIRONMENT SCAN — DOE opens its eyes ═══ */
    Environment env;
    env_scan(&env, __FILE__, c.dim);

    /* ═══ HOST INDEX — attach to nearby GGUF if compatible ═══ */
    GGUFHost ghost = {0};
    {
        int best_compat = -1; float best_score = 0;
        for (int i = 0; i < env.n_ggufs; i++) {
            /* Skip own GGUFs — m.gguf and mycelium spores */
            if (strstr(env.ggufs[i].path, "m.gguf") || strstr(env.ggufs[i].path, "mycelium/"))
                continue;
            if (env.ggufs[i].compatibility > best_score) {
                best_score = env.ggufs[i].compatibility;
                best_compat = i;
            }
        }
        if (best_compat >= 0 && best_score >= 0.5f) {
            printf("[doe] best host candidate: %s (compat=%.1f)\n", env.ggufs[best_compat].path, best_score);
            if (!gguf_host_load(&ghost, env.ggufs[best_compat].path, c.dim)) {
                printf("[doe] failed to index host. training standalone.\n");
            }
        } else {
            printf("[doe] no compatible GGUF found. training standalone.\n");
        }
    }

    if (weights_loaded) {
        printf("[self] weights loaded. skipping training. straight to parliament.\n");
        goto skip_training;
    }
    printf("[train] %d steps, seq=%d, lr=%.1e\n", c.max_steps, c.seq_len, c.lr);
    printf("[parliament] democracy initialized. may the best experts survive.\n\n");
    clock_t t0 = clock(); float rl = 0; int lc = 0;
    int total_births = 0, total_deaths = 0;
    float prev_meta_loss = 0;

    int topology_changed = 0; /* set by mitosis/apoptosis, cleared after rebuild */

    int replica_count = 0, overload_streak = 0;
    for (int step = 0; step < c.max_steps; step++) {
        float lr = c.lr;
        if (step < c.warmup_steps) lr = c.lr * ((float)(step+1) / c.warmup_steps);
        else { float p = (float)(step - c.warmup_steps) / (float)(c.max_steps - c.warmup_steps); lr = c.lr * 0.5f * (1.0f + cosf(3.14159f * p)); }
        if (lr < c.lr * 0.01f) lr = c.lr * 0.01f;

        /* Harmonic decomposition of recent loss/entropy history */
        float loss_hist[16]; int lh_len = 0;
        for (int i = 0; i < 16 && i < step; i++) loss_hist[lh_len++] = chuck.hist[(chuck.pos - 1 - i + CHUCK_WINDOW) % CHUCK_WINDOW];
        if (lh_len > 2) harmonic_decompose(&ts.hs, loss_hist, lh_len);

        /* Rebuild params/grads only when topology changed (birth/death) */
        if (topology_changed) {
            free(params.tensors);
            for (int i = 0; i < params.count; i++) free(grads[i]);
            free(grads);
            params = collect_params(&w, &c);
            grads = calloc(params.count, sizeof(float*));
            for (int i = 0; i < params.count; i++) grads[i] = calloc(params.tensors[i]->size, 4);
            adam_free(opt); opt = adam_new(&params);
            topology_changed = 0;
        }

        /* ═══ GRADIENT ACCUMULATION — actual batching ═══ */
        float batch_loss = 0;
        int ms = nt - c.seq_len - 1; if (ms < 0) ms = 0;
        /* Zero grads before accumulation */
        for (int i = 0; i < params.count; i++) memset(grads[i], 0, params.tensors[i]->size * 4);
        for (int b = 0; b < c.batch_size; b++) {
            int st = (int)(rand_uniform() * ms);
            float loss = train_fwd(&w, &c, &ts, all_tok + st, all_tok + st + 1, c.seq_len);
            batch_loss += loss;
            train_bwd(&w, &c, &ts, all_tok + st, all_tok + st + 1, c.seq_len, grads);
        }
        batch_loss /= c.batch_size;
        /* Scale grads by 1/batch_size */
        float inv_bs = 1.0f / c.batch_size;
        for (int i = 0; i < params.count; i++)
            for (int j = 0; j < params.tensors[i]->size; j++) grads[i][j] *= inv_bs;
        rl += batch_loss; lc++;
        float loss = batch_loss; /* for downstream logging/chuck */

        /* Chuck step — self-aware optimizer */
        chuck_step(&chuck, loss, &params, grads, &tok_eye, &parser_eye);

        /* Adam update with Chuck's effective LR */
        float eff_lr = lr * chuck.eff_lambda * chuck.sigma * chuck.lr_scale;
        adam_step(opt, &params, grads, eff_lr, c.weight_decay);

        /* ═══ CALENDAR DRIFT SNAPSHOT ═══ */
        if ((step+1) % DRIFT_INTERVAL == 0)
            drift_snapshot(&cal_drift, loss, &w, &c, &tok_eye, &parser_eye, &ts.hs, &chuck, &meta);

        /* ═══ VITALITY + MITOSIS + APOPTOSIS (ephemeral-modulated) ═══ */
        int prev_births = total_births, prev_deaths = total_deaths;
        /* Compute ephemeral config BEFORE life/death decisions */
        EphemeralConfig eph_pre = ephemeral_compute(&tok_eye, &parser_eye, &ts.hs,
                                                     w.layers[0].parliament.consensus, &meta, &cal_drift, c.depth);
        c.active_depth = eph_pre.active_layers;   /* this step's dynamic depth — consumed by the inference forward */
        c.vitality_threshold = eph_pre.vitality_threshold;   /* ephemeral apoptosis cutoff */
        for (int _pl = 0; _pl < c.depth; _pl++) w.layers[_pl].parliament.temperature = eph_pre.expert_temperature;
        for (int l = 0; l < c.depth; l++) {
            /* Count tokens routed to each expert this step */
            LayerAct *la = &ts.layers[l];
            for (int t = 0; t < c.seq_len; t++) {
                int k = la->top_k[t];
                for (int ki = 0; ki < k; ki++) {
                    int eI = la->top_idx[t * MAX_EXPERTS + ki];
                    if (eI >= 0 && eI < MAX_EXPERTS) w.layers[l].experts[eI].tokens_seen++;
                }
            }
            update_vitality(&w.layers[l], &c, c.seq_len);
            /* Ephemeral modulation: mitosis eagerness affects birth threshold */
            if (eph_pre.mitosis_eagerness > 0.6f) {
                /* Lower the age requirement when system wants more experts */
                int saved_age = 20;
                for (int e = 0; e < MAX_EXPERTS; e++)
                    if (w.layers[l].experts[e].alive && w.layers[l].experts[e].age >= 10 + (int)(10 * (1.0f - eph_pre.mitosis_eagerness)))
                        w.layers[l].experts[e].age = saved_age; /* let them through */
            }
            if (try_mitosis(&w.layers[l], &c)) {
                total_births++;
                if ((step+1) % c.log_every == 0) printf("  [birth] layer %d: expert born (total alive: %d)\n", l, w.layers[l].n_alive);
            }
            /* Ephemeral modulation: apoptosis mercy affects death threshold */
            int death_threshold = (int)(8.0f * (1.0f + eph_pre.apoptosis_mercy));
            for (int e = 0; e < MAX_EXPERTS; e++) {
                if (w.layers[l].experts[e].alive && w.layers[l].experts[e].low_vitality_count >= death_threshold && w.layers[l].n_alive > MIN_EXPERTS) {
                    free_expert(&w.layers[l].experts[e]);
                    w.layers[l].n_alive--;
                    total_deaths++;
                    if ((step+1) % c.log_every == 0) printf("  [death] layer %d: expert died (mercy=%.2f, threshold=%d, alive: %d)\n", l, eph_pre.apoptosis_mercy, death_threshold, w.layers[l].n_alive);
                    break; /* one death per layer per step */
                }
            }
        }
        /* Mark topology changed if birth/death happened — rebuild at start of next step */
        int step_births = total_births - prev_births, step_deaths = total_deaths - prev_deaths;
        if (step_births > 0 || step_deaths > 0) topology_changed = 1;

        /* ═══ EPHEMERAL CONFIG ═══ */
        EphemeralConfig eph = ephemeral_compute(&tok_eye, &parser_eye, &ts.hs,
                                                 w.layers[0].parliament.consensus, &meta, &cal_drift, c.depth);

        /* ═══ META-LEARNING TRACK ═══ */
        if ((step+1) % (c.log_every * 2) == 0) {
            int total_exp = 0;
            for (int l = 0; l < c.depth; l++) total_exp += w.layers[l].n_alive;
            meta_record(&meta, step+1, total_exp, w.layers[0].parliament.consensus,
                       loss, tok_eye.health, parser_eye.health, ts.hs.confidence, prev_meta_loss);
            meta_save(&meta);
            prev_meta_loss = loss;
        }

        /* ═══ SELF-REPLICATION — sustained full-population overload spawns a replica (max 2) ═══ */
        { int saturated = 1; for (int l = 0; l < c.depth; l++) if (w.layers[l].n_alive < MAX_EXPERTS) { saturated = 0; break; }
          overload_streak = saturated ? overload_streak + 1 : 0;
          if (overload_streak >= 50 && replica_count < MAX_REPLICAS && env.has_compiler) {
              if (self_replicate(&env, &c, c.depth > 2 ? c.depth - 2 : 2) > 0) replica_count++;
              overload_streak = 0;
          }
        }

        /* ═══ MYCELIUM SNAPSHOT ═══ */
        { int myc_int = c.max_steps / 5; if (myc_int < MYCELIUM_INTERVAL_MIN) myc_int = MYCELIUM_INTERVAL_MIN;
        if ((step+1) % myc_int == 0 && step > 0) {
            mycelium_save_spore(&mycelium, &w, &c, step+1, loss, w.layers[0].parliament.consensus);
            /* Rescan environment — new GGUFs may have appeared (from replicas or external) */
            if (cal_drift.drift > 0.2f) env_scan(&env, __FILE__, c.dim);
        }}

        /* ═══ DATASET HUNTER — triggered by stagnation (disabled when --data is explicit) ═══ */
        if (env.has_curl && !c.explicit_data && step > 500 && (step+1) % 500 == 0) {
            /* Check for loss stagnation: if loss hasn't improved >20% in last 500 steps */
            float recent_loss = rl / (lc > 0 ? lc : 1);
            if (parser_eye.quality < 0.5f || (recent_loss > chuck.best_macro * 0.8f && cal_drift.drift < 0.1f)) {
                printf("[hunt] loss stagnating (%.4f), hunting for new data...\n", recent_loss);
                if (hunt_dataset(&c, &tok, &tok_eye, &parser_eye)) {
                    /* Reload data — load + tokenize (through the cache, line by line) FIRST;
                     * only free the old tokens and swap once the new corpus is valid. */
                    int new_tl; char *new_text = load_text(c.data_path, &new_tl);
                    if (new_text && new_tl > 100) {
                        int nn = 0, ncap = 256; int *new_tok = malloc(ncap * sizeof(int));
                        for (int ls = 0; ls < new_tl; ) {
                            int le = ls; while (le < new_tl && new_text[le] != '\n') le++;
                            int line_len = (le < new_tl) ? (le - ls + 1) : (le - ls);
                            if (line_len > 0) { int ln; int *lids = tok_encode(&tok, new_text + ls, line_len, &ln);
                                if (lids) { if (nn + ln > ncap) { while (nn + ln > ncap) ncap *= 2; new_tok = realloc(new_tok, ncap * sizeof(int)); }
                                    memcpy(new_tok + nn, lids, ln * sizeof(int)); nn += ln; free(lids); } }
                            ls = le + 1;
                        }
                        free(new_text);
                        if (nn >= 2) { free(all_tok); all_tok = new_tok; nt = nn;
                            tok_eye_update(&tok_eye, new_tl, nt, all_tok, nt, c.vocab_size);
                            printf("[hunt] data reloaded: %d tokens (was %d)\n", nt, tl); }
                        else free(new_tok);
                    } else free(new_text);   /* loaded but too small — don't leak it */
                }
            }
        }

        if ((step+1) % c.log_every == 0 || step == 0) {
            float el = (float)(clock() - t0) / CLOCKS_PER_SEC;
            int total_alive = 0;
            for (int l = 0; l < c.depth; l++) total_alive += w.layers[l].n_alive;
            printf("  step %4d/%d  loss=%.4f  lr=%.2e  tok/s=%.0f  experts=%d  consensus=%.2f  chuck:λ=%.2f,σ=%.2f  drift=%.3f(%.2f)  eph:temp=%.2f,layers=%d  (%.1fs)\n",
                   step+1, c.max_steps, rl/lc, eff_lr,
                   (float)((step+1)*c.seq_len*c.batch_size)/el,
                   total_alive,
                   w.layers[0].parliament.consensus,
                   chuck.dampen, chuck.sigma,
                   cal_drift.drift, cal_drift.stability,
                   eph.expert_temperature, eph.active_layers, el);
            rl = 0; lc = 0;
            /* Expert load distribution (from last forward pass) */
            for (int l = 0; l < c.depth; l++) {
                LayerAct *la = &ts.layers[l];
                float efrac[MAX_EXPERTS] = {0};
                for (int t = 0; t < c.seq_len; t++) {
                    int kk = la->top_k[t];
                    for (int ki = 0; ki < kk; ki++)
                        efrac[la->top_idx[t*MAX_EXPERTS+ki]] += la->top_wt[t*MAX_EXPERTS+ki];
                }
                for (int e = 0; e < MAX_EXPERTS; e++) efrac[e] /= (float)c.seq_len;
                printf("    L%d:", l);
                for (int e = 0; e < MAX_EXPERTS; e++)
                    if (w.layers[l].experts[e].alive) printf(" e%d=%.1f%%", e, efrac[e]*100.0f);
                printf("\n");
            }
        }
    }
    printf("\n[train] done in %.1fs — births: %d, deaths: %d, mycelium: %d spores, meta: %d entries\n",
           (float)(clock()-t0)/CLOCKS_PER_SEC, total_births, total_deaths,
           mycelium.n_spores, meta.n_entries);
    printf("[meta] config biases: [%.2f, %.2f, %.2f, %.2f] prediction_error: %.4f\n",
           meta.config_bias[0], meta.config_bias[1], meta.config_bias[2], meta.config_bias[3],
           meta.prediction_error);
    if (mycelium.best_idx >= 0)
        printf("[mycelium] best spore: %s (fitness=%.2f, loss=%.4f)\n",
               mycelium.spores[mycelium.best_idx].path,
               mycelium.spores[mycelium.best_idx].fitness,
               mycelium.spores[mycelium.best_idx].loss);
    printf("[drift] final: drift=%.3f stability=%.2f accel=%.4f snapshots=%d\n",
           cal_drift.drift, cal_drift.stability, cal_drift.drift_accel, cal_drift.n_snapshots);
    if (g_tokcache_ready)
        printf("[tokcache] hits=%d misses=%d rate=%.2f\n",
               g_tokcache.total_hits, g_tokcache.total_misses, g_tokcache.hit_rate);
    int resonance = drift_find_resonance(&cal_drift);
    if (resonance > 0)
        printf("[drift] system resonates with state from %d snapshots ago (step ~%d)\n",
               resonance, cal_drift.history[(cal_drift.head - 1 - resonance + DRIFT_SNAPSHOTS*2) % DRIFT_SNAPSHOTS].step);
    printf("[env] cpu=%d mem=%.1fGB ggufs_found=%d compiler=%s curl=%s\n",
           env.cpu_count, (float)env.mem_available/(1024*1024*1024), env.n_ggufs,
           env.has_compiler?"yes":"no", env.has_curl?"yes":"no");
    if (ghost.active)
        printf("[doe] indexed %s — dim=%d layers=%d\n",
               ghost.host_path, ghost.host_dim, ghost.host_n_layers);
    if (tok_eye.code_mode)
        printf("[code] code detected: %.0f%% of input. DOE sees source.\n", tok_eye.code_ratio * 100);

    skip_training:
    /* Personality finetune — runs even if weights loaded from GGUF */
    {
        struct stat pst;
        if (stat(c.personality_path, &pst) == 0 && pst.st_size > 10) {
            int pl; char *ptxt = load_text(c.personality_path, &pl);
            if (ptxt && pl > 10) {
                int pnt; int *ptok; int *pmask;
                sft_tokenize(&tok, ptxt, pl, &ptok, &pmask, &pnt);
                int ai_tok = 0; for (int i = 0; i < pnt; i++) ai_tok += pmask[i];
                int sft_mode = (ai_tok < pnt); /* has <human>/<ai> tags */
                /* Auto personality_steps: 3 epochs over data */
                int p_steps = c.personality_steps;
                if (p_steps == 0) {
                    int effective = sft_mode ? ai_tok : pnt; /* count only AI tokens for SFT */
                    int steps_per_epoch = effective / c.seq_len;
                    if (steps_per_epoch < 1) steps_per_epoch = 1;
                    p_steps = steps_per_epoch * 3;
                    if (p_steps < 500) p_steps = 500;
                    if (p_steps > 2000) p_steps = 2000;
                }
                printf("[personality] found %s (%d tokens, %s mode, %d ai tokens), finetuning %d steps...\n",
                       c.personality_path, pnt, sft_mode ? "SFT" : "continuation", ai_tok, p_steps);
                if (weights_loaded || topology_changed) {
                    /* Rebuild params/grads/adam: needed after GGUF load or if last
                     * training step triggered mitosis/apoptosis (use-after-free fix) */
                    { for (int i = 0; i < params.count; i++) free(grads[i]); free(grads); adam_free(opt); }
                    params = collect_params(&w, &c);
                    grads = calloc(params.count, sizeof(float*));
                    for (int i = 0; i < params.count; i++) grads[i] = calloc(params.tensors[i]->size, 4);
                    opt = adam_new(&params);
                    topology_changed = 0;
                }
                int p_batch = c.batch_size;
                int p_ms = pnt - c.seq_len - 1; if (p_ms < 1) p_ms = 1;
                int *sft_targets = malloc(c.seq_len * sizeof(int));
                for (int step = 0; step < p_steps && pnt > c.seq_len + 1; step++) {
                    for (int i = 0; i < params.count; i++) memset(grads[i], 0, params.tensors[i]->size * 4);
                    float ploss = 0;
                    for (int b = 0; b < p_batch; b++) {
                        int ps = (int)(rand_uniform() * p_ms);
                        /* Build targets: mask human tokens (target=-1 skips loss) */
                        for (int t = 0; t < c.seq_len; t++)
                            sft_targets[t] = pmask[ps+t+1] ? ptok[ps+t+1] : -1;
                        ploss += train_fwd(&w, &c, &ts, ptok+ps, sft_targets, c.seq_len);
                        train_bwd(&w, &c, &ts, ptok+ps, sft_targets, c.seq_len, grads);
                    }
                    ploss /= p_batch;
                    float inv_pb = 1.0f / p_batch;
                    for (int i = 0; i < params.count; i++) for (int j = 0; j < params.tensors[i]->size; j++) grads[i][j] *= inv_pb;
                    float gn = 0; for (int i = 0; i < params.count; i++) for (int j = 0; j < params.tensors[i]->size; j++) gn += grads[i][j]*grads[i][j]; gn = sqrtf(gn);
                    if (gn > 1.0f) { float s = 1.0f/gn; for (int i = 0; i < params.count; i++) for (int j = 0; j < params.tensors[i]->size; j++) grads[i][j] *= s; }
                    adam_step(opt, &params, grads, c.lr * 0.1f, c.weight_decay);
                    if ((step+1) % 100 == 0) printf("  personality step %d/%d  loss=%.4f\n", step+1, p_steps, ploss);
                }
                free(sft_targets);
                free(pmask);
                free(ptok);
            }
            free(ptxt);
        } else if (c.personality_path[0]) printf("[personality] no %s found, skipping\n", c.personality_path);
    }
    export_gguf(&w, &c); /* always save — training or personality may have updated weights */
    if (ghost.active) {
        printf("\n[doe] auto-indexed host %s — the parliament speaks through it:\n  ", ghost.host_path);
        host_generate(&ghost, "the parliament speaks:", 64);
    }
    chat(&w, &c, &tok);

    adam_free(opt);
    free(params.tensors);
    for (int i = 0; i < params.count; i++) free(grads[i]);
    free(grads);
    free(all_tok);
    if (ghost.active) gguf_host_free(&ghost);
    printf("[m] parliament adjourned.\n");
    return 0;
}
