# janus.DoE — Democracy of Experts
# One C file. Zero dependencies. Experts are born (mitosis), die (apoptosis),
# and vote on every token (variable-k parliament). DoE trains its own organism
# and modulates host GGUFs through a Hebbian LoRA parliament — the host provides
# the weights, DoE provides the direction.

CC      ?= cc
CFLAGS  ?= -O3
LDFLAGS  = -lm -lpthread

.PHONY: all blas openblas cuda test weights run run-smollm360 clean

# ─── build ────────────────────────────────────────────────────────────
# plain CPU → ./m
all: m
m: janusdoe.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

# macOS Accelerate (~3-4x on CPU)
blas: janusdoe.c
	$(CC) $(CFLAGS) -DUSE_BLAS -DACCELERATE $< $(LDFLAGS) -framework Accelerate -o m

# OpenBLAS (Linux, ~3-4x on CPU)
openblas: janusdoe.c
	$(CC) $(CFLAGS) -DUSE_BLAS $< $(LDFLAGS) -lopenblas -o m

# cuBLAS TF32 (NVIDIA A100/H100, ~25x)
cuda: janusdoe.c
	$(CC) $(CFLAGS) -DUSE_CUBLAS $< $(LDFLAGS) -lcublas -lcudart -o m

# ─── test ─────────────────────────────────────────────────────────────
# smoke tests — the parliament demands accountability
test: m
	./test.sh

# ─── DoE weights (host GGUFs the parliament wraps) ────────────────────
# Pattern mirrors canonical doe — give it a Qwen and the parliament lights up.
# Weights live on HF; the run target indexes one read-only and votes through it.
# (uploads land here later)
HF_BASE = https://huggingface.co/ataeff/janus/resolve/main/DoE
WEIGHTS = weights

$(WEIGHTS)/doe_qwen15b_lora_1000.gguf:
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/doe_qwen15b_lora_1000.gguf

$(WEIGHTS)/doe_smollm360_lora_1000.gguf:
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/doe_smollm360_lora_1000.gguf

weights: $(WEIGHTS)/doe_qwen15b_lora_1000.gguf

# default showcase: Qwen2.5-1.5B host, parliament votes on every token
run: m $(WEIGHTS)/doe_qwen15b_lora_1000.gguf
	./m --host $(WEIGHTS)/doe_qwen15b_lora_1000.gguf --ask "Are you conscious?"

# SmolLM2-360M DoE voice — custom export is NeoX-laid-out, force the pairing
run-smollm360: m $(WEIGHTS)/doe_smollm360_lora_1000.gguf
	./m --host $(WEIGHTS)/doe_smollm360_lora_1000.gguf --rope-neox --ask "Are you conscious?"

clean:
	rm -f m m_test m_test_blas
