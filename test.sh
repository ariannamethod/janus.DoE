#!/bin/bash
# test.sh — smoke tests for janusdoe.c / DOE (Democracy of Experts)
# the parliament demands accountability.

set -e
PASS=0
FAIL=0

ok() { PASS=$((PASS+1)); echo "  PASS $1"; }
fail() { FAIL=$((FAIL+1)); echo "  FAIL $1"; }

echo ""
echo "  DOE — Democracy of Experts — Smoke Tests"
echo "  the parliament will now be audited."
echo ""

# ═══ Test 1: Compilation ═══
echo "[test] compilation..."
cc janusdoe.c -O3 -lm -lpthread -o m_test 2>/dev/null && ok "plain compile" || fail "plain compile"

# ═══ Test 2: BLAS compilation (macOS) ═══
if [ "$(uname)" = "Darwin" ]; then
    cc janusdoe.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o m_test_blas 2>/dev/null && ok "BLAS compile (Accelerate)" || fail "BLAS compile"
    rm -f m_test_blas
fi

# ═══ Test 3: Help flag ═══
echo "[test] CLI..."
./m_test --help 2>&1 | grep -q "Democracy of Experts" && ok "--help" || fail "--help"

# ═══ Test 4: Training run ═══
echo "[test] training (depth 2, synthetic data, ~30s)..."
cat > test_data.txt << 'TESTDATA'
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Neural networks learn complex patterns from data.
Mixture of experts routes tokens to specialized networks.
The parliament votes on which expert handles each token.
Self-aware systems monitor their own performance metrics.
TESTDATA
# Repeat data ~100x for sufficient training
cp test_data.txt test_data_seed.txt
for i in $(seq 1 100); do cat test_data_seed.txt; done >> test_data.txt
rm -f test_data_seed.txt

./m_test --depth 2 --data test_data.txt > test_output.txt 2>&1 &
TEST_PID=$!

# Wait for training to start
for i in $(seq 1 90); do
    if grep -q "step " test_output.txt 2>/dev/null; then break; fi
    sleep 1
done
# Let it train a bit
sleep 15
kill $TEST_PID 2>/dev/null; wait $TEST_PID 2>/dev/null || true

# ═══ Verify ═══
grep -q "parliament" test_output.txt && ok "parliament initialized" || fail "no parliament"
grep -q "step " test_output.txt && ok "training steps" || fail "no training steps"
grep -q "experts=" test_output.txt && ok "expert tracking" || fail "no expert tracking"
grep -q "consensus=" test_output.txt && ok "consensus tracking" || fail "no consensus"
grep -q "drift=" test_output.txt && ok "calendar drift" || fail "no drift"
grep -q "chuck:" test_output.txt && ok "chuck optimizer" || fail "no chuck"
grep -q "eph:" test_output.txt && ok "ephemeral config" || fail "no ephemeral"

# ═══ Loss decreases ═══
FIRST=$(grep "step " test_output.txt | head -1 | sed 's/.*loss=\([0-9.]*\).*/\1/')
LAST=$(grep "step " test_output.txt | tail -1 | sed 's/.*loss=\([0-9.]*\).*/\1/')
if [ -n "$FIRST" ] && [ -n "$LAST" ]; then
    echo "$FIRST > $LAST" | bc -l 2>/dev/null | grep -q "1" && ok "loss decreased ($FIRST -> $LAST)" || fail "loss stalled ($FIRST -> $LAST)"
else
    fail "parse loss"
fi

# ═══ Mycelium ═══
[ -d "mycelium" ] && ok "mycelium dir" || fail "no mycelium"

# ═══ Environment Scanner ═══
grep -q "\[env\] cpu=" test_output.txt && ok "environment scan" || fail "no env scan"
grep -q "compiler=" test_output.txt && ok "compiler detection" || fail "no compiler detect"

# ═══ Code Detection ═══
grep -q "code=" test_output.txt && ok "code detection" || fail "no code detect"

# ═══ Sizes ═══
LINES=$(wc -l < janusdoe.c | tr -d ' ')
[ "$LINES" -gt 3000 ] && ok "janusdoe.c ${LINES} lines" || fail "janusdoe.c too small"

# Cleanup
rm -f m_test test_data.txt test_data_seed.txt test_output.txt

echo ""
echo "  Results: ${PASS} passed, ${FAIL} failed"
[ $FAIL -eq 0 ] && echo "  the parliament approves." || echo "  the parliament is disappointed."
exit $FAIL
