#!/usr/bin/env bash
# End-to-end test of kitten-tts-native on a CUDA GPU with a prepared bundle.
#
#   KT2N_TEST_ASSETS=models/kitten bash tests/cli/e2e.sh
#
# KT2N_TEST_ASSETS   bundle directory (tools/prepare_assets.py); unset = skip (exit 77), or a failure
#                    when KT2N_REQUIRE_GPU=1 (set by `ctest --preset gpu`)
# KT2N_CLI_BINARY    kitten-tts-native executable (default: the one on PATH)
# KT2N_TEST_REQUIRE_CLONING=1
#                    fail instead of skipping the cloning steps when the bundle has no cloning files
# KT2N_TEST_KEEP=1   keep the work directory
#
# Covers: --version, --check-assets, --list-voices (every built-in voice of the prepared model); streaming WAV to a file;
# PCM and WAV to standard output (streamed and complete); sampling settings reaching the engine
# (--preset, and an out-of-range value of each of --temperature, --top-p, --min-p, --top-k and
# --max-tokens refused by the engine under its own name); --no-normalize; --voice-cache,
# --voice-cache-voices, --voice-cache-mib and --memory-profile (the same complete-mode audio as the
# defaults, and a budget smaller than one voice refused by the engine); --stream false (one
# complete file rendered segment by segment, not the streamed audio, with the output mode read from
# the engine's response headers); --repeat with --report; cloning with --save-voice; --load-voice;
# Ctrl-C (SIGINT) cancellation of a request that would otherwise run far longer than the time
# allowed for cancelling.
#
# Not covered here: the per-segment generation budgets of --stream false. They are internal to the
# engine (no CLI option sets them); tests/cpp/textplan_test.cc (Text.Budgets) checks the budget
# rules.
set -euo pipefail

if [[ -z "${KT2N_TEST_ASSETS:-}" ]]; then
  if [[ "${KT2N_REQUIRE_GPU:-0}" == 1 ]]; then
    echo "FAIL: set KT2N_TEST_ASSETS to a prepared model directory (KT2N_REQUIRE_GPU=1 turns this skip into a failure)"
    exit 1
  fi
  echo "SKIP: set KT2N_TEST_ASSETS to a prepared bundle directory"
  exit 77
fi
BIN="${KT2N_CLI_BINARY:-$(command -v kitten-tts-native || true)}"
[[ -x "$BIN" ]] || { echo "FAIL: kitten-tts-native not found (set KT2N_CLI_BINARY)"; exit 1; }
ASSETS="$KT2N_TEST_ASSETS"
PY="${PYTHON:-python3}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/kt2n-cli-e2e.XXXXXX")"
if [[ "${KT2N_TEST_KEEP:-0}" != 1 ]]; then trap 'rm -rf "$WORK"' EXIT; else echo "work directory: $WORK"; fi

step() { echo "== $*"; }
fail() { echo "FAIL: $*"; exit 1; }

# check_wav FILE: a PCM16 mono 24 kHz WAV whose data chunk matches the file; prints its duration.
check_wav() {
  "$PY" - "$1" <<'EOF'
import struct, sys
data = open(sys.argv[1], "rb").read()
assert len(data) > 44, f"{sys.argv[1]}: {len(data)} bytes"
riff, size, wave, fmt, fmt_size, tag, channels, rate, byte_rate, align, bits, chunk, data_size = \
    struct.unpack("<4sI4s4sIHHIIHH4sI", data[:44])
assert (riff, wave, fmt, chunk) == (b"RIFF", b"WAVE", b"fmt ", b"data"), "not a canonical WAV header"
assert (fmt_size, tag, channels, rate, byte_rate, align, bits) == (16, 1, 1, 24000, 48000, 2, 16), "not PCM16 mono 24 kHz"
assert data_size == len(data) - 44 and size == data_size + 36, f"header sizes {size}/{data_size} for {len(data)} bytes"
seconds = data_size / 48000
assert seconds > 0, "no audio"
print(f"{seconds:.2f}")
EOF
}

# check_pcm FILE: non-empty whole 16-bit samples; prints the duration.
check_pcm() {
  local bytes
  bytes=$(stat -c %s "$1")
  (( bytes > 0 && bytes % 2 == 0 )) || fail "$1: $bytes PCM bytes"
  "$PY" -c "print(f'{$bytes / 48000:.2f}')"
}

# report FILE EXPR: evaluates a Python expression over the report object `r` (tuples print space-separated).
report() {
  "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); v=($2); print(*v) if isinstance(v, tuple) else print(v)" "$1"
}

SHORT="Hello! This is a short test of the speech engine."
REF_TEXT="The quick brown fox jumps over the lazy dog, then runs back across the quiet field before the sun goes down."
cat > "$WORK/long.txt" <<'EOF'
The library opened at nine in the morning, and the first visitors were already waiting on the steps. A student carried a stack of books about the history of the river, while an old man asked where he could find the newspapers from last spring. Inside, the reading room was quiet except for the soft hum of the lamps. By noon the tables were full, and the librarian had answered more than forty questions, most of them about maps, train timetables and the opening hours on public holidays. In the afternoon a class of children arrived for a story, and they listened without a sound until the very last page.
EOF

step "version"
"$BIN" --version | tee "$WORK/version.txt"
grep -q '^kitten-tts-native ' "$WORK/version.txt" || fail "--version output"

step "check assets"
"$BIN" --assets "$ASSETS" --check-assets | tee "$WORK/check.txt"
CLONING=0
grep -q 'Voice cloning: available' "$WORK/check.txt" && CLONING=1

step "list voices"
"$BIN" --assets "$ASSETS" --list-voices > "$WORK/voices.txt"
VOICES=$(wc -l < "$WORK/voices.txt")
echo "$VOICES voices: $(cut -f1 "$WORK/voices.txt" | head -5 | tr '\n' ' ')..."
# The prepared voices file decides the count (the model revision sets it): every voice in its metadata
# that is not marked hidden.
EXPECTED=$("$PY" - "$ASSETS" <<'EOF'
import json, pathlib, struct, sys
root = pathlib.Path(sys.argv[1])
entry = json.loads((root / "config.json").read_text(encoding="utf-8"))["native"]["voices"]["file"]
data = (root / entry).read_bytes()
magic, _, _, meta_off, meta_len = struct.unpack_from("<8sIIQQ", data)
assert magic == b"KT2NVOX1", magic
meta = json.loads(data[meta_off:meta_off + meta_len].decode("utf-8"))
print(sum(1 for v in meta["voices"] if v.get("hidden") is not True))
EOF
)
(( VOICES >= 1 && VOICES == EXPECTED )) || fail "--list-voices printed $VOICES voices, expected the $EXPECTED built-in voices of the voices file"
grep -q -x -E 'Bruno(\s.*)?' "$WORK/voices.txt" || fail "--list-voices does not list Bruno (the voice of the README examples)"
FIRST_VOICE=$(head -1 "$WORK/voices.txt" | cut -f1)

step "stream WAV to a file"
"$BIN" --assets "$ASSETS" --voice "$FIRST_VOICE" --text "$REF_TEXT" --seed 11 --output "$WORK/stream.wav" \
  --report "$WORK/stream-wav.json"
echo "duration $(check_wav "$WORK/stream.wav") s"
[[ "$(report "$WORK/stream-wav.json" 'r["stream"], r["format"], r["audio_seconds"] > 0, r["rtf"] is not None')" == "True wav True True" ]] \
  || fail "streaming WAV report: $(cat "$WORK/stream-wav.json")"

step "PCM to standard output"
"$BIN" --assets "$ASSETS" --text "$SHORT" --format pcm --output - --seed 12 > "$WORK/stdout.pcm" 2> "$WORK/stdout.err"
echo "duration $(check_pcm "$WORK/stdout.pcm") s"
tail -1 "$WORK/stdout.err" > "$WORK/stdout.json"
[[ "$(report "$WORK/stdout.json" 'r["bytes"]')" == "$(stat -c %s "$WORK/stdout.pcm")" ]] || fail "stdout byte count"

step "streamed WAV to standard output"
"$BIN" --assets "$ASSETS" --text "$SHORT" --format wav --output - --seed 12 > "$WORK/stdout.wav" 2> /dev/null
"$PY" - "$WORK/stdout.wav" <<'EOF'
import struct, sys
data = open(sys.argv[1], "rb").read()
riff, size, wave, fmt, fmt_size, tag, channels, rate, byte_rate, align, bits, chunk, data_size = \
    struct.unpack("<4sI4s4sIHHIIHH4sI", data[:44])
assert (riff, wave, fmt, chunk) == (b"RIFF", b"WAVE", b"fmt ", b"data"), "not a WAV header"
assert (tag, channels, rate, bits) == (1, 1, 24000, 16), "not PCM16 mono 24 kHz"
assert size == data_size == 0xFFFFFFFF, f"length not marked unknown: {size:#x}/{data_size:#x}"
assert len(data) > 44 and (len(data) - 44) % 2 == 0, f"{len(data)} bytes"
print(f"header marks the length unknown; {(len(data) - 44) / 48000:.2f} s of audio")
EOF

step "complete WAV to standard output"
"$BIN" --assets "$ASSETS" --text "$SHORT" --format wav --output - --seed 12 --stream false \
  > "$WORK/stdout-complete.wav" 2> /dev/null
echo "duration $(check_wav "$WORK/stdout-complete.wav") s with the final lengths in the header"

step "sampling settings reach the engine"
"$BIN" --assets "$ASSETS" --text "$SHORT" --format pcm --seed 21 --stream false --preset stable \
  --output "$WORK/stable.pcm" 2> /dev/null
"$BIN" --assets "$ASSETS" --text "$SHORT" --format pcm --seed 21 --stream false --preset expressive \
  --output "$WORK/expressive.pcm" 2> /dev/null
if cmp -s "$WORK/stable.pcm" "$WORK/expressive.pcm"; then fail "--preset stable and expressive gave the same audio"; fi
echo "--preset stable and --preset expressive differ for the same seed"
set +e
"$BIN" --assets "$ASSETS" --text "$SHORT" --stream false --max-tokens 20 --output "$WORK/refused.wav" \
  2> "$WORK/refused.err"
STATUS=$?
set -e
(( STATUS == 1 )) || fail "--max-tokens 20: expected exit status 1, got $STATUS"
grep -q 'max_new_tokens must be an integer between 500 and 12000' "$WORK/refused.err" \
  || fail "--max-tokens 20: the engine's range error is missing: $(cat "$WORK/refused.err")"
echo "the engine refuses --max-tokens 20 (allowed range 500-12000)"
# refused OPTION VALUE FIELD: the engine (not the command-line parser) refuses VALUE and names FIELD,
# so each option reaches its own request field.
refused() {
  local status
  printf 'previous take\n' > "$WORK/refused.wav"
  set +e
  "$BIN" --assets "$ASSETS" --text "$SHORT" --stream false "$1" "$2" --output "$WORK/refused.wav" \
    2> "$WORK/refused.err"
  status=$?
  set -e
  (( status == 1 )) || fail "$1 $2: expected exit status 1, got $status"
  [[ "$(cat "$WORK/refused.wav")" == "previous take" ]] || fail "$1 $2: the refused run changed the existing output file"
  grep -Eq "$3 must be an? (number|integer) between" "$WORK/refused.err" \
    || fail "$1 $2: the engine's $3 range error is missing: $(cat "$WORK/refused.err")"
  echo "the engine refuses $1 $2 as $3"
}
refused --temperature 0.1 temperature
refused --top-p 0.2 top_p
refused --min-p 0.7 min_p
refused --top-k 300 top_k

step "--no-normalize reaches the engine"
NUMBERS='On 12/05/2024 the shop sold 1,234 items for $56.78 each.'
"$BIN" --assets "$ASSETS" --text "$NUMBERS" --format pcm --seed 22 --stream false \
  --output "$WORK/normalized.pcm" --report "$WORK/normalized.json" 2> /dev/null
"$BIN" --assets "$ASSETS" --text "$NUMBERS" --format pcm --seed 22 --stream false --no-normalize \
  --output "$WORK/raw.pcm" --report "$WORK/raw.json" 2> /dev/null
if cmp -s "$WORK/normalized.pcm" "$WORK/raw.pcm"; then fail "--no-normalize gave the same audio"; fi
echo "audio tokens: normalised $(report "$WORK/normalized.json" 'r["engine"]["audio_tokens"]')," \
  "--no-normalize $(report "$WORK/raw.json" 'r["engine"]["audio_tokens"]'); the audio differs"

step "voice cache and memory profile options"
# Complete-mode audio for one request at a time is the same in every cache mode and memory profile.
for opts in "--voice-cache lru --voice-cache-voices 1" "--voice-cache none" \
            "--voice-cache lru --voice-cache-mib 2048 --memory-profile low"; do
  # shellcheck disable=SC2086
  "$BIN" --assets "$ASSETS" --text "$NUMBERS" --format pcm --seed 22 --stream false $opts \
    --output "$WORK/cache.pcm" 2> /dev/null
  cmp -s "$WORK/normalized.pcm" "$WORK/cache.pcm" || fail "$opts: the audio differs from the default settings"
  echo "$opts: same audio as the default settings"
done
set +e
"$BIN" --assets "$ASSETS" --text "$SHORT" --stream false --voice-cache lru --voice-cache-mib 1 \
  --output "$WORK/refused.wav" 2> "$WORK/refused.err"
STATUS=$?
set -e
(( STATUS == 1 )) || fail "--voice-cache-mib 1: expected exit status 1, got $STATUS"
grep -q 'cannot hold the largest voice' "$WORK/refused.err" \
  || fail "--voice-cache-mib 1: the engine's budget error is missing: $(cat "$WORK/refused.err")"
echo "the engine refuses --voice-cache-mib 1 (smaller than one voice)"

step "--stream false versus streaming, same seed"
"$BIN" --assets "$ASSETS" --text-file "$WORK/long.txt" --format pcm --seed 1234 --stream true \
  --output "$WORK/long-stream.pcm" --report "$WORK/long-stream.json"
"$BIN" --assets "$ASSETS" --text-file "$WORK/long.txt" --format pcm --seed 1234 --stream false \
  --output "$WORK/long-complete.pcm" --report "$WORK/long-complete.json"
"$BIN" --assets "$ASSETS" --text-file - --format wav --seed 1234 --stream false \
  --output "$WORK/long-complete.wav" < "$WORK/long.txt"
echo "streamed $(check_pcm "$WORK/long-stream.pcm") s, complete $(check_pcm "$WORK/long-complete.pcm") s," \
  "complete WAV $(check_wav "$WORK/long-complete.wav") s"
echo "stream:   $(report "$WORK/long-stream.json" 'r["engine"]')"
echo "complete: $(report "$WORK/long-complete.json" 'r["engine"]')"
# Complete mode renders every segment whole: one render row per segment. Several sentences give
# several segments. Streaming renders each segment in several commits.
[[ "$(report "$WORK/long-complete.json" 'r["engine"]["output"]')" == complete ]] || fail "complete mode: the engine's headers do not say complete"
[[ "$(report "$WORK/long-stream.json" 'r["engine"]["output"]')" == stream ]] || fail "streaming: the engine's headers do not say stream"
[[ "$(report "$WORK/long-complete.json" 'r["engine"]["segments"] >= 2')" == True ]] || fail "complete mode: expected several segments"
[[ "$(report "$WORK/long-complete.json" 'r["engine"]["render_rows"] == r["engine"]["segments"]')" == True ]] \
  || fail "complete mode: render rows != segments (not whole-segment rendering)"
[[ "$(report "$WORK/long-stream.json" 'r["engine"]["render_rows"] > r["engine"]["segments"]')" == True ]] \
  || fail "streaming: expected more render rows than segments"
if cmp -s "$WORK/long-stream.pcm" "$WORK/long-complete.pcm"; then fail "--stream false output equals the streamed output"; fi
echo "complete-mode PCM differs from the streamed PCM for the same seed"
tail -c +45 "$WORK/long-complete.wav" | cmp -s - "$WORK/long-complete.pcm" \
  || fail "--stream false: the WAV samples differ from the PCM output for the same seed"
echo "complete-mode WAV samples equal the complete-mode PCM for the same seed"

step "--repeat 2 with --report"
"$BIN" --assets "$ASSETS" --text "$SHORT" --seed 23 --stream false --repeat 2 --output "$WORK/repeat.wav" \
  --report "$WORK/repeat.json" 2> "$WORK/repeat.err"
echo "duration $(check_wav "$WORK/repeat.wav") s"
grep '^{' "$WORK/repeat.err" > "$WORK/repeat.lines" || fail "--repeat 2 printed no report: $(cat "$WORK/repeat.err")"
"$PY" - "$WORK/repeat.lines" "$WORK/repeat.json" <<'EOF'
import json, sys
runs = [json.loads(line) for line in open(sys.argv[1])]
assert [r["iteration"] for r in runs] == [1, 2], f"report lines: {runs}"
last = json.load(open(sys.argv[2]))
assert last == runs[1], f"--report file {last} is not the last run {runs[1]}"
a, b = runs
assert a["bytes"] == b["bytes"] > 0, "the two runs wrote different amounts of audio"
# Engine counters are per run (differences of the engine statistics), not running totals.
for key in ("renders", "render_rows", "segments", "audio_tokens"):
    assert a["engine"][key] == b["engine"][key] > 0, f"engine {key}: {a['engine'][key]} then {b['engine'][key]}"
print("two reports (iterations 1 and 2) with equal per-run counters; the file holds the last one")
EOF

if (( CLONING )); then
  # Both runs use --stream false: streamed audio is split into decoder chunks by timing, so
  # only complete-mode audio is byte-comparable between runs.
  step "clone from a reference recording and save the voice"
  "$BIN" --assets "$ASSETS" --reference-audio "$WORK/stream.wav" --reference-transcript "$REF_TEXT" \
    --save-voice "$WORK/me.voice" --text "$SHORT" --seed 13 --stream false --output "$WORK/clone.wav" \
    --report "$WORK/clone.json"
  [[ -s "$WORK/me.voice" ]] || fail "--save-voice wrote no file"
  echo "voice file $(stat -c %s "$WORK/me.voice") bytes; clone duration $(check_wav "$WORK/clone.wav") s"
  [[ "$(report "$WORK/clone.json" 'r["voice"]')" == reference ]] || fail "clone report voice"

  step "speak with the saved voice"
  "$BIN" --assets "$ASSETS" --load-voice "$WORK/me.voice" --text "$SHORT" --seed 13 --stream false \
    --output "$WORK/load.wav" --report "$WORK/load.json"
  echo "duration $(check_wav "$WORK/load.wav") s, voice $(report "$WORK/load.json" 'r["voice"]')"
  cmp -s "$WORK/clone.wav" "$WORK/load.wav" || fail "the saved voice does not reproduce the cloned voice for the same seed"
  echo "the loaded voice reproduces the cloned voice's audio byte for byte (same text and seed)"
else
  [[ "${KT2N_TEST_REQUIRE_CLONING:-0}" == 1 ]] && fail "the bundle has no cloning files"
  echo "SKIP clone, --save-voice and --load-voice: the bundle has no cloning files"
fi

step "Ctrl-C cancels a stream"
# About 30000 characters (the engine accepts at most 40000): without cancellation this runs far
# longer than the CANCEL_LIMIT seconds allowed below, so only a cancel that reaches the engine passes.
CANCEL_LIMIT=5
for _ in $(seq 1 50); do cat "$WORK/long.txt"; done > "$WORK/longer.txt"
"$BIN" --assets "$ASSETS" --text-file "$WORK/longer.txt" --output "$WORK/cancel.wav" 2> "$WORK/cancel.err" &
PID=$!
for _ in $(seq 1 1200); do
  [[ -f "$WORK/cancel.wav" ]] && (( $(stat -c %s "$WORK/cancel.wav") > 44 )) && break
  kill -0 "$PID" 2> /dev/null || break
  sleep 0.1
done
kill -0 "$PID" 2> /dev/null || fail "the process ended before any audio arrived: $(cat "$WORK/cancel.err")"
SENT=$(date +%s.%N)
kill -INT "$PID"
set +e
wait "$PID"
STATUS=$?
set -e
ELAPSED=$("$PY" -c "import time; print(f'{time.time() - $SENT:.2f}')")
echo "exit status $STATUS after ${ELAPSED} s: $(tail -1 "$WORK/cancel.err")"
(( STATUS == 130 )) || fail "expected exit status 130 after SIGINT"
grep -q 'cancelled' "$WORK/cancel.err" || fail "no cancellation message"
"$PY" -c "import sys; sys.exit(0 if float('$ELAPSED') < $CANCEL_LIMIT else 1)" \
  || fail "cancellation took ${ELAPSED} s (limit ${CANCEL_LIMIT} s)"
echo "partial WAV $(check_wav "$WORK/cancel.wav") s with a valid header"

echo "PASS: kitten-tts-native end-to-end (cloning steps: $([[ $CLONING == 1 ]] && echo run || echo skipped))"
