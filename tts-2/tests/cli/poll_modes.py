#!/usr/bin/env python3
"""kitten-tts-native with the conductor's poll mode on (the default) and off (KT2N_CONDUCTOR_POLL=0).

Each run must finish within a time limit (on Windows, a launch-submission bug once made poll mode wait forever at
0% GPU), and a seeded --stream false request must give the same audio in both modes. The low-memory profile keeps
it small enough for most GPUs.

  KT2N_TEST_ASSETS   prepared model directory; unset = skip (exit 77), or a failure when KT2N_REQUIRE_GPU=1
  KT2N_CLI_BINARY    kitten-tts-native executable
"""
import os
import subprocess
import sys
import tempfile
import time

TEXT = "The ferry left the harbour a little after dawn. It costs 25 dollars and 50 cents."
LIMIT_S = 240


def main() -> int:
    assets = os.environ.get("KT2N_TEST_ASSETS", "")
    binary = os.environ.get("KT2N_CLI_BINARY", "")
    if not assets:
        print("SKIP: set KT2N_TEST_ASSETS to a prepared model directory")
        return 1 if os.environ.get("KT2N_REQUIRE_GPU") == "1" else 77
    if not os.path.isfile(binary):
        print(f"FAIL: KT2N_CLI_BINARY is not a file: {binary!r}")
        return 1
    with tempfile.TemporaryDirectory() as work:
        outputs = {}
        for stream in ("true", "false"):
            for poll in ("1", "0"):
                out = os.path.join(work, f"stream-{stream}-poll-{poll}.wav")
                env = dict(os.environ, KT2N_CONDUCTOR_POLL=poll)
                t0 = time.monotonic()
                try:
                    subprocess.run([binary, "--assets", assets, "--text", TEXT, "--voice", "Bruno", "--seed", "7",
                                    "--stream", stream, "--output", out, "--memory-profile", "low",
                                    "--voice-cache", "none"], env=env, timeout=LIMIT_S, check=True,
                                   capture_output=True)
                except subprocess.TimeoutExpired:
                    print(f"FAIL: --stream {stream} with KT2N_CONDUCTOR_POLL={poll} did not finish in {LIMIT_S} s")
                    return 1
                except subprocess.CalledProcessError as e:
                    print(f"FAIL: --stream {stream} with KT2N_CONDUCTOR_POLL={poll} exited {e.returncode}: "
                          f"{e.stderr.decode(errors='replace')[-500:]}")
                    return 1
                with open(out, "rb") as f:
                    outputs[(stream, poll)] = f.read()
                print(f"--stream {stream} poll {poll}: {len(outputs[(stream, poll)])} bytes in "
                      f"{time.monotonic() - t0:.1f} s")
                if len(outputs[(stream, poll)]) <= 44:
                    print("FAIL: no audio")
                    return 1
        if outputs[("false", "1")] != outputs[("false", "0")]:
            print("FAIL: seeded --stream false audio differs between poll mode on and off")
            return 1
    print("PASS: poll mode on and off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
