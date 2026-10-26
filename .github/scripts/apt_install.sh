#!/bin/bash
# apt-get install for CI. The runners' Ubuntu mirror is sometimes slow (about 100 kB/s: installing ffmpeg then took 18
# to 30 minutes). Recommended packages are left out (for ffmpeg: hardware video drivers, a third of the download), a
# stalled connection is dropped after 30 s and retried, a failed install is tried again (3 tries, each at most 20
# minutes), and apt's output stays in the log, so a slow step shows where it waits.
#   .github/scripts/apt_install.sh PACKAGE...
set -u
APT=(-q -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o DPkg::Lock::Timeout=120)
for try in 1 2 3; do
  if sudo timeout 1200 apt-get "${APT[@]}" update && \
     sudo timeout 1200 apt-get "${APT[@]}" install -y --no-install-recommends "$@"; then
    exit 0
  fi
  echo "::warning::apt-get try $try of 3 failed or timed out; trying again"
  sudo dpkg --configure -a || true
  sleep 10
done
echo "::error::apt-get install $* failed 3 times"
exit 1
