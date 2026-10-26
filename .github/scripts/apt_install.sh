#!/bin/bash
# apt-get install for CI with retries: apt-get on the hosted runners has hung for 18 to 30 minutes (a slow mirror or
# the dpkg lock held by another apt run). Each try is cut off after 4 minutes and tried again, up to 3 times, and the
# output stays in the log so a slow step shows where it waits.
#   .github/scripts/apt_install.sh PACKAGE...
set -u
APT=(-q -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o DPkg::Lock::Timeout=120)
for try in 1 2 3; do
  if sudo timeout 240 apt-get "${APT[@]}" update && sudo timeout 240 apt-get "${APT[@]}" install -y "$@"; then
    exit 0
  fi
  echo "::warning::apt-get try $try of 3 failed or timed out; trying again"
  sudo dpkg --configure -a || true
  sleep 10
done
echo "::error::apt-get install $* failed 3 times"
exit 1
