#!/usr/bin/env python3
"""Check `auditwheel show` output against the wheel's platform tag (standard library only).

  auditwheel_tag.py AUDITWHEEL_SHOW_OUTPUT manylinux_2_28_x86_64

auditwheel names the oldest manylinux policy the wheel's files satisfy ('... is consistent with the following platform
tag: "manylinux_2_27_x86_64"'); the wheel's tag must be at least that new. auditwheel wraps its sentences at 70 columns
and the break points move with the length of the wheel's file name (0.2.0 vs 0.2.0.post12), so the text is matched
with every run of white space read as one space. Exit status 0 when the tag is satisfied, 1 otherwise."""
import re
import sys


def satisfied(text: str, tag: str):
    want = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", tag)
    if not want:
        raise ValueError(f"not a manylinux tag: {tag}")
    flat = " ".join(text.split())
    m = re.search(r'consistent with the following platform tag: "manylinux_(\d+)_(\d+)_(\w+)"', flat)
    if not m or m.group(3) != want.group(3):
        return None
    oldest = (int(m.group(1)), int(m.group(2)))
    return oldest if oldest <= (int(want.group(1)), int(want.group(2))) else False


def main() -> int:
    text, tag = open(sys.argv[1], encoding="utf-8").read(), sys.argv[2]
    got = satisfied(text, tag)
    if not got:
        print(f"auditwheel: the wheel does not satisfy {tag}" + (" (no platform tag in the output)" if got is None else ""))
        return 1
    print(f"auditwheel: OK (satisfies manylinux_{got[0]}_{got[1]}, tagged {tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
