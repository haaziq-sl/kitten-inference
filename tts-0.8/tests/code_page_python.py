"""Windows only: copy this Python's interpreter with a manifest that selects a locale's ANSI code page.

Narrow C file APIs read paths in the process's ANSI code page. GitHub's Windows runners use UTF-8 (65001) as
the system code page, where such paths happen to work; most users' PCs use a legacy code page such as 1252.
The copy lets a test run in that setting. Needs Windows 11 or Server 2025 (manifest activeCodePage with a
locale name) and mt.exe from the Windows SDK. Prints the path of the new interpreter.

  python code_page_python.py [--locale en-US]
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile


def find_mt():
    found = glob.glob(os.path.expandvars(r"%ProgramFiles(x86)%\Windows Kits\10\bin\10.*\x64\mt.exe"))
    if not found:
        raise SystemExit("mt.exe not found in the Windows SDK")
    return max(found, key=lambda p: [int(x) for x in re.findall(r"\d+", p.split("\\bin\\")[1].split("\\")[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", default="en-US")
    args = ap.parse_args()

    base = getattr(sys, "_base_executable", sys.executable)
    # Next to the original, so it finds python3XX.dll and the standard library.
    copy = os.path.join(os.path.dirname(base), f"python-{args.locale}.exe")
    shutil.copyfile(base, copy)
    mt = find_mt()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = os.path.join(tmp, "python.manifest")
        subprocess.run([mt, "-nologo", f"-inputresource:{base};#1", f"-out:{manifest}"], check=True)
        text = open(manifest, encoding="utf-8").read()
        print("embedded manifest of", base, file=sys.stderr)
        print(text, file=sys.stderr)
        setting = (
            '<activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/WindowsSettings">'
            f"{args.locale}</activeCodePage>"
        )
        if "activeCodePage" in text:
            text = re.sub(r"<activeCodePage[^>]*>[^<]*</activeCodePage>", setting, text)
        elif "</windowsSettings>" in text:
            text = text.replace("</windowsSettings>", setting + "</windowsSettings>", 1)
        else:
            text = text.replace(
                "</assembly>",
                '<application xmlns="urn:schemas-microsoft-com:asm.v3"><windowsSettings>'
                + setting
                + "</windowsSettings></application></assembly>",
            )
        open(manifest, "w", encoding="utf-8").write(text)
        subprocess.run([mt, "-nologo", "-manifest", manifest, f"-outputresource:{copy};#1"], check=True)
    print(copy)


if __name__ == "__main__":
    main()
