"""Render social.html to site/img/social-preview.png (1280x640) with headless Chrome.

    python tools/social_preview/render.py [--chrome PATH]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "site" / "img" / "social-preview.png"
CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome",
]


def find_chrome() -> str:
    for c in CANDIDATES:
        path = c if os.path.isabs(c) and os.path.exists(c) else shutil.which(c)
        if path:
            return path
    sys.exit("Chrome/Chromium not found; pass --chrome PATH")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default=None)
    args = ap.parse_args()
    chrome = args.chrome or find_chrome()
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([
            chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--force-device-scale-factor=1", "--window-size=1280,640",
            "--virtual-time-budget=3000",  # let the web fonts load
            f"--user-data-dir={profile}", f"--screenshot={OUT}",
            (HERE / "social.html").as_uri(),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
