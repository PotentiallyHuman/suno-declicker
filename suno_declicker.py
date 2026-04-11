#!/usr/bin/env python3
"""
suno-declicker — interactive wizard for removing clicks from Suno AI songs.

Uses the original song and its instrumental stem to isolate the vocal layer,
erase click artifacts, and reconstruct a clean full mix.

Run with:
    python3 suno_declicker.py
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENHANCE    = os.path.join(SCRIPT_DIR, "suno_enhance.py")


# ─────────────────────────────────────────────────────────────────────────────

def _enter(msg=None):
    if msg:
        print(f"\n  {msg}")
    input("  [Press Enter to continue]\n")


def _mp3s(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp3"))


def _wait_for_exactly(folder, count, action_hint):
    while True:
        files = _mp3s(folder)
        n = len(files)
        if n == count:
            return files
        if n < count:
            missing = count - n
            print(f"\n  ! {n} file(s) in folder — need {count}.  "
                  f"({missing} more to add: {action_hint})")
        else:
            extra = n - count
            print(f"\n  ! {n} file(s) in folder — only {count} should be there.  "
                  f"Remove {extra} extra file(s).")
        input("  [Press Enter to check again]\n")


def _open_folder(path):
    try:
        subprocess.run(["xdg-open", path], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(ENHANCE):
        sys.exit(f"  Error: suno_enhance.py not found in {SCRIPT_DIR}\n"
                 f"  Keep this script in the same folder as suno_enhance.py.")

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │               suno-declicker wizard                  │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  This tool removes clicks from Suno AI generated songs.")
    print("  It uses the song's instrumental stem to isolate the vocal,")
    print("  erase the click artifacts, and rebuild a clean full mix.")
    print()
    print("  Requirements:")
    print("    • Suno Pro subscription")
    print("    • At least 20 credits (one instrumental export ≈ 10–20 credits)")
    print()
    print("  You will need:")
    print("    • The original Suno song (with clicks)")
    print("    • The instrumental stem exported from Suno Studio")
    print("      (beat grid snap OFF — use the native song tempo)")
    print()
    _enter()

    # ── Step 1 — Instrumental export ─────────────────────────────────────────
    print("  ┌─ Step 1 of 3 — Export the instrumental from Suno Studio ─┐")
    print()
    print("  1. Open your song in Suno Studio.")
    print("  2. IMPORTANT: disable beat grid snap before exporting.")
    print("     (Snap ON causes timing drift that misaligns the stems.)")
    print("  3. Export  Instrumental  (not the full song).")
    print("  4. Download the file.")
    print()
    _enter("Done downloading the instrumental?")

    # ── Song name + folder ───────────────────────────────────────────────────
    while True:
        name = input("  Song name (used for folder name): ").strip()
        if name:
            break
        print("  Please enter a name.")

    folder = os.path.join(SCRIPT_DIR, name)
    os.makedirs(folder, exist_ok=True)
    print(f"\n  Folder: {folder}")
    _open_folder(folder)

    print()
    print("  Place ONLY the instrumental file into that folder.")
    print()
    input("  [Press Enter when it is in the folder]\n")
    files_after_1   = _wait_for_exactly(folder, 1, "the instrumental file")
    instrumental_name = files_after_1[0]
    instrumental      = os.path.join(folder, instrumental_name)
    print(f"\n  Registered as instrumental: {instrumental_name}")

    # ── Step 2 — Original song ────────────────────────────────────────────────
    print()
    print("  ┌─ Step 2 of 3 — Original song ─────────────────────────────┐")
    print()
    print("  Place your original Suno song (the one with the clicks)")
    print("  into the same folder.")
    print()
    input("  [Press Enter when it is in the folder]\n")
    files_after_2 = _wait_for_exactly(folder, 2, "the original song")
    original_name = next(f for f in files_after_2 if f != instrumental_name)
    original      = os.path.join(folder, original_name)
    print(f"\n  Registered as original: {original_name}")

    # ── Step 3 — Declicking ───────────────────────────────────────────────────
    print()
    print("  ┌─ Step 3 of 3 — Declicking ────────────────────────────────┐")
    print()
    print(f"  Original     : {original_name}")
    print(f"  Instrumental : {instrumental_name}")
    print()
    _enter("Press Enter to start — this may take a minute.")

    out_name = f"{name}_declicked.mp3"
    out_path = os.path.join(folder, out_name)

    cmd = [
        sys.executable, ENHANCE,
        original,
        "--instrumental", instrumental,
        "--repair-mode",  "median",
        "--out",          out_path,
        "--no-compare",
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("\n  Declicker failed — see output above.")

    print(f"\n  Done!")
    print(f"  Output: {out_name}")
    print(f"  Folder: {folder}")
    print()


if __name__ == "__main__":
    main()
