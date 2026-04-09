#!/usr/bin/env python3
import os
import subprocess
import tempfile
import getpass
from pathlib import Path


def ask_yes_no(prompt):
    ans = input(prompt + " (y/N): ").strip().lower()
    return ans == "y"


def open_folder(path):
    try:
        subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


def list_files(folder):
    files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    for i, f in enumerate(files, 1):
        print(f"{i}: {f}")
    return files


def choose_file(folder, prompt):
    files = list_files(folder)
    while True:
        choice = input(prompt).strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return os.path.join(folder, files[idx])
        print("Invalid selection. Try again.")


def ensure_wav(input_path, out_dir, sr=44100):
    """
    Converts mp3/flac/etc -> wav if needed.
    Always returns a WAV path.
    """
    input_path = str(input_path)

    if input_path.lower().endswith(".wav"):
        return input_path

    out_path = os.path.join(out_dir, Path(input_path).stem + ".wav")

    print(f"\nConverting to WAV:\n  {input_path}\n  -> {out_path}\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sr),
        "-ac", "2",
        out_path
    ], check=True)

    return out_path


def count_patch_files(patches_dir):
    if not os.path.exists(patches_dir):
        return 0
    return len([f for f in os.listdir(patches_dir) if f.endswith("_info.txt")])


def run_click_detection(clean_wav, patches_dir):
    """
    Generates click patches.
    This calls click_remover.py which should create *_info.txt files.
    """
    cmd = [
        "python3",
        "/home/openclaw/suno-declicker/click_remover.py",
        clean_wav,
        "--patches_dir", patches_dir
    ]

    print("\nRunning click detection:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_regenerate(clean_wav, regen_wav, patches_dir, output_wav, patch_strength=0.5, patch_window=0.06):
    cmd = [
        "python3",
        "/home/openclaw/suno-declicker/regenerate_patches.py",
        clean_wav,
        regen_wav,
        "--patches_dir", patches_dir,
        "--output_file", output_wav,
        "--patch_strength", str(patch_strength),
        "--patch_window", str(patch_window)
    ]

    print("\nRunning regeneration:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_compare(a_wav, b_wav):
    cmd = [
        "python3",
        "/home/openclaw/suno-declicker/compare.py",
        a_wav,
        b_wav
    ]

    print("\nLaunching A/B comparison:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    print("=== Final Mastering Tool ===\n")

    user = getpass.getuser()
    base_dir = Path(f"/home/{user}/suno-declicker")

    song_name = input("Name of song? ").strip()
    session_dir = base_dir / song_name

    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated folder: {session_dir}")

    do_deshimmer = ask_yes_no("\nDo you want deshimmer?")
    do_demetalizer = ask_yes_no("Do you want demetalizer?")

    # Determine required files based on features
    required = ["Original _clean file"]

    if do_deshimmer or do_demetalizer:
        required += [
            "Instrumental stem",
            "Vocal stem with effects",
            "Vocal stem dry"
        ]

    print("\nPlease place the following files into the folder:")
    for r in required:
        print(f"  - {r}")

    open_folder(session_dir)
    input("\nPress ENTER when done to continue...")

    # Select required files
    selections = {}
    for r in required:
        print()
        selections[r] = choose_file(session_dir, f"Select {r} (#): ")

    print("\nSelections confirmed:")
    for k, v in selections.items():
        print(f"  {k}: {v}")

    # Convert clean to wav
    clean_input = selections["Original _clean file"]
    clean_wav = ensure_wav(clean_input, session_dir)

    # Create click patches folder
    patches_dir = session_dir / "click_patches"
    patches_dir.mkdir(exist_ok=True)

    # Step 1: Click detection (generate patches)
    before_count = count_patch_files(patches_dir)
    run_click_detection(clean_wav, str(patches_dir))
    after_count = count_patch_files(patches_dir)

    print(f"\nPatch files before: {before_count}")
    print(f"Patch files after : {after_count}")

    if after_count == 0:
        print("\nERROR: click detection produced 0 patch files.")
        print("That means click_remover.py isn't writing *_info.txt patches.")
        print("Fix click_remover.py output first.")
        return

    # Step 2: Regenerate
    regen_simple_wav = session_dir / f"{song_name}_regenerated_simple.wav"

    # If no pre-existing regenerated_simple exists, just use clean as base regen input
    if regen_simple_wav.exists():
        regen_simple_wav = str(regen_simple_wav)
    else:
        regen_simple_wav = clean_wav

    output_wav = session_dir / f"{song_name}_regenerated_gentler.wav"

    print(f"\nOutput will be:\n  {output_wav}")

    run_regenerate(
        clean_wav=clean_wav,
        regen_wav=regen_simple_wav,
        patches_dir=str(patches_dir),
        output_wav=str(output_wav),
        patch_strength=0.5,
        patch_window=0.06
    )

    print("\nDeclick/regeneration complete.")

    # Optional compare
    if ask_yes_no("\nPlay song comparison (original vs enhanced)?"):
        run_compare(clean_wav, str(output_wav))

    print(f"\nAll done! Files saved in:\n  {session_dir}")


if __name__ == "__main__":
    main()
