"""
suno-pipeline — full enhancement pipeline for Suno AI generated songs.

Steps:
  1. Declicker  — remove broadband click/tick artifacts (STFT spectral repair)
  2. Polish     — normalize to -14 LUFS (streaming standard)

Note: the demetalizer is NOT applied to the full mix here — applying a vocal
reverb fingerprint to drums and bass damages transients and muddies low end.
Run demetalizer.py separately on the vocal stem only if needed.

Usage:
    python pipeline.py                    # interactive
    python pipeline.py song.mp3 --wet vocal_fx.mp3 --dry vocal_raw.mp3
"""

import sys, os, argparse, tempfile, subprocess
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter1d

# ── import shared functions from sibling scripts ───────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from declicker import load, save, detect_clicks, remove_clicks_spectral


# ── demetalizer (clean, no instrumental replacement) ──────────────────────────

def demetalize(song, fingerprint, sr, strength=0.35, floor=0.75, n_fft=2048, hop=512):
    """
    Spectral subtraction of the reverb fingerprint.
    floor=0.75 means no frequency can be reduced more than 25% — keeps depth.
    No instrumental replacement — just careful subtraction.
    """
    n   = len(song)
    out = np.zeros_like(song)

    for ch in range(song.shape[1]):
        _, _, S = stft(song[:, ch], fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, window='hann', boundary='even')
        mag   = np.abs(S)
        phase = np.angle(S)

        mag_clean = np.maximum(mag - strength * fingerprint[:, None], mag * floor)

        Z_clean = mag_clean * np.exp(1j * phase)
        _, rep  = istft(Z_clean, fs=sr, nperseg=n_fft,
                        noverlap=n_fft - hop, window='hann', boundary='even')
        rep = rep[:n] if len(rep) >= n else np.pad(rep, (0, n - len(rep)))
        out[:, ch] = rep

    return out


# ── polish: loudness + air ─────────────────────────────────────────────────────

def lufs_integrated(data, sr, block=0.4, hop=0.1):
    """Approximate integrated LUFS (EBU R128)."""
    from scipy.signal import lfilter
    # K-weighting pre-filter (simplified)
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])
    mono = data.mean(axis=1) if data.ndim > 1 else data
    weighted = lfilter(b1, a1, mono)
    weighted = lfilter(b2, a2, weighted)
    block_n  = int(block * sr)
    hop_n    = int(hop * sr)
    powers   = []
    for i in range(0, len(weighted) - block_n, hop_n):
        block_pow = np.mean(weighted[i:i+block_n] ** 2)
        if block_pow > 1e-10:
            powers.append(block_pow)
    if not powers:
        return -70.0
    return -0.691 + 10 * np.log10(np.mean(powers))

def high_shelf(data, sr, gain_db=1.5, freq=8000):
    """Gentle high shelf boost to restore air."""
    from scipy.signal import bilinear
    A  = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    S  = 1.0
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
    b0 =      A * ((A+1) + (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha)
    b1 = -2 * A * ((A-1) + (A+1)*np.cos(w0))
    b2 =      A * ((A+1) + (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha)
    a0 =           (A+1) - (A-1)*np.cos(w0) + 2*np.sqrt(A)*alpha
    a1 =  2 *     ((A-1) - (A+1)*np.cos(w0))
    a2 =           (A+1) - (A-1)*np.cos(w0) - 2*np.sqrt(A)*alpha
    from scipy.signal import lfilter
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out[:, ch] = lfilter([b0/a0, b1/a0, b2/a0], [1, a1/a0, a2/a0], data[:, ch])
    return out

def polish(data, sr, target_lufs=-14.0, shelf_db=1.5):
    current = lufs_integrated(data, sr)
    gain    = 10 ** ((target_lufs - current) / 20)
    data    = data * gain
    data    = high_shelf(data, sr, gain_db=shelf_db)
    # safety limiter — never clip
    peak = np.abs(data).max()
    if peak > 0.98:
        data = data * (0.98 / peak)
    print(f"  Loudness: {current:.1f} LUFS → {target_lufs:.1f} LUFS  (gain {20*np.log10(gain):+.1f} dB)")
    return data


# ── interactive picker ─────────────────────────────────────────────────────────

def list_audio_files():
    exts = {".mp3", ".wav", ".flac", ".aif", ".aiff"}
    return sorted(f for f in os.listdir(".")
                  if os.path.splitext(f)[1].lower() in exts and os.path.isfile(f))

def pick_file(prompt, files, exclude=None):
    exclude = exclude or []
    available = [(i, f) for i, f in enumerate(files, 1) if f not in exclude]
    if not available:
        sys.exit("No audio files left to choose from.")
    print(f"\n  {prompt}")
    for i, f in available:
        print(f"    {i:>3}.  {f}")
    while True:
        try:
            n = int(input("\n  Enter number: ").strip())
            match = [f for i, f in available if i == n]
            if match: return match[0]
        except (ValueError, EOFError):
            pass
        print("  Invalid — try again.")

def interactive_setup():
    files = list_audio_files()
    if not files:
        sys.exit("No audio files found in the current directory.")
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │       suno enhancement pipeline         │")
    print("  └─────────────────────────────────────────┘\n")
    song  = pick_file("Select the FULL SONG mix (original):", files)
    wet   = pick_file("Select the WET vocal stem (with effects):", files, exclude=[song])
    dry   = pick_file("Select the DRY vocal stem (no effects):", files, exclude=[song, wet])
    instr = pick_file("Select the INSTRUMENTAL stem:", files, exclude=[song, wet, dry])
    print(f"\n  Song         : {song}")
    print(f"  Wet vocal    : {wet}")
    print(f"  Dry vocal    : {dry}")
    print(f"  Instrumental : {instr}")
    input("\n  Press Enter to start, or Ctrl+C to cancel. ")
    return song, wet, dry, instr


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Full Suno enhancement pipeline.")
    ap.add_argument("input",         nargs="?", default=None)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--threshold",   type=float, default=5.0)
    ap.add_argument("--similarity",  type=float, default=0.70)
    ap.add_argument("--target-lufs", type=float, default=-14.0)
    args = ap.parse_args()

    if not args.input:
        files = list_audio_files()
        args.input = pick_file("Select the FULL SONG mix:", files)

    out_path = args.out or (os.path.splitext(args.input)[0] + "_enhanced.wav")

    print(f"\n{'━'*52}")
    print(f"  suno enhancement pipeline")
    print(f"{'━'*52}")
    print(f"  Input    : {os.path.basename(args.input)}")
    print(f"  Output   : {os.path.basename(out_path)}\n")

    song, sr = load(args.input)
    print(f"  {sr} Hz | {song.shape[1]}ch | {len(song)/sr:.1f}s\n")

    # ── step 1: declicker ────────────────────────────────────────────────────
    print("  [1/2] Declicker — scanning for artifacts...")
    clicks = detect_clicks(song, sr, args.threshold, args.similarity)
    print(f"        Found {len(clicks)} click(s) — repairing with STFT spectral repair...")
    song = remove_clicks_spectral(song, clicks, sr)
    print(f"        Done.\n")

    # ── step 2: polish ───────────────────────────────────────────────────────
    print("  [2/2] Polish — loudness normalization...")
    song = polish(song, sr, target_lufs=args.target_lufs, shelf_db=0)
    print(f"        Done.\n")

    save(out_path, song, sr)
    print(f"  Original untouched : {args.input}")
    print(f"  Enhanced copy saved: {out_path}")
    print(f"{'━'*52}\n")

if __name__ == "__main__":
    main()
