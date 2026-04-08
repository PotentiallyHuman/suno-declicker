"""
suno-demetalizer — removes the metallic/watery reverb artifact from Suno AI vocals.

The "metallic reverb" problem: Suno's AI-generated reverb has unnatural resonances
in the 2–6 kHz range that ring longer than a real room would. The result sounds
metallic, watery, or tunnel-like — the most complained-about artifact in AI music.

How it works:
  1. Load the wet vocal (with Suno's effects) and the dry vocal (no effects,
     exported from Suno Studio).
  2. In STFT domain, compute where the wet vocal has energy but the dry has
     already decayed — that is the reverb tail.
  3. Find which frequency bands in the tail are over-represented (ringing too
     long) compared to a natural exponential decay profile.
  4. Suppress those bands with a smooth spectral mask.
  5. Reconstruct. You get a vocal with natural-sounding reverb decay — the
     metallic resonances are gone.

Optionally: apply a natural convolution reverb to the dry vocal instead.

Usage:
    python demetalizer.py                          # interactive
    python demetalizer.py --wet vocal.mp3 --dry vocal_nofx.mp3
    python demetalizer.py --wet vocal.mp3 --dry vocal_nofx.mp3 --suppress 0.7
"""

import sys, os, argparse, tempfile, subprocess
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from scipy.ndimage import uniform_filter1d

# ── audio I/O ─────────────────────────────────────────────────────────────────

def load(path, sr_target=44100):
    if path.lower().endswith(".mp3"):
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(["ffmpeg","-y","-i",path,"-ar",str(sr_target),tmp],
                           capture_output=True, check=True)
            data, sr = sf.read(tmp)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    else:
        data, sr = sf.read(path)
    if data.ndim == 1: data = data[:, None]
    return data, sr

def save(path, data, sr):
    sf.write(path, data, sr, subtype="PCM_24")

def align(a, b):
    """Trim or pad b to match length of a."""
    n = len(a)
    if len(b) >= n: return b[:n]
    return np.pad(b, ((0, n - len(b)), (0, 0)))


# ── core demetalizer ──────────────────────────────────────────────────────────

def demetalize(wet, dry, sr, suppress=0.8, n_fft=2048, hop=512):
    """
    suppress: 0.0 = no change, 1.0 = fully remove metallic resonances.
    """
    dry = align(wet, dry)
    out = np.zeros_like(wet)

    for ch in range(wet.shape[1]):
        w = wet[:, ch]
        d = dry[:, ch]

        _, _, W = stft(w, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, window='hann', boundary='even')
        _, _, D = stft(d, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, window='hann', boundary='even')

        mag_w = np.abs(W)   # (n_freqs, n_frames)
        mag_d = np.abs(D)
        phase = np.angle(W)  # keep wet phase — only magnitude is modified

        # ── identify reverb tail ──────────────────────────────────────────────
        # reverb tail = frames where wet has energy but dry has decayed
        dry_floor  = mag_d.max() * 0.01  # -40dB below dry peak = "dry is gone"
        in_tail    = mag_d < dry_floor   # True where dry has decayed

        # ── measure over-resonance per frequency bin ──────────────────────────
        # natural reverb decays exponentially — energy in tail should be low.
        # we compute the mean energy in the tail vs the mean in the body for each bin.
        # bins where tail_energy / body_energy is high are the metallic resonators.
        body_energy = np.where(~in_tail, mag_w, 0).mean(axis=1) + 1e-10
        tail_energy = np.where( in_tail, mag_w, 0).mean(axis=1) + 1e-10
        resonance   = tail_energy / body_energy  # high = over-resonant = metallic

        # normalise resonance to a 0-1 suppression mask per frequency bin
        r_min, r_max = resonance.min(), resonance.max()
        if r_max > r_min:
            mask_1d = (resonance - r_min) / (r_max - r_min)  # 0=natural, 1=metallic
        else:
            mask_1d = np.zeros_like(resonance)

        # smooth the mask across frequencies to avoid notch-filter sound
        mask_1d = uniform_filter1d(mask_1d, size=8)

        # suppress = how much of the metallic resonance to remove (0–1)
        gain_1d = 1.0 - suppress * mask_1d          # (n_freqs,)
        gain    = gain_1d[:, None]                   # broadcast over time

        # apply gain only in the tail — body of the vocal is untouched
        tail_broadcast = in_tail[None, :] if in_tail.ndim == 1 else in_tail
        # in_tail is (n_freqs, n_frames) — apply per-bin gain only in tail frames
        gain_full = np.where(in_tail, gain, 1.0)     # (n_freqs, n_frames)

        mag_out = mag_w * gain_full
        Z_out   = mag_out * np.exp(1j * phase)

        _, repaired = istft(Z_out, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                            window='hann', boundary='even')
        n = len(w)
        repaired = repaired[:n] if len(repaired) >= n else np.pad(repaired, (0, n-len(repaired)))
        out[:, ch] = repaired

    return out


# ── interactive picker ────────────────────────────────────────────────────────

def list_audio_files(directory="."):
    exts = {".mp3", ".wav", ".flac", ".aif", ".aiff"}
    return sorted(f for f in os.listdir(directory)
                  if os.path.splitext(f)[1].lower() in exts and os.path.isfile(f))

def pick_file(prompt, files, exclude=None):
    exclude = exclude or []
    available = [(i, f) for i, f in enumerate(files, 1) if f not in exclude]
    if not available:
        sys.exit("No audio files found.")
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
    print("  │         suno-demetalizer setup          │")
    print("  └─────────────────────────────────────────┘")
    print("\n  You need two files from Suno Studio:")
    print("  • Wet vocal  — vocal stem WITH effects (the one that sounds metallic)")
    print("  • Dry vocal  — vocal stem WITHOUT effects (export from Suno Studio)\n")

    wet = pick_file("Select the WET vocal (with effects):", files)
    dry = pick_file("Select the DRY vocal (no effects):", files, exclude=[wet])

    print(f"\n  Wet vocal : {wet}")
    print(f"  Dry vocal : {dry}")
    input("\n  Press Enter to start, or Ctrl+C to cancel. ")
    return wet, dry


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Remove metallic reverb artifacts from Suno AI vocal stems.")
    ap.add_argument("--wet",      default=None, help="Wet vocal stem (with effects)")
    ap.add_argument("--dry",      default=None, help="Dry vocal stem (no effects)")
    ap.add_argument("--out",      default=None, help="Output path (default: <wet>_demetalized.wav)")
    ap.add_argument("--suppress", type=float, default=0.8,
                    help="Suppression strength 0.0–1.0 (default 0.8)")
    args = ap.parse_args()

    if not args.wet or not args.dry:
        args.wet, args.dry = interactive_setup()

    for p in [args.wet, args.dry]:
        if not os.path.isfile(p):
            sys.exit(f"File not found: {p}")

    out_path = args.out or (os.path.splitext(args.wet)[0] + "_demetalized.wav")
    if os.path.abspath(out_path) == os.path.abspath(args.wet):
        sys.exit("Output path matches input — refusing to overwrite.")

    print(f"\nsuno-demetalizer")
    print(f"  Wet vocal : {os.path.basename(args.wet)}")
    print(f"  Dry vocal : {os.path.basename(args.dry)}")
    print(f"  Output    : {os.path.basename(out_path)}")
    print(f"  Suppress  : {args.suppress}\n")

    wet, sr = load(args.wet)
    dry, _  = load(args.dry)
    print(f"  {sr} Hz | {wet.shape[1]}ch | {len(wet)/sr:.1f}s")

    print("  Analysing reverb tail and suppressing metallic resonances...")
    result = demetalize(wet, dry, sr, suppress=args.suppress)

    save(out_path, result, sr)
    print(f"\n  Original untouched : {args.wet}")
    print(f"  Clean copy saved   : {out_path}\n")

if __name__ == "__main__":
    main()
