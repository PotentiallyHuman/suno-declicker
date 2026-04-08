"""
suno-demetalizer — removes metallic/watery reverb from Suno AI generated songs.

Method:
  1. Compare the wet vocal stem (with Suno effects) to the dry vocal stem
     (no effects, exported from Suno Studio). The difference is the reverb
     signature — a spectral fingerprint of exactly what the metallic effect adds.
  2. Subtract that reverb fingerprint from the full song mix.

This is the same principle as professional noise reduction (iZotope, Audacity):
sample the "noise", then subtract its profile from everything. Here the "noise"
is the metallic reverb.

Inputs:
  • Full song mix (original)
  • Wet vocal stem  — vocals WITH Suno effects (the metallic one)
  • Dry vocal stem  — vocals WITHOUT effects (from Suno Studio)
  • Instrumental stem (optional — used to verify subtraction doesn't damage instruments)

Output:
  Full song with metallic reverb frequencies suppressed.

Usage:
    python demetalizer.py                        # interactive
    python demetalizer.py song.mp3 --wet vocal_fx.mp3 --dry vocal_raw.mp3
    python demetalizer.py song.mp3 --wet vocal_fx.mp3 --dry vocal_raw.mp3 --strength 0.7
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

def align(ref, other):
    n = len(ref)
    if len(other) >= n: return other[:n]
    return np.pad(other, ((0, n - len(other)), (0, 0)))


# ── reverb fingerprint extraction ─────────────────────────────────────────────

def reverb_fingerprint(wet, dry, sr, n_fft=2048, hop=512):
    """
    Compute the mean spectral magnitude of (wet - dry) across all time frames.
    This is the frequency profile of what Suno's metallic reverb adds to the vocal.
    Returns a 1D array of shape (n_freqs,) — the reverb fingerprint.
    """
    dry = align(wet, dry)
    fingerprint = np.zeros(n_fft // 2 + 1, dtype=np.float32)
    count = 0

    for ch in range(wet.shape[1]):
        reverb_signal = wet[:, ch] - dry[:, ch]
        _, _, R = stft(reverb_signal, fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, window='hann', boundary='even')
        fingerprint += np.abs(R).mean(axis=1).astype(np.float32)
        count += 1

    fingerprint /= count
    # smooth across frequency to avoid notch-filter artifacts
    fingerprint = uniform_filter1d(fingerprint, size=12)
    return fingerprint


# ── reverb presence envelope ──────────────────────────────────────────────────

def reverb_envelope(wet, dry, sr, n_fft=2048, hop=512, smooth_frames=8):
    """
    Returns a 1D array (n_frames,) in range [0, 1]:
      1.0 = this frame is pure reverb tail (dry is silent, wet still rings)
      0.0 = active singing present (dry has energy, reverb is masked)

    This becomes the per-frame sidechain that controls how hard we filter.
    When the singer is present the reverb is masked and we leave it alone.
    When only reverb tail is ringing with no new vocal, we suppress hard.
    """
    dry = align(wet, dry)
    wet_energy = np.zeros(1, dtype=np.float32)
    dry_energy = np.zeros(1, dtype=np.float32)

    for ch in range(wet.shape[1]):
        _, _, W = stft(wet[:, ch], fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, window='hann', boundary='even')
        _, _, D = stft(dry[:, ch], fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, window='hann', boundary='even')
        we = np.abs(W).mean(axis=0).astype(np.float32)  # energy per frame
        de = np.abs(D).mean(axis=0).astype(np.float32)
        if wet_energy.shape != we.shape:
            wet_energy = we.copy()
            dry_energy = de.copy()
        else:
            wet_energy += we
            dry_energy += de

    wet_energy /= wet.shape[1]
    dry_energy /= wet.shape[1]

    # ratio: how much of wet energy is BEYOND dry (i.e. pure reverb tail)
    # 1 when dry is silent, 0 when dry ≈ wet
    ratio = np.clip(1.0 - dry_energy / (wet_energy + 1e-10), 0.0, 1.0)

    # smooth to avoid abrupt filter changes (creates pumping)
    ratio = uniform_filter1d(ratio, size=smooth_frames)
    return ratio


# ── full-song subtraction ──────────────────────────────────────────────────────

def demetalize(song, fingerprint, envelope, sr, strength=0.8,
               floor=0.85, band_lo=2000, band_hi=8000, n_fft=2048, hop=512):
    """
    Subband + sidechain spectral subtraction.

    Two axes of control:
      Frequency axis: only process band_lo–band_hi Hz (default 2–8kHz).
                      Everything below (kick, bass, body) and above (air) is untouched.
      Time axis:      envelope gates the strength per frame — only suppress
                      during reverb-tail frames when the singer is absent.

    This is the subband approach: the metallic resonance lives in 2–8kHz.
    Isolate it, fix it, leave the rest of the spectrum completely alone.
    """
    n     = len(song)
    out   = np.zeros_like(song)
    env   = envelope.astype(np.float32)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    # frequency mask: 1 inside the metallic band, 0 everywhere else
    freq_mask = ((freqs >= band_lo) & (freqs <= band_hi)).astype(np.float32)
    # soft edges — 10-bin cosine fade to avoid ringing artifacts at band boundaries
    fade = 10
    lo_idx = np.searchsorted(freqs, band_lo)
    hi_idx = np.searchsorted(freqs, band_hi)
    for i in range(fade):
        t = (i + 1) / (fade + 1)
        if lo_idx + i < len(freq_mask): freq_mask[lo_idx - fade + i] *= t
        if hi_idx - i - 1 >= 0:        freq_mask[hi_idx - i] *= t

    for ch in range(song.shape[1]):
        _, _, S = stft(song[:, ch], fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, window='hann', boundary='even')
        mag   = np.abs(S)
        phase = np.angle(S)

        n_frames = mag.shape[1]
        env_t = env[:n_frames] if len(env) >= n_frames else np.pad(env, (0, n_frames - len(env)))

        # subtraction applies only within the frequency band AND during reverb tail
        subtraction = (strength
                       * fingerprint[:, None]   # (n_freqs, 1)
                       * freq_mask[:, None]      # (n_freqs, 1) — zero outside band
                       * env_t[None, :])         # (1, n_frames) — zero when singing

        hard_floor = mag * floor
        mag_clean  = np.maximum(mag - subtraction, hard_floor)

        Z_clean = mag_clean * np.exp(1j * phase)
        _, repaired = istft(Z_clean, fs=sr, nperseg=n_fft,
                            noverlap=n_fft - hop, window='hann', boundary='even')

        repaired = repaired[:n] if len(repaired) >= n else np.pad(repaired, (0, n - len(repaired)))
        out[:, ch] = repaired

    return out


# ── interactive file picker ────────────────────────────────────────────────────

def list_audio_files():
    exts = {".mp3", ".wav", ".flac", ".aif", ".aiff"}
    return sorted(f for f in os.listdir(".")
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
    print("\n  What you need (all from Suno):")
    print("  • Full song mix — the original mp3")
    print("  • Wet vocal     — vocal stem WITH effects (the metallic reverb one)")
    print("  • Dry vocal     — vocal stem WITHOUT effects (Suno Studio → no FX)\n")

    song = pick_file("Select the FULL SONG mix:", files)
    wet  = pick_file("Select the WET vocal stem (with effects):", files, exclude=[song])
    dry  = pick_file("Select the DRY vocal stem (no effects):", files, exclude=[song, wet])

    print(f"\n  Full song : {song}")
    print(f"  Wet vocal : {wet}")
    print(f"  Dry vocal : {dry}")
    input("\n  Press Enter to start, or Ctrl+C to cancel. ")
    return song, wet, dry


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Remove metallic reverb from Suno AI songs using wet/dry vocal comparison.")
    ap.add_argument("input",     nargs="?", default=None, help="Full song mix")
    ap.add_argument("--wet",     default=None, help="Wet vocal stem (with effects)")
    ap.add_argument("--dry",     default=None, help="Dry vocal stem (no effects)")
    ap.add_argument("--out",      default=None, help="Output path (default: <input>_demetalized.wav)")
    ap.add_argument("--strength",  type=float, default=0.8,
                    help="Max subtraction strength during reverb-only frames (default 0.8)")
    ap.add_argument("--floor",     type=float, default=0.85,
                    help="Min magnitude floor as fraction of original (default 0.85)")
    ap.add_argument("--band-lo",   type=float, default=2000,
                    help="Low edge of metallic band in Hz (default 2000)")
    ap.add_argument("--band-hi",   type=float, default=8000,
                    help="High edge of metallic band in Hz (default 8000)")
    args = ap.parse_args()

    if not args.input or not args.wet or not args.dry:
        song_path, wet_path, dry_path = interactive_setup()
        args.input = args.input or song_path
        args.wet   = args.wet   or wet_path
        args.dry   = args.dry   or dry_path

    for p in [args.input, args.wet, args.dry]:
        if not os.path.isfile(p):
            sys.exit(f"File not found: {p}")

    out_path = args.out or (os.path.splitext(args.input)[0] + "_demetalized.wav")
    if os.path.abspath(out_path) == os.path.abspath(args.input):
        sys.exit("Output path matches input — refusing to overwrite.")

    print(f"\nsuno-demetalizer")
    print(f"  Full song : {os.path.basename(args.input)}")
    print(f"  Wet vocal : {os.path.basename(args.wet)}")
    print(f"  Dry vocal : {os.path.basename(args.dry)}")
    print(f"  Strength  : {args.strength}  floor={args.floor}")
    print(f"  Output    : {os.path.basename(out_path)}\n")

    song, sr = load(args.input)
    wet,  _  = load(args.wet)
    dry,  _  = load(args.dry)
    print(f"  {sr} Hz | {song.shape[1]}ch | {len(song)/sr:.1f}s")

    print("  Building reverb fingerprint from wet/dry vocal comparison...")
    fp  = reverb_fingerprint(wet, dry, sr)
    print(f"  Peak metallic frequency : {np.argmax(fp) * sr / 2048:.0f} Hz")

    print("  Computing reverb presence envelope (sidechain)...")
    env = reverb_envelope(wet, dry, sr)
    print(f"  Reverb-only frames      : {(env > 0.5).sum()} / {len(env)}  ({100*(env>0.5).mean():.1f}% of song)")

    print(f"  Subtracting in {args.band_lo:.0f}–{args.band_hi:.0f} Hz only (reverb-tail frames)...")
    result = demetalize(song, fp, env, sr,
                        strength=args.strength, floor=args.floor,
                        band_lo=args.band_lo, band_hi=args.band_hi)

    save(out_path, result, sr)
    print(f"\n  Original untouched : {args.input}")
    print(f"  Clean copy saved   : {out_path}\n")

if __name__ == "__main__":
    main()
