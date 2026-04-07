"""
suno-declicker — removes click/tick artifacts from Suno AI generated audio.

Detection uses three filters:
  1. Local amplitude ratio spike (>> surrounding music)
  2. Spectral fingerprint match — compared against 3 confirmed Suno v5.5 artifact
     clicks, using cosine similarity. Musical transients (hi-hats, guitar picks,
     vocal consonants) score low; Suno artifacts score high.
  3. Duration gate — artifact clicks are sub-millisecond (< 15 samples at 44.1kHz)

Removal uses cubic spline interpolation — fits a smooth curve through the audio
on both sides of the click, matching slope at the boundaries. Inaudible even on
fast-moving waveforms because only 4-10 samples are replaced.

Usage:
    python declicker.py song.mp3
    python declicker.py song.mp3 --dry-run
    python declicker.py song.mp3 --threshold 5 --similarity 0.65
"""

import sys, os, argparse, tempfile, subprocess
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.fft import rfft
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

# ── Suno click fingerprint ────────────────────────────────────────────────────
# Spectral profile (normalised FFT magnitude, 256-point Hanning window at 44.1kHz)
# derived from 3 confirmed Suno v5.5 artifact clicks.
# Similarity threshold: >0.70 = artifact-like, <0.50 = musical transient.
SUNO_CLICK_FINGERPRINT = np.array([
    0.3996, 0.4099, 0.2450, 0.1539, 0.0822, 0.0330, 0.0224, 0.0083, 0.0136,
    0.0195, 0.0214, 0.0249, 0.0323, 0.0394, 0.0428, 0.0715, 0.1124, 0.1588,
    0.1912, 0.2434, 0.2781, 0.2557, 0.2982, 0.4096, 0.3555, 0.3195, 0.3033,
    0.3169, 0.3161, 0.3779, 0.3473, 0.2496, 0.4481, 0.5223, 0.4139, 0.4384,
    0.3810, 0.3376, 0.2440, 0.1937, 0.2388, 0.2165, 0.2075, 0.1574, 0.1649,
    0.1734, 0.1993, 0.1181, 0.0931, 0.0708, 0.0706, 0.0861, 0.1135, 0.0839,
    0.1748, 0.2061, 0.2625, 0.1493, 0.1380, 0.1090, 0.0810, 0.0669, 0.1030,
    0.1116, 0.1582, 0.1175, 0.0822, 0.0947, 0.1036, 0.0947, 0.0708, 0.0799,
    0.0983, 0.0988, 0.0642, 0.0377, 0.0786, 0.0986, 0.0801, 0.0643, 0.0378,
    0.0280, 0.0447, 0.0679, 0.0711, 0.0332, 0.0335, 0.0323, 0.0266, 0.0308,
    0.0251, 0.0138, 0.0195, 0.0257, 0.0185, 0.0092, 0.0097, 0.0159, 0.0230,
    0.0316, 0.0391, 0.0304, 0.0257, 0.0148, 0.0049, 0.0051, 0.0012, 0.0002,
    0.0001, 0.00003, 0.00002, 0.00002, 0.00001, 0.00001, 0.000007, 0.000009,
    0.000006, 0.000004, 0.0000015, 0.0000083, 0.0000063, 0.0000056, 0.0000077,
    0.0000065, 0.0000045, 0.0000035, 0.0000076, 0.0000070, 0.0000021,
], dtype=np.float32)

FINGERPRINT_WIN = 256  # must match window used to build fingerprint


# ── audio I/O ─────────────────────────────────────────────────────────────────

def load(path):
    if path.lower().endswith(".mp3"):
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(["ffmpeg","-y","-i",path,"-ar","44100",tmp],
                           capture_output=True, check=True)
            data, sr = sf.read(tmp)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    else:
        data, sr = sf.read(path)
    if data.ndim == 1: data = data[:, np.newaxis]
    return data, sr

def save(path, data, sr):
    sf.write(path, data, sr, subtype="PCM_24")


# ── detection ─────────────────────────────────────────────────────────────────

def spectral_mag(data, centre, win=FINGERPRINT_WIN):
    n = len(data)
    s = max(0, centre - win // 2)
    e = min(n, s + win)
    seg = data[s:e].mean(axis=1) if data.ndim > 1 else data[s:e]
    if len(seg) < win: return None
    mag = np.abs(rfft(seg * np.hanning(win))).astype(np.float32)
    norm = mag.max()
    return mag / (norm + 1e-10) if norm > 0 else None

def cosine_sim(a, b):
    b = b[:len(a)]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def fingerprint_similarity(data, centre):
    """Cosine similarity between this position's spectrum and the Suno click fingerprint."""
    mag = spectral_mag(data, centre)
    if mag is None: return 0.0
    return cosine_sim(mag, SUNO_CLICK_FINGERPRINT)

def local_novelty(data, sr, centre):
    """How spectrally different is this position from its musical context?
    A Suno artifact is foreign to the surrounding audio. A musical transient is not."""
    mag = spectral_mag(data, centre)
    if mag is None: return 0.0
    offsets = [int(o * sr) for o in [0.15, 0.30, -0.15, -0.30]]
    neighbors = [m for o in offsets if (m := spectral_mag(data, centre + o)) is not None]
    if not neighbors: return 0.0
    local_avg = np.mean(neighbors, axis=0)
    return 1.0 - cosine_sim(mag, local_avg)

def detect_clicks(data, sr, ratio_threshold, similarity_threshold, top_n=None):
    mono = np.abs(data.mean(axis=1))
    ctx  = uniform_filter1d(mono, size=int(0.4 * sr))
    ctx  = np.maximum(ctx, 1e-6)
    ratio = mono / ctx

    # if top_n calibration mode: use permissive pre-filter, similarity gate still applies
    pre_threshold = min(ratio_threshold, 3.0) if top_n else ratio_threshold
    raw = np.where(ratio > pre_threshold)[0]
    if len(raw) == 0: return []

    groups, group = [], [raw[0]]
    for p in raw[1:]:
        if p - group[-1] <= 50: group.append(p)
        else: groups.append(group); group = [p]
    groups.append(group)

    body_end = max(0, len(mono) - int(2 * sr))
    candidates = []

    for g in groups:
        centre = g[np.argmax(mono[g])]
        if centre > body_end: continue

        fsim = fingerprint_similarity(data, centre)
        if fsim < similarity_threshold: continue

        thr   = ctx[centre] * 2
        start = centre
        while start > 0 and mono[start] > thr: start -= 1
        end = centre
        while end < len(mono) - 1 and mono[end] > thr: end += 1

        if (end - start) > 15: continue

        novelty = local_novelty(data, sr, centre)
        peak_ratio = ratio[centre]
        score = fsim * peak_ratio  # combined click confidence
        candidates.append((centre, start, end, fsim, novelty, score))

    if top_n:
        # keep top_n by score, then re-sort by time for display
        candidates.sort(key=lambda x: x[5], reverse=True)
        candidates = candidates[:top_n]
        candidates.sort(key=lambda x: x[0])

    return [(c, s, e, f, n) for c, s, e, f, n, _ in candidates]


# ── removal ───────────────────────────────────────────────────────────────────

def remove_clicks(data, clicks, sr):
    """
    Cubic spline repair — fits a smooth curve through audio on both sides
    of the click, matching slope at both boundaries.
    Only 4-10 samples are replaced per click.
    """
    out     = data.copy()
    n       = len(out)
    ctx_len = 6

    for centre, start, end, fsim, novelty in clicks:
        if end - start < 1: continue
        pre_s  = max(0, start - ctx_len)
        post_e = min(n - 1, end + ctx_len)

        x_anchor = np.concatenate([np.arange(pre_s, start + 1),
                                   np.arange(end, post_e + 1)])
        x_fill   = np.arange(start, end + 1)

        for ch in range(out.shape[1]):
            y_anchor = np.concatenate([out[pre_s:start + 1, ch],
                                       out[end:post_e + 1, ch]])
            if len(x_anchor) < 4: continue
            cs = CubicSpline(x_anchor, y_anchor)
            out[start:end + 1, ch] = cs(x_fill)

    return out


def _lp(sig, cutoff, sr):
    b, a = butter(4, cutoff / (sr / 2), btype='low')
    return filtfilt(b, a, sig)

def _hp(sig, cutoff, sr):
    b, a = butter(4, cutoff / (sr / 2), btype='high')
    return filtfilt(b, a, sig)

def remove_clicks_with_stems(data, clicks, vocal, instrumental, sr,
                              crossover_hz=2000, fade_len=20):
    """
    Band-split repair:
      Low band  (<crossover_hz): cubic spline — preserves the original's
                                  low-frequency envelope, avoids stem phase mismatch.
      High band (>crossover_hz): stem crossfade — stems are clean here and the
                                  click's broadband energy is correctly replaced.
    The two bands are repaired independently then summed.
    """
    n        = len(data)
    stem_mix = (vocal[:n] + instrumental[:n])
    if stem_mix.shape[1] < data.shape[1]:
        stem_mix = np.tile(stem_mix, (1, data.shape[1]))

    ctx_len = 6

    # split original and stems into bands — operate on full arrays once
    # (filtfilt on entire song is faster and avoids edge artefacts per-click)
    out_lo = np.zeros_like(data)
    out_hi = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out_lo[:, ch] = _lp(data[:, ch], crossover_hz, sr)
        out_hi[:, ch] = _hp(data[:, ch], crossover_hz, sr)

    stem_hi = np.zeros_like(data)
    for ch in range(data.shape[1]):
        stem_hi[:, ch] = _hp(stem_mix[:, ch], crossover_hz, sr)

    for centre, start, end, fsim, novelty in clicks:

        # ── low band: cubic spline ────────────────────────────────────────────
        if end - start >= 1:
            pre_s  = max(0, start - ctx_len)
            post_e = min(n - 1, end + ctx_len)
            x_anc  = np.concatenate([np.arange(pre_s, start + 1),
                                     np.arange(end,   post_e + 1)])
            x_fill = np.arange(start, end + 1)
            for ch in range(data.shape[1]):
                y_anc = np.concatenate([out_lo[pre_s:start+1, ch],
                                        out_lo[end:post_e+1,  ch]])
                if len(x_anc) < 4: continue
                cs = CubicSpline(x_anc, y_anc)
                out_lo[start:end+1, ch] = cs(x_fill)

        # ── high band: stem crossfade ─────────────────────────────────────────
        s = max(0, start - fade_len)
        e = min(n - 1, end + fade_len)
        length = e - s
        env = np.ones(length)
        env[:fade_len]  = np.linspace(0, 1, fade_len)
        env[-fade_len:] = np.linspace(1, 0, fade_len)
        for ch in range(data.shape[1]):
            orig_hi = out_hi[s:e, ch]
            st_hi   = stem_hi[s:e, ch]
            out_hi[s:e, ch] = orig_hi * (1 - env) + st_hi * env

    return out_lo + out_hi


# ── interactive file picker ───────────────────────────────────────────────────

def list_audio_files(directory="."):
    exts = {".mp3", ".wav", ".flac", ".aif", ".aiff"}
    files = sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts and os.path.isfile(os.path.join(directory, f))
    )
    return files

def pick_file(prompt, files, exclude=None):
    exclude = exclude or []
    available = [(i, f) for i, f in enumerate(files, 1) if f not in exclude]
    if not available:
        sys.exit("No audio files found in the current directory.")
    print(f"\n  {prompt}")
    for i, f in available:
        print(f"    {i:>3}.  {f}")
    while True:
        try:
            raw = input("\n  Enter number: ").strip()
            n = int(raw)
            match = [f for i, f in available if i == n]
            if match:
                return match[0]
        except (ValueError, EOFError):
            pass
        print("  Invalid — try again.")

def interactive_setup():
    files = list_audio_files()
    if not files:
        sys.exit("No mp3/wav/flac/aif files found in the current directory.")

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │          suno-declicker setup           │")
    print("  └─────────────────────────────────────────┘")
    print("\n  Place your files in the same folder and run this script.")
    print("  The original is NEVER modified — a clean copy is saved.\n")

    original    = pick_file("Select the ORIGINAL full mix:", files)
    vocal       = pick_file("Select the VOCAL stem:", files, exclude=[original])
    instrumental = pick_file("Select the INSTRUMENTAL stem:", files, exclude=[original, vocal])

    print(f"\n  Original      : {original}")
    print(f"  Vocal stem    : {vocal}")
    print(f"  Instrumental  : {instrumental}")
    confirm = input("\n  Looks good? Press Enter to start, or Ctrl+C to cancel. ")

    return original, vocal, instrumental


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Remove Suno AI click/tick artifacts from generated audio.")
    ap.add_argument("input",        nargs="?", default=None,
                    help="Original audio file (omit to use interactive mode)")
    ap.add_argument("--out",        default=None,
                    help="Output path (default: <input>_clean.wav)")
    ap.add_argument("--threshold",  type=float, default=5.0,
                    help="Local amplitude ratio to flag candidate (default 5.0)")
    ap.add_argument("--similarity", type=float, default=0.70,
                    help="Min fingerprint similarity to confirm Suno artifact (default 0.70)")
    ap.add_argument("--vocal",       default=None,
                    help="Vocal stem (mp3/wav) for stem-based repair")
    ap.add_argument("--instrumental", default=None,
                    help="Instrumental stem (mp3/wav) for stem-based repair")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Detect and report only — do not save output")
    args = ap.parse_args()

    # interactive mode when no file is provided on the command line
    if args.input is None:
        original, vocal_path, instr_path = interactive_setup()
        args.input        = original
        args.vocal        = args.vocal or vocal_path
        args.instrumental = args.instrumental or instr_path

    if not os.path.isfile(args.input):
        sys.exit(f"Error: file not found: {args.input}")

    out_path = args.out or (os.path.splitext(args.input)[0] + "_clean.wav")

    # safety: never overwrite the original
    if os.path.abspath(out_path) == os.path.abspath(args.input):
        sys.exit("Error: output path matches input — refusing to overwrite original.")

    print(f"\nsuno-declicker")
    print(f"  Input    : {os.path.basename(args.input)}")
    if not args.dry_run:
        print(f"  Output   : {os.path.basename(out_path)}")
    print()

    data, sr = load(args.input)
    print(f"  {sr} Hz | {data.shape[1]}ch | {len(data)/sr:.1f}s\n")

    clicks = detect_clicks(data, sr, args.threshold, args.similarity)

    if not clicks:
        print("  No Suno artifact clicks detected.")
        if not args.dry_run:
            save(out_path, data, sr)
    else:
        print(f"  Found {len(clicks)} click(s):\n")
        print(f"  {'#':>4}  {'Time':>8}  {'Dur':>7}  {'Match':>7}  {'Novelty':>8}")
        print(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}")
        for i, (c, s, e, fsim, nov) in enumerate(clicks):
            t = c / sr
            m, sec = divmod(t, 60)
            dur_ms = (e - s) / sr * 1000
            print(f"  {i+1:>4}  {int(m):02d}:{sec:05.2f}  {dur_ms:>5.2f}ms  {fsim:>7.3f}  {nov:>8.3f}")

        if not args.dry_run:
            if args.vocal and args.instrumental:
                print("\n  Repairing with stems (original kept everywhere else)...")
                vocal, _  = load(args.vocal)
                instr, _  = load(args.instrumental)
                data = remove_clicks_with_stems(data, clicks, vocal, instr, sr)
            else:
                print("\n  Repairing with cubic spline...")
                data = remove_clicks(data, clicks, sr)
            save(out_path, data, sr)
            print(f"\n  Original untouched : {args.input}")
            print(f"  Clean copy saved   : {out_path}")
    print()

if __name__ == "__main__":
    main()
