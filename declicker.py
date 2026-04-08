"""
suno-declicker — removes click/tick artifacts from Suno AI generated audio.

Detection: spectral fingerprint + amplitude ratio + duration gate
  Fingerprint derived from confirmed Suno v5.5 artifact clicks via cosine
  similarity. Rejects hi-hats, snares, picks. No false positives on drums.

Repair: per-band STFT spectral interpolation
  Audio split into 6 perceptual bands. Each band repaired independently via
  STFT magnitude interpolation from surrounding frames. Narrower signal =
  more accurate reconstruction. Bands recombined. Spectrogram is seamless.

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
from scipy.signal import butter, sosfilt, stft, istft

# ── Suno click fingerprint ────────────────────────────────────────────────────
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

FINGERPRINT_WIN = 256


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


# ── detection (fingerprint-based) ─────────────────────────────────────────────

def spectral_mag(data, centre, win=FINGERPRINT_WIN):
    n   = len(data)
    s   = max(0, centre - win // 2)
    e   = min(n, s + win)
    seg = data[s:e].mean(axis=1) if data.ndim > 1 else data[s:e]
    if len(seg) < win: return None
    mag  = np.abs(rfft(seg * np.hanning(win))).astype(np.float32)
    norm = mag.max()
    return mag / (norm + 1e-10) if norm > 0 else None

def cosine_sim(a, b):
    b = b[:len(a)]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def fingerprint_similarity(data, centre):
    mag = spectral_mag(data, centre)
    if mag is None: return 0.0
    return cosine_sim(mag, SUNO_CLICK_FINGERPRINT)

def detect_clicks(data, sr, threshold=5.0, similarity_threshold=0.70):
    mono = np.abs(data.mean(axis=1))
    ctx  = uniform_filter1d(mono, size=int(0.4 * sr))
    ctx  = np.maximum(ctx, 1e-6)
    ratio = mono / ctx

    raw = np.where(ratio > threshold)[0]
    if len(raw) == 0: return []

    groups, group = [], [raw[0]]
    for p in raw[1:]:
        if p - group[-1] <= 50: group.append(p)
        else: groups.append(group); group = [p]
    groups.append(group)

    body_end = max(0, len(mono) - int(2 * sr))
    clicks   = []

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

        clicks.append((centre, start, end, fsim))

    return clicks


# ── filterbank ────────────────────────────────────────────────────────────────

BANDS = [
    (20,    200),
    (200,   800),
    (800,   3000),
    (3000,  6000),
    (6000,  12000),
    (12000, 20000),
]

def bandpass(data, lo, hi, sr, order=4):
    nyq = sr / 2
    sos = butter(order, [max(lo,20)/nyq, min(hi, nyq*0.999)/nyq],
                 btype='band', output='sos')
    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out[:, ch] = sosfilt(sos, data[:, ch])
    return out

def split_bands(data, sr):
    return [bandpass(data, lo, hi, sr) for lo, hi in BANDS]

def merge_bands(bands):
    return sum(bands)


# ── per-band STFT repair ──────────────────────────────────────────────────────

def repair_band(band_data, clicks, sr, n_fft=2048, hop=512, ctx_frames=6):
    out = band_data.copy()
    n   = len(band_data)

    for ch in range(band_data.shape[1]):
        sig = band_data[:, ch]
        _, _, Z = stft(sig, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                       window='hann', boundary='even')
        mag      = np.abs(Z)
        phase    = np.angle(Z)
        n_frames = Z.shape[1]

        for centre, start, end, _ in clicks:
            f0 = max(ctx_frames, start // hop)
            f1 = min(n_frames - 1 - ctx_frames, end // hop + 1)
            if f0 > f1:
                f0 = f1 = (start + end) // 2 // hop

            pre_mag  = mag[:, max(0, f0-ctx_frames):f0].mean(axis=1)
            post_mag = mag[:, f1+1:min(n_frames, f1+1+ctx_frames)].mean(axis=1)
            n_repair = f1 - f0 + 1

            for fi, frame in enumerate(range(f0, f1 + 1)):
                t = (fi + 1) / (n_repair + 1)
                mag[:, frame]   = pre_mag * (1-t) + post_mag * t
                phase[:, frame] = (phase[:, max(0,f0-1)] * (1-t) +
                                   phase[:, min(n_frames-1,f1+1)] * t)

        Z_fixed = mag * np.exp(1j * phase)
        _, rep  = istft(Z_fixed, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                        window='hann', boundary='even')
        rep = rep[:n] if len(rep) >= n else np.pad(rep, (0, n-len(rep)))
        out[:, ch] = rep

    return out


def remove_clicks_spectral(data, clicks, sr):
    """
    Split into 6 perceptual bands → repair each independently → recombine.
    Narrower band signal = more accurate STFT magnitude interpolation.
    """
    if not clicks: return data
    bands    = split_bands(data, sr)
    repaired = [repair_band(b, clicks, sr) for b in bands]
    return merge_bands(repaired)


# ── interactive file picker ───────────────────────────────────────────────────

def list_audio_files(directory="."):
    exts = {".mp3", ".wav", ".flac", ".aif", ".aiff"}
    return sorted(f for f in os.listdir(directory)
                  if os.path.splitext(f)[1].lower() in exts
                  and os.path.isfile(os.path.join(directory, f)))

def pick_file(prompt, files, exclude=None):
    exclude = exclude or []
    available = [(i, f) for i, f in enumerate(files, 1) if f not in exclude]
    if not available: sys.exit("No audio files found.")
    print(f"\n  {prompt}")
    for i, f in available:
        print(f"    {i:>3}.  {f}")
    while True:
        try:
            n = int(input("\n  Enter number: ").strip())
            match = [f for i, f in available if i == n]
            if match: return match[0]
        except (ValueError, EOFError): pass
        print("  Invalid — try again.")

def interactive_setup():
    files = list_audio_files()
    if not files: sys.exit("No audio files found in the current directory.")
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │          suno-declicker setup           │")
    print("  └─────────────────────────────────────────┘")
    print("\n  The original is NEVER modified — a clean copy is saved.\n")
    original = pick_file("Select the ORIGINAL full mix:", files)
    print(f"\n  Selected: {original}")
    input("  Press Enter to start, or Ctrl+C to cancel. ")
    return original


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Remove click artifacts — fingerprint detection, per-band STFT repair.")
    ap.add_argument("input",        nargs="?", default=None)
    ap.add_argument("--out",        default=None)
    ap.add_argument("--threshold",  type=float, default=5.0,
                    help="Amplitude ratio to flag candidate (default 5.0)")
    ap.add_argument("--similarity", type=float, default=0.70,
                    help="Min fingerprint similarity (default 0.70)")
    ap.add_argument("--dry-run",    action="store_true")
    args = ap.parse_args()

    if args.input is None:
        args.input = interactive_setup()
    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")

    out_path = args.out or (os.path.splitext(args.input)[0] + "_clean.wav")
    if os.path.abspath(out_path) == os.path.abspath(args.input):
        sys.exit("Output path matches input — refusing to overwrite original.")

    print(f"\nsuno-declicker")
    print(f"  Input  : {os.path.basename(args.input)}")
    if not args.dry_run:
        print(f"  Output : {os.path.basename(out_path)}")
    print()

    data, sr = load(args.input)
    print(f"  {sr} Hz | {data.shape[1]}ch | {len(data)/sr:.1f}s\n")
    print("  Scanning for clicks...")
    clicks = detect_clicks(data, sr, args.threshold, args.similarity)

    if not clicks:
        print("  No clicks detected.")
        if not args.dry_run: save(out_path, data, sr)
    else:
        print(f"  Found {len(clicks)} click(s):\n")
        print(f"  {'#':>4}  {'Time':>8}  {'Dur':>7}  {'Match':>7}")
        print(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*7}")
        for i, (c, s, e, fsim) in enumerate(clicks):
            t = c / sr
            m, sec = divmod(t, 60)
            print(f"  {i+1:>4}  {int(m):02d}:{sec:05.2f}  {(e-s)/sr*1000:>5.2f}ms  {fsim:>7.3f}")

        if not args.dry_run:
            print("\n  Repairing per-band with STFT interpolation...")
            data = remove_clicks_spectral(data, clicks, sr)
            save(out_path, data, sr)
            print(f"\n  Original untouched : {args.input}")
            print(f"  Clean copy saved   : {out_path}")
    print()

if __name__ == "__main__":
    main()
