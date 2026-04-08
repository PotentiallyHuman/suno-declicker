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
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.fft import rfft
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfilt, stft, istft, correlate

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


# ── stem warp alignment ───────────────────────────────────────────────────────

def build_stem_warp(mix_mono, dry_mono, sr):
    """
    Use WhisperX phoneme timestamps from the dry stem as anchor points,
    then cross-correlate each anchor in the mix to get precise mix positions.
    Returns a (mix_times, stem_times) pair for use with np.interp.
    Suno stems drift up to 500ms over a 4-min song.
    """
    try:
        import whisperx
    except ImportError:
        print("  [warp] whisperx not installed — stem alignment skipped")
        return None, None

    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, dry_mono, sr)
    try:
        device = "cpu"
        model  = whisperx.load_model("base", device, compute_type="float32")
        audio  = whisperx.load_audio(tmp)
        result = model.transcribe(audio, batch_size=4, language="en")
        ma, meta = whisperx.load_align_model(language_code="en", device=device)
        result   = whisperx.align(result["segments"], ma, meta, audio, device)
        words    = [w for w in result["word_segments"]
                    if "start" in w and "end" in w and w["end"] > w["start"]]
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

    n      = min(len(mix_mono), len(dry_mono))
    search = int(0.5 * sr)
    pairs  = []

    for w in words:
        stem_mid = (w["start"] + w["end"]) / 2
        seg_len  = max(int((w["end"] - w["start"]) * sr), int(0.08 * sr))
        s_dry    = int(w["start"] * sr)
        e_dry    = min(n, s_dry + seg_len)
        seg_dry  = dry_mono[s_dry:e_dry]
        if len(seg_dry) < 256: continue

        exp_mix  = int(stem_mid * sr)
        s_mix    = max(0, exp_mix - search - seg_len)
        e_mix    = min(n, exp_mix + search + seg_len)
        seg_mix  = mix_mono[s_mix:e_mix]
        if len(seg_mix) < len(seg_dry): continue

        cc       = correlate(seg_mix, seg_dry, mode='valid')
        offset   = cc.argmax()
        mix_mid  = (s_mix + offset + len(seg_dry) // 2) / sr
        pairs.append((stem_mid, mix_mid))

    if len(pairs) < 10:
        return None, None

    pairs      = np.array(pairs)
    stem_t     = pairs[:, 0]
    mix_t      = pairs[:, 1]
    drift      = (stem_t - mix_t) * 1000

    # reject outliers vs sliding median
    idx        = np.argsort(stem_t)
    stem_t     = stem_t[idx]; mix_t = mix_t[idx]; drift = drift[idx]
    med        = median_filter(drift, size=15)
    good       = np.abs(drift - med) < 80
    stem_clean = stem_t[good]
    mix_clean  = mix_t[good]

    print(f"  [warp] {good.sum()} anchors, drift {drift[good].min():.0f}ms→{drift[good].max():.0f}ms")
    return mix_clean, stem_clean


def warp_stem_time(mix_time_s, mix_anchors, stem_anchors):
    """Given a mix timestamp (seconds), return the corresponding stem timestamp."""
    if mix_anchors is None:
        return mix_time_s
    return float(np.interp(mix_time_s, mix_anchors, stem_anchors))


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

def merge_click_regions(clicks, sr, gap=0.08):
    """
    Merge clicks within `gap` seconds of each other into one region.
    Returns list of (start, end, blend_strength) tuples.
    blend_strength scales from 1.0 (short/isolated) down to 0.25 (long cluster)
    so long clusters keep most of the original signal underneath.
    """
    if not clicks: return []
    regions = []
    gs, ge = clicks[0][1], clicks[0][2]
    for _, start, end, _ in clicks[1:]:
        if start - ge < int(gap * sr):
            ge = max(ge, end)
        else:
            regions.append((gs, ge))
            gs, ge = start, end
    regions.append((gs, ge))

    result = []
    for s, e in regions:
        dur = (e - s) / sr
        # full repair for short windows; taper to 0.25 for windows >0.5s
        strength = max(0.25, 1.0 - (dur - 0.01) / 0.5)
        result.append((s, e, strength))
    return result

def repair_band(band_data, clicks, sr, n_fft=2048, hop=512, ctx_frames=6):
    out = band_data.copy()
    n   = len(band_data)

    # merge nearby clicks so dense clusters are repaired as one region
    regions = merge_click_regions(clicks, sr)

    for ch in range(band_data.shape[1]):
        sig = band_data[:, ch]
        _, _, Z = stft(sig, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                       window='hann', boundary='even')
        mag      = np.abs(Z)
        phase    = np.angle(Z)
        orig_mag = mag.copy()
        n_frames = Z.shape[1]

        for start, end, strength in regions:
            f0 = max(ctx_frames, start // hop)
            f1 = min(n_frames - 1 - ctx_frames, end // hop + 1)
            if f0 > f1:
                f0 = f1 = (start + end) // 2 // hop

            pre_mag  = mag[:, max(0, f0-ctx_frames):f0].mean(axis=1)
            post_mag = mag[:, f1+1:min(n_frames, f1+1+ctx_frames)].mean(axis=1)
            n_repair = f1 - f0 + 1

            for fi, frame in enumerate(range(f0, f1 + 1)):
                t = (fi + 1) / (n_repair + 1)
                interp = pre_mag * (1-t) + post_mag * t
                # blend: full interp for isolated, mostly original for clusters
                mag[:, frame]   = interp * strength + orig_mag[:, frame] * (1 - strength)
                phase[:, frame] = (phase[:, max(0,f0-1)] * (1-t) +
                                   phase[:, min(n_frames-1,f1+1)] * t)

        Z_fixed = mag * np.exp(1j * phase)
        _, rep  = istft(Z_fixed, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                        window='hann', boundary='even')
        rep = rep[:n] if len(rep) >= n else np.pad(rep, (0, n-len(rep)))
        out[:, ch] = rep

    return out


def remove_clicks_spectral(data, clicks, sr, dry=None, mix_anchors=None, stem_anchors=None):
    """
    Split into 6 perceptual bands → repair each independently → recombine.
    Blend repaired back into original via a cosine-faded click mask.

    If dry vocal stem + warp anchors provided: fill click windows with the
    warp-aligned stem (sample-accurate sync). Falls back to STFT interpolation.
    """
    if not clicks: return data
    n    = len(data)
    fade = int(0.005 * sr)  # 5ms crossfade each side

    if dry is not None and mix_anchors is not None:
        # ── warp-aligned stem-fill mode ─────────────────────────────────────
        dry_n = len(dry)
        out   = data.copy()

        for _, start, end, _ in clicks:
            mix_mid_s  = (start + end) / 2 / sr
            stem_mid_s = warp_stem_time(mix_mid_s, mix_anchors, stem_anchors)
            half       = (end - start) // 2 + fade

            # extract aligned stem window
            stem_s = int(stem_mid_s * sr) - half
            stem_e = stem_s + 2 * half
            if stem_s < 0 or stem_e > dry_n: continue
            stem_win = dry[stem_s:stem_e]

            # scale to local mix amplitude
            ctx_s   = max(0, start - int(0.02 * sr))
            ctx_e   = min(n, end   + int(0.02 * sr))
            mix_rms = np.sqrt(np.mean(data[ctx_s:ctx_e] ** 2) + 1e-10)
            dry_rms = np.sqrt(np.mean(stem_win ** 2) + 1e-10)
            scale   = mix_rms / dry_rms

            # write into output with cosine fade
            s = max(0, start - fade)
            e = min(n, end   + fade)
            w = np.ones(e - s, dtype=np.float32)
            fi = min(fade, start - s)
            if fi > 0: w[:fi] = (1 - np.cos(np.pi * np.arange(fi) / fi)) / 2
            fo = min(fade, e - end)
            if fo > 0: w[e - s - fo:] = (1 + np.cos(np.pi * np.arange(fo) / fo)) / 2

            win_len   = e - s
            stem_trim = stem_win[:win_len]
            if stem_trim.shape[0] < win_len: continue
            out[s:e]  = out[s:e] * (1 - w[:, None]) + stem_trim * scale * w[:, None]

        return out

    else:
        # ── STFT interpolation mode (no stems) ──────────────────────────────
        bands    = split_bands(data, sr)
        repaired = [repair_band(b, clicks, sr) for b in bands]
        combined = merge_bands(repaired)

        # mask over merged regions (not individual clicks)
        regions = merge_click_regions(clicks, sr)
        mask = np.zeros(n, dtype=np.float32)
        for start, end, strength in regions:
            s = max(0, start - fade)
            e = min(n, end   + fade)
            mask[s:e] = strength
            fi = min(fade, start - s)
            if fi > 0:
                mask[s:s+fi] = strength * (1 - np.cos(np.pi * np.arange(fi) / fi)) / 2
            fo = min(fade, e - end)
            if fo > 0:
                mask[end:end+fo] = strength * (1 + np.cos(np.pi * np.arange(fo) / fo)) / 2

        mask = mask[:, None]
        return data * (1 - mask) + combined * mask


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
    ap.add_argument("--dry",        default=None,
                    help="Dry vocal stem — fills click windows with natural vocal texture")
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
            dry = None
            mix_anchors = stem_anchors = None
            if args.dry:
                if not os.path.isfile(args.dry):
                    sys.exit(f"Dry stem not found: {args.dry}")
                dry, _ = load(args.dry)
                print("\n  Aligning stem to mix with WhisperX...")
                mix_mono = data.mean(axis=1)
                dry_mono = dry.mean(axis=1)
                mix_anchors, stem_anchors = build_stem_warp(mix_mono, dry_mono, sr)
                if mix_anchors is not None:
                    print("  Repairing with warp-aligned dry vocal stem...")
                else:
                    print("  Warp failed — falling back to STFT interpolation...")
                    dry = None
            else:
                print("\n  Repairing per-band with STFT interpolation...")
            data = remove_clicks_spectral(data, clicks, sr, dry=dry,
                                          mix_anchors=mix_anchors, stem_anchors=stem_anchors)
            save(out_path, data, sr)
            print(f"\n  Original untouched : {args.input}")
            print(f"  Clean copy saved   : {out_path}")
    print()

if __name__ == "__main__":
    main()
