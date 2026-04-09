"""
suno-enhance — standalone Suno AI audio enhancement pipeline.

Steps:
  1. Declicker  — detects and removes Suno's sub-millisecond click/tick artifacts
                  using cubic spline interpolation + stem crossfade repair.
  2. Deshimmer  — removes the AI generation noise floor (broadband hiss) using
                  the instrumental stem as a reference (Suno's extractor removes
                  the noise as a byproduct, giving us an exact noise fingerprint).
                  Adds back a faint fill reverb shaped to exactly what was removed.

Everything is in this one file. No other scripts needed.

Usage:
    python suno_enhance.py                              # interactive
    python suno_enhance.py song.mp3 --vocal v.mp3 --instrumental i.mp3
    python suno_enhance.py song.mp3 --instrumental i.mp3   # deshimmer only
    python suno_enhance.py song.mp3                        # declicker only
"""

import sys, os, argparse, tempfile, subprocess, threading
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.fft import rfft
from scipy.interpolate import CubicSpline
from scipy.signal import (butter, filtfilt, stft, istft,
                           fftconvolve, sosfiltfilt)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED I/O
# ══════════════════════════════════════════════════════════════════════════════

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
    if data.ndim == 1: data = data[:, np.newaxis]
    return data, sr

def save(path, data, sr):
    sf.write(path, data, sr, subtype="PCM_24")


# ══════════════════════════════════════════════════════════════════════════════
#  DECLICKER
# ══════════════════════════════════════════════════════════════════════════════

# Spectral fingerprint of Suno v5.5 artifact clicks (256-pt Hanning, 44.1kHz).
# Cosine similarity > 0.70 = artifact. Musical transients score < 0.50.
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


def _spectral_mag(data, centre, win=FINGERPRINT_WIN):
    n = len(data)
    s = max(0, centre - win // 2)
    e = min(n, s + win)
    seg = data[s:e].mean(axis=1) if data.ndim > 1 else data[s:e]
    if len(seg) < win: return None
    mag = np.abs(rfft(seg * np.hanning(win))).astype(np.float32)
    norm = mag.max()
    return mag / (norm + 1e-10) if norm > 0 else None

def _cosine_sim(a, b):
    b = b[:len(a)]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def _fingerprint_sim(data, centre):
    mag = _spectral_mag(data, centre)
    if mag is None: return 0.0
    return _cosine_sim(mag, SUNO_CLICK_FINGERPRINT)

def _local_novelty(data, sr, centre):
    mag = _spectral_mag(data, centre)
    if mag is None: return 0.0
    offsets   = [int(o * sr) for o in [0.15, 0.30, -0.15, -0.30]]
    neighbors = [m for o in offsets if (m := _spectral_mag(data, centre + o)) is not None]
    if not neighbors: return 0.0
    return 1.0 - _cosine_sim(mag, np.mean(neighbors, axis=0))

def detect_clicks(data, sr, ratio_threshold=5.0, similarity_threshold=0.70):
    mono = np.abs(data.mean(axis=1))
    ctx  = uniform_filter1d(mono, size=int(0.4 * sr))
    ctx  = np.maximum(ctx, 1e-6)
    ratio = mono / ctx

    raw = np.where(ratio > ratio_threshold)[0]
    if len(raw) == 0: return []

    groups, group = [], [raw[0]]
    for p in raw[1:]:
        if p - group[-1] <= 50: group.append(p)
        else: groups.append(group); group = [p]
    groups.append(group)

    body_end   = max(0, len(mono) - int(2 * sr))
    candidates = []
    for g in groups:
        centre = g[np.argmax(mono[g])]
        if centre > body_end: continue
        fsim = _fingerprint_sim(data, centre)
        if fsim < similarity_threshold: continue
        thr   = ctx[centre] * 2
        start = centre
        while start > 0 and mono[start] > thr: start -= 1
        end = centre
        while end < len(mono) - 1 and mono[end] > thr: end += 1
        if (end - start) > 15: continue
        novelty = _local_novelty(data, sr, centre)
        candidates.append((centre, start, end, fsim, novelty))

    return candidates

def remove_clicks(data, clicks, sr):
    """Cubic spline repair — only 4-10 samples replaced per click."""
    out, n, ctx_len = data.copy(), len(data), 6
    for centre, start, end, fsim, novelty in clicks:
        if end - start < 1: continue
        pre_s  = max(0, start - ctx_len)
        post_e = min(n - 1, end + ctx_len)
        x_anc  = np.concatenate([np.arange(pre_s, start + 1),
                                  np.arange(end,   post_e + 1)])
        x_fill = np.arange(start, end + 1)
        for ch in range(out.shape[1]):
            y_anc = np.concatenate([out[pre_s:start+1, ch],
                                     out[end:post_e+1,  ch]])
            if len(x_anc) < 4: continue
            out[start:end+1, ch] = CubicSpline(x_anc, y_anc)(x_fill)
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
    Low band (<2kHz): cubic spline.
    High band (>2kHz): stem crossfade with 20-sample linear fade.
    """
    n        = len(data)
    stem_mix = vocal[:n] + instrumental[:n]
    if stem_mix.shape[1] < data.shape[1]:
        stem_mix = np.tile(stem_mix, (1, data.shape[1]))

    out_lo = np.zeros_like(data)
    out_hi = np.zeros_like(data)
    for ch in range(data.shape[1]):
        out_lo[:, ch] = _lp(data[:, ch], crossover_hz, sr)
        out_hi[:, ch] = _hp(data[:, ch], crossover_hz, sr)

    stem_hi = np.zeros_like(data)
    for ch in range(data.shape[1]):
        hp = _hp(stem_mix[:, ch], crossover_hz, sr)
        stem_hi[:len(hp), ch] = hp[:len(stem_hi)]

    ctx_len = 6
    for centre, start, end, fsim, novelty in clicks:
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
                out_lo[start:end+1, ch] = CubicSpline(x_anc, y_anc)(x_fill)

        s = max(0, start - fade_len)
        e = min(n - 1, end + fade_len)
        length = e - s
        env = np.ones(length)
        env[:fade_len]  = np.linspace(0, 1, fade_len)
        env[-fade_len:] = np.linspace(1, 0, fade_len)
        for ch in range(data.shape[1]):
            out_hi[s:e, ch] = (out_hi[s:e, ch] * (1 - env)
                                + stem_hi[s:e, ch] * env)

    return out_lo + out_hi


# ══════════════════════════════════════════════════════════════════════════════
#  DESHIMMER
# ══════════════════════════════════════════════════════════════════════════════

def _noise_fingerprint_from_stem(mix, instr, sr, ref_seconds=4.0,
                                  n_fft=2048, hop=512):
    """
    Noise fingerprint = mix - instrumental over the first ref_seconds
    (pre-vocal region where mix ≈ instrumental + noise).
    Suno's stem extractor removes the AI noise floor as a byproduct.
    """
    n_ref = min(int(ref_seconds * sr), len(mix), len(instr))
    mx    = mix[:n_ref].mean(axis=1)
    ix    = instr[:n_ref].mean(axis=1)

    corr   = np.correlate(mx, ix, mode='full')
    offset = int(np.argmax(np.abs(corr)) - (len(ix) - 1))
    offset = int(np.clip(offset, -sr // 4, sr // 4))

    def _align(sig):
        if offset > 0: return np.pad(sig, (offset, 0))[:n_ref]
        if offset < 0:
            sig = sig[-offset:]
            return np.pad(sig, (0, n_ref - len(sig)))
        return sig

    ix = _align(ix)
    fingerprint = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for ch in range(mix.shape[1]):
        m    = mix[:n_ref, ch]
        i_ch = instr[:n_ref, ch] if ch < instr.shape[1] else instr[:n_ref, 0]
        i_ch = _align(i_ch)
        _, _, N = stft(m - i_ch, fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, window='hann', boundary='even')
        fingerprint += np.abs(N).mean(axis=1)

    fingerprint /= mix.shape[1]
    print(f"  Peak noise bin : {np.argmax(fingerprint) * sr / n_fft:.0f} Hz")
    print(f"  Alignment offset: {offset} samples ({offset/sr*1000:.1f} ms)")
    return fingerprint.astype(np.float32)

def _estimate_noise_floor(mag, window_frames=50):
    """Per-bin rolling minimum — fallback when no instrumental stem is available."""
    noise = np.empty_like(mag)
    half  = window_frames // 2
    for t in range(mag.shape[1]):
        lo = max(0, t - half)
        hi = min(mag.shape[1], t + half + 1)
        noise[:, t] = mag[:, lo:hi].min(axis=1)
    return noise

def deshimmer(song, sr, fingerprint=None, strength=0.106, floor=0.90,
              band_lo=500, n_fft=2048, hop=512, noise_window=50):
    """
    Subtract the noise fingerprint from the song above band_lo Hz.
    floor ensures no frequency is reduced more than (1-floor)*100%.
    """
    n     = len(song)
    out   = np.zeros_like(song)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    freq_mask = (freqs >= band_lo).astype(np.float32)
    lo_idx    = np.searchsorted(freqs, band_lo)
    for i in range(10):
        idx = lo_idx - 10 + i
        if 0 <= idx < len(freq_mask):
            freq_mask[idx] = (i + 1) / 11

    for ch in range(song.shape[1]):
        _, _, S = stft(song[:, ch], fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, window='hann', boundary='even')
        mag   = np.abs(S).astype(np.float32)
        phase = np.angle(S)

        if fingerprint is not None:
            sub = strength * fingerprint[:, None] * freq_mask[:, None]
        else:
            sub = strength * _estimate_noise_floor(mag, noise_window) * freq_mask[:, None]

        mag_clean = np.maximum(mag - sub, mag * floor)
        _, rep    = istft(mag_clean * np.exp(1j * phase), fs=sr, nperseg=n_fft,
                          noverlap=n_fft - hop, window='hann', boundary='even')
        rep = rep[:n] if len(rep) >= n else np.pad(rep, (0, n - len(rep)))
        out[:, ch] = rep.astype(np.float32)

    return out

def _make_fill_ir(sr, rt60=0.6, pre_delay_ms=12.0):
    """
    Bright, airy fill IR (0.6s, 800Hz–14kHz, frequency-domain smoothed).
    Used only for reverbing what was removed — nothing else.
    """
    ir_len = int(rt60 * sr * 1.3)
    t      = np.arange(ir_len) / sr
    rng    = np.random.default_rng(42)
    ir     = rng.standard_normal(ir_len) * np.exp(-6.908 * t / rt60)

    build = int(0.004 * sr)
    ir[:build] *= np.linspace(0, 1, build)

    IR  = np.fft.rfft(ir)
    mag = uniform_filter1d(np.abs(IR), size=60)
    mag = uniform_filter1d(mag, size=60)
    ir  = np.fft.irfft(mag * np.exp(1j * np.angle(IR)), n=ir_len)
    ir  = sosfiltfilt(butter(3, 14000/(sr/2), btype='low',  output='sos'), ir)
    ir  = sosfiltfilt(butter(3,   800/(sr/2), btype='high', output='sos'), ir)
    ir  = np.pad(ir, (int(pre_delay_ms / 1000 * sr), 0))
    ir /= np.abs(ir).max() + 1e-10
    return ir.astype(np.float32)

def fill_removed(original, cleaned, sr, wet=0.012):
    """
    Convolve only what was removed (original - cleaned) with a bright fill IR
    and mix back at wet level. Antispace drops are preserved automatically:
    removed ≈ 0 at silence → fill reverb ≈ 0.
    """
    removed = original - cleaned
    ir      = _make_fill_ir(sr)
    out     = cleaned.copy()
    for ch in range(cleaned.shape[1]):
        rev = fftconvolve(removed[:, ch], ir)[:len(cleaned)]
        out[:, ch] += wet * rev.astype(np.float32)
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  A/B COMPARATOR  (inline — no external script needed)
# ══════════════════════════════════════════════════════════════════════════════

def ab_compare(path_a, path_b):
    try:
        import sounddevice as sd
    except ImportError:
        print("  (sounddevice not installed — skipping A/B comparison)")
        return

    a, sr_a = load(path_a)
    b, sr_b = load(path_b)
    if sr_a != sr_b:
        print("  Sample rates differ — cannot compare.")
        return

    sr = sr_a
    n  = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ch = max(a.shape[1], b.shape[1])
    if a.shape[1] < ch: a = np.tile(a, (1, ch))[:, :ch]
    if b.shape[1] < ch: b = np.tile(b, (1, ch))[:, :ch]

    dip_len = int(0.03 * sr)
    state   = {'pos': 0, 'src': 'A', 'dip': 0, 'done': False}

    def fmt(t):
        m, s = divmod(t, 60)
        return f"{int(m):02d}:{s:05.2f}"

    def status():
        t = state['pos'] / sr
        lbl = os.path.basename(path_a if state['src'] == 'A' else path_b)
        print(f"\r  [{fmt(t)}]  ▶ {state['src']}: {lbl}          ", end='', flush=True)

    def callback(outdata, frames, time_info, status_):
        pos  = state['pos']
        end  = min(pos + frames, n)
        size = end - pos
        if size <= 0:
            outdata[:] = 0; state['done'] = True; raise sd.CallbackStop
        src   = a if state['src'] == 'A' else b
        chunk = src[pos:end].copy()
        dip   = state['dip']
        if dip > 0:
            d = min(dip, size)
            chunk[:d] *= 0
            state['dip'] = dip - d
        outdata[:size] = chunk
        if size < frames:
            outdata[size:] = 0; state['done'] = True
        state['pos'] = end

    def keys():
        import tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not state['done']:
                k = sys.stdin.read(1)
                if k == ' ':
                    state['src'] = 'B' if state['src'] == 'A' else 'A'
                    state['dip'] = dip_len; status()
                elif k in '123456789':
                    state['pos'] = int(int(k) / 10 * n); state['dip'] = 0; status()
                elif k in ('q', 'Q', '\x03'):
                    state['done'] = True; break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print(f"\n  A — {os.path.basename(path_a)}")
    print(f"  B — {os.path.basename(path_b)}")
    print(f"  {sr} Hz | {ch}ch | {n/sr:.1f}s")
    print(f"\n  SPACE=switch  1–9=seek  Q=quit\n")
    status()

    t = threading.Thread(target=keys, daemon=True)
    t.start()
    with sd.OutputStream(samplerate=sr, channels=ch, callback=callback):
        while not state['done']:
            sd.sleep(50)
    print(f"\n\n  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE SETUP + CLI
# ══════════════════════════════════════════════════════════════════════════════

def _list_audio(directory="."):
    exts = {".mp3", ".wav", ".flac", ".aif", ".aiff"}
    return sorted(f for f in os.listdir(directory)
                  if os.path.splitext(f)[1].lower() in exts
                  and os.path.isfile(os.path.join(directory, f)))

def _pick(prompt, files, exclude=None):
    exclude   = exclude or []
    available = [(i, f) for i, f in enumerate(files, 1) if f not in exclude]
    if not available: sys.exit("No audio files available.")
    print(f"\n  {prompt}")
    for i, f in available:
        print(f"    {i:>3}.  {f}")
    while True:
        try:
            n = int(input("\n  Enter number: ").strip())
            m = [f for i, f in available if i == n]
            if m: return m[0]
        except (ValueError, EOFError):
            pass
        print("  Invalid — try again.")

def _interactive():
    files = _list_audio()
    if not files:
        sys.exit("No audio files found in the current directory.")

    print("\n  ┌──────────────────────────────────────────────┐")
    print("  │           suno-enhance  pipeline             │")
    print("  │  declicker → deshimmer → fill reverb         │")
    print("  └──────────────────────────────────────────────┘")
    print("\n  Place your Suno files in this folder.")
    print("  The original is NEVER modified — a clean copy is saved.\n")

    original     = _pick("Select the FULL SONG MIX:", files)
    vocal        = _pick("Select the VOCAL stem (for click repair):",
                         files, exclude=[original])
    instrumental = _pick("Select the INSTRUMENTAL stem (for noise fingerprint):",
                         files, exclude=[original, vocal])

    print(f"\n  Song         : {original}")
    print(f"  Vocal stem   : {vocal}")
    print(f"  Instrumental : {instrumental}")
    input("\n  Press Enter to start, or Ctrl+C to cancel. ")
    return original, vocal, instrumental


def main():
    ap = argparse.ArgumentParser(description="Suno AI audio enhancement pipeline.")
    ap.add_argument("input",            nargs="?",  default=None)
    ap.add_argument("--vocal",          default=None)
    ap.add_argument("--instrumental",   default=None)
    ap.add_argument("--out",            default=None)
    ap.add_argument("--threshold",      type=float, default=5.0)
    ap.add_argument("--similarity",     type=float, default=0.70)
    ap.add_argument("--shimmer-strength", type=float, default=0.106,
                    help="Deshimmer subtraction strength (default 0.106)")
    ap.add_argument("--shimmer-floor",  type=float, default=0.90)
    ap.add_argument("--fill-wet",       type=float, default=0.012,
                    help="Fill-reverb level for removed content (default 0.012)")
    ap.add_argument("--no-compare",     action="store_true")
    args = ap.parse_args()

    if not args.input:
        args.input, args.vocal, args.instrumental = _interactive()

    for p in filter(None, [args.input, args.vocal, args.instrumental]):
        if not os.path.isfile(p):
            sys.exit(f"File not found: {p}")

    out_path = args.out or (os.path.splitext(args.input)[0] + "_enhanced.wav")
    if os.path.abspath(out_path) == os.path.abspath(args.input):
        sys.exit("Output path matches input — refusing to overwrite.")

    print(f"\n{'━'*54}")
    print(f"  suno-enhance")
    print(f"{'━'*54}")
    print(f"  Input        : {os.path.basename(args.input)}")
    if args.vocal:        print(f"  Vocal stem   : {os.path.basename(args.vocal)}")
    if args.instrumental: print(f"  Instrumental : {os.path.basename(args.instrumental)}")
    print(f"  Output       : {os.path.basename(out_path)}")
    print(f"{'━'*54}\n")

    data, sr = load(args.input)
    print(f"  {sr} Hz | {data.shape[1]}ch | {len(data)/sr:.1f}s\n")

    # ── step 1: declicker ────────────────────────────────────────────────────
    print("  [1/2] Declicker")
    clicks = detect_clicks(data, sr, args.threshold, args.similarity)
    if not clicks:
        print("        No Suno artifact clicks detected.\n")
    else:
        print(f"        Found {len(clicks)} click(s):\n")
        print(f"  {'#':>4}  {'Time':>8}  {'Dur':>7}  {'Match':>7}")
        print(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*7}")
        for i, (c, s, e, fsim, nov) in enumerate(clicks):
            t = c / sr; m, sec = divmod(t, 60)
            print(f"  {i+1:>4}  {int(m):02d}:{sec:05.2f}  {(e-s)/sr*1000:>5.2f}ms  {fsim:>7.3f}")
        if args.vocal and args.instrumental:
            print("\n        Repairing with stems...")
            vocal, _ = load(args.vocal)
            instr, _ = load(args.instrumental)
            data = remove_clicks_with_stems(data, clicks, vocal, instr, sr)
        else:
            print("\n        Repairing with cubic spline...")
            data = remove_clicks(data, clicks, sr)
        print("        Done.\n")

    # ── step 2: deshimmer ────────────────────────────────────────────────────
    print("  [2/2] Deshimmer")
    fingerprint = None
    if args.instrumental:
        if 'instr' not in dir():
            instr, _ = load(args.instrumental)
        print(f"        Extracting noise fingerprint from instrumental stem...")
        fingerprint = _noise_fingerprint_from_stem(data, instr, sr)
    else:
        print("        No instrumental stem — using minimum-statistics fallback.")

    original_for_fill = data.copy()
    data = deshimmer(data, sr, fingerprint=fingerprint,
                     strength=args.shimmer_strength, floor=args.shimmer_floor)

    if args.fill_wet > 0:
        print(f"        Fill reverb (wet={args.fill_wet})...")
        data = fill_removed(original_for_fill, data, sr, wet=args.fill_wet)
    print("        Done.\n")

    save(out_path, data, sr)
    print(f"  Original untouched : {args.input}")
    print(f"  Enhanced copy saved: {out_path}")
    print(f"{'━'*54}\n")

    if not args.no_compare:
        ans = input("  Play A/B comparison now? (y/N): ").strip().lower()
        if ans == 'y':
            ab_compare(args.input, out_path)


if __name__ == "__main__":
    main()
