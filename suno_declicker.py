"""
suno-enhance — standalone Suno AI audio enhancement pipeline.

Steps:
  1. Declicker × 3 — detects and removes Suno's sub-millisecond click/tick
                      artifacts using cubic spline interpolation + stem
                      crossfade repair.  Three passes catch residual artifacts
                      that the prior pass exposed.

Everything is in this one file. No other scripts needed.

Usage:
    python suno_enhance.py                              # interactive
    python suno_enhance.py song.mp3 --vocal v.mp3 --instrumental i.mp3
    python suno_enhance.py song.mp3                    # spline-only declicker
"""

import sys, os, argparse, tempfile, subprocess, threading
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.fft import rfft
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, stft, istft, correlate


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED I/O
# ══════════════════════════════════════════════════════════════════════════════

def load(path, sr_target=44100):
    if path.lower().endswith(".mp3"):
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run(["ffmpeg","-y","-i",path,"-ar",str(sr_target),tmp],
                           capture_output=True, stdin=subprocess.DEVNULL, check=True)
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

def _flatness_candidates(mono, sr, flatness_threshold=0.60, hop=64, nperseg=256):
    """
    Vectorised spectral-flatness scan.

    Spectral flatness = geometric_mean(|X[f]|) / arithmetic_mean(|X[f]|).
    White noise → 1.0.  Broadband click → 0.7–0.9.  Music/speech → 0.05–0.3.

    Returns a set of sample indices (frame centres) where flatness exceeds
    the threshold AND there is a local amplitude spike (> 2× the 10 ms RMS
    context) — this dual condition keeps false positives low on fricatives
    and breath noise which are spectrally flat but not impulsive.
    """
    from scipy.signal import stft as _stft
    _, _, S = _stft(mono, fs=sr, nperseg=nperseg,
                    noverlap=nperseg - hop, boundary='even')
    mag      = np.abs(S[1:])                        # drop DC
    log_geom = np.mean(np.log(mag + 1e-12), axis=0)
    arith    = np.mean(mag, axis=0)
    flatness = np.exp(log_geom) / (arith + 1e-12)

    flat_frames  = np.where(flatness > flatness_threshold)[0]
    flat_samples = (flat_frames * hop + nperseg // 2).astype(np.intp)
    flat_samples = flat_samples[flat_samples < len(mono)]

    # Amplitude gate: must be a genuine impulse (4× the 10 ms context),
    # not just a noisy-but-quiet frame.  This rules out breath noise,
    # fricatives, and the STFT halo around already-detected clicks.
    ctx_10ms  = uniform_filter1d(mono, size=max(1, int(0.010 * sr)))
    ratio_10  = mono / (ctx_10ms + 1e-6)
    spike_set = set(np.where(ratio_10 > 4.0)[0].tolist())

    return {int(s) for s in flat_samples} & spike_set


def detect_clicks(data, sr, ratio_threshold=5.0, similarity_threshold=0.70,
                  dry_vocal=None):
    mono = np.abs(data.mean(axis=1))
    ctx  = uniform_filter1d(mono, size=int(0.4 * sr))
    ctx  = np.maximum(ctx, 1e-6)
    ratio = mono / ctx

    # Short-context secondary detector: 30ms window, 4.5× threshold.
    # Catches clicks buried in audio where the 400ms context is inflated
    # but a tighter 30ms local baseline reveals the transient more clearly.
    ctx_short  = uniform_filter1d(mono, size=max(1, int(0.030 * sr)))
    ctx_short  = np.maximum(ctx_short, 1e-6)
    ratio_short = mono / ctx_short

    raw_long  = set(np.where(ratio       > ratio_threshold)[0].tolist())
    raw_short = set(np.where(ratio_short > 4.5)[0].tolist())
    # Flatness scan: catches clicks buried under loud sections where the
    # amplitude ratio is diluted but the broadband spectral shape is distinct.
    raw_flat  = _flatness_candidates(mono, sr, flatness_threshold=0.75)
    raw_all   = np.array(sorted(raw_long | raw_short | raw_flat), dtype=np.intp)

    if len(raw_all) == 0: return []

    groups, group = [], [raw_all[0]]
    for p in raw_all[1:]:
        if p - group[-1] <= 50: group.append(p)
        else: groups.append(group); group = [p]
    groups.append(group)

    dry_mono = (np.abs(dry_vocal.mean(axis=1))
                if dry_vocal is not None else None)

    body_end   = max(0, len(mono) - int(2 * sr))
    candidates = []
    for g in groups:
        centre = g[np.argmax(mono[g])]
        if centre > body_end: continue

        fsim = _fingerprint_sim(data, centre)

        thr   = ctx[centre] * 2
        start = centre
        while start > 0 and mono[start] > thr: start -= 1
        end = centre
        while end < len(mono) - 1 and mono[end] > thr: end += 1
        dur = end - start

        if dur > 150:
            continue

        # Check dry vocal presence at this exact window
        dry_confirmed = False
        if dry_mono is not None:
            s0 = max(0, start - 5)
            s1 = min(len(dry_mono) - 1, end + 5)
            mix_peak = mono[s0:s1].max()
            dry_peak = dry_mono[s0:s1].max()
            dry_confirmed = (dry_peak / (mix_peak + 1e-10)) < 0.10

        if dry_confirmed and fsim >= 0.40:
            pass  # stem confirms this is absent from dry vocal — accept any duration
        elif dur > 15:
            continue  # long spike, no stem confirmation — skip
        elif fsim < similarity_threshold:
            continue  # short spike, weak fingerprint match — skip

        novelty = _local_novelty(data, sr, centre)
        candidates.append((centre, start, end, fsim, novelty))

    return candidates

def _attenuate_click(out, start, end, n, floor=0.05, fade=8):
    """Smooth gain-dip envelope over [start, end] — for long artifacts."""
    s = max(0, start - fade)
    e = min(n - 1, end + fade)
    length = e - s
    # build envelope: 1 → floor → 1
    env = np.ones(length, dtype=np.float32) * floor
    env[:fade]  = np.linspace(1.0, floor, fade)
    env[-fade:] = np.linspace(floor, 1.0, fade)
    for ch in range(out.shape[1]):
        out[s:e, ch] *= env

def remove_clicks(data, clicks, sr):
    """Short clicks: cubic spline. Long clicks (>15 samples): gain attenuation."""
    out, n, ctx_len = data.copy(), len(data), 6
    for centre, start, end, fsim, novelty in clicks:
        if end - start < 1: continue
        if end - start > 15:
            _attenuate_click(out, start, end, n)
            continue
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
        dur = end - start
        if dur > 15:
            # Long artifact: attenuate on the full mix rather than stem crossfade
            _attenuate_click(out_lo, start, end, n)
            _attenuate_click(out_hi, start, end, n)
            continue

        if dur >= 1:
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


def repair_clicks_ar(data, clicks, sr, order=256, ctx=512, tail_pad=12):
    """
    Click repair via autoregressive (AR) / linear-predictive prediction.

    For each click:
      1. Fit an AR(order) model to `ctx` clean samples before the gap
         (forward model — captures the spectral structure arriving at the click).
      2. Fit an AR(order) model to `ctx` clean samples after the gap, reversed
         (backward model — captures the structure departing from the click).
      3. Predict `gap` samples forward from the left context and `gap` samples
         backward from the right context.
      4. Blend: forward weight 1→0, backward weight 0→1 across the gap.

    This is how iZotope RX's Click Repair works.  The AR model captures
    harmonics, formants, and periodicity of whatever mix is present at that
    moment — vowels, chords, consonant onsets — without needing any external
    reference.  The blend from two directions prevents the forward prediction
    from drifting away from the true post-click audio.

    order : AR model order.  256 ≈ 5.8 ms at 44.1 kHz — enough to capture
            one full period of a 170 Hz bass note or a 200 Hz vocal fundamental.
    ctx   : clean samples used to fit the model on each side.
    """
    from scipy.linalg import solve_toeplitz

    def _ar_coeffs(sig, p):
        n = len(sig)
        r = np.array([np.dot(sig[:n - k], sig[k:]) / (n - k)
                      for k in range(p + 1)])
        try:
            return solve_toeplitz(r[:p], r[1:p + 1])
        except Exception:
            return np.zeros(p)

    def _predict(buf, coeffs, steps):
        p   = len(coeffs)
        buf = list(buf[-p:])
        out = []
        for _ in range(steps):
            v = float(np.dot(coeffs, buf[-p:][::-1]))
            out.append(v)
            buf.append(v)
        return np.array(out)

    if not clicks:
        return data.copy()

    result = data.copy()
    n      = len(data)

    for centre, start, end, fsim, novelty in clicks:
        s   = max(0,     start - tail_pad)
        e   = min(n - 1, end   + tail_pad)
        gap = e - s + 1

        for ch in range(result.shape[1]):
            pre_s  = max(0, s - ctx)
            post_e = min(n, e + 1 + ctx)
            pre    = result[pre_s:s,      ch].astype(np.float64)
            post   = result[e + 1:post_e, ch].astype(np.float64)

            p = min(order, len(pre) - 5, len(post) - 5)
            if p < 10:
                continue

            a_fwd = _ar_coeffs(pre,       p)
            a_bwd = _ar_coeffs(post[::-1], p)

            fwd = _predict(pre,       a_fwd, gap)
            bwd = _predict(post[::-1], a_bwd, gap)[::-1]

            # Linear blend: forward→backward across the gap
            alpha   = np.linspace(1.0, 0.0, gap, dtype=np.float64)
            blended = alpha * fwd + (1.0 - alpha) * bwd
            # Clamp to ±2.0 to guard against AR divergence on complex signals
            blended = np.clip(blended, -2.0, 2.0)
            result[s:e + 1, ch] = blended.astype(np.float32)

    return result


def erase_and_fill_clicks(data, clicks, sr, tail_pad=12, ctx=30):
    """
    Zero the click (+ tail_pad samples each side) then fill the gap using
    cubic spline interpolation from the clean audio on both sides.

    Because both sides of the gap are now clean (no click contamination),
    the spline interpolates smoothly through the gap without the phase-
    cancellation 'gulp' that occurs when interpolating across a live click.

    tail_pad : samples to extend each side beyond the detected click bounds
               (catches low-amplitude click tails that remain visible)
    ctx      : clean anchor samples used on each side for the spline fit
    """
    if not clicks:
        return data.copy()

    out = data.copy()
    n   = len(data)

    for centre, start, end, fsim, novelty in clicks:
        s = max(0, start - tail_pad)
        e = min(n - 1, end + tail_pad)
        gap = e - s + 1
        if gap < 1:
            continue

        for ch in range(out.shape[1]):
            # Anchor points from clean audio on each side
            pre_s  = max(0, s - ctx)
            post_e = min(n - 1, e + ctx)

            x_pre  = np.arange(pre_s, s)
            x_post = np.arange(e + 1, post_e + 1)
            x_gap  = np.arange(s, e + 1)

            if len(x_pre) < 2 or len(x_post) < 2:
                # Not enough context — just zero the region
                out[s:e + 1, ch] = 0.0
                continue

            x_anc = np.concatenate([x_pre, x_post])
            y_anc = np.concatenate([out[pre_s:s, ch], out[e + 1:post_e + 1, ch]])

            out[s:e + 1, ch] = CubicSpline(x_anc, y_anc)(x_gap).astype(np.float32)

    return out


def delete_clicks(data, clicks, sr, pad=4, tail_pad=12):
    """
    Remove click samples entirely and stitch surrounding audio together.

    Each detected click region is expanded by `tail_pad` samples on each side
    to cover the click's low-amplitude tails (which are below the detection
    threshold but still visible/audible). A linear ramp of `pad` samples at
    each splice junction prevents step discontinuities.

    The output is shorter than the input by the total deleted samples
    (typically ~30 ms for a full song with tail padding).
    """
    if not clicks:
        return data.copy()

    n = len(data)

    # Build sorted, non-overlapping deletion regions — expanded by tail_pad
    raw = sorted((max(0, start - tail_pad), min(n - 1, end + tail_pad))
                 for _, start, end, *_ in clicks)
    merged = [list(raw[0])]
    for s, e in raw[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Concatenate kept segments
    segments = []
    prev = 0
    for s, e in merged:
        if s > prev:
            segments.append(data[prev:s].copy())
        prev = e + 1
    if prev < n:
        segments.append(data[prev:].copy())

    if not segments:
        return np.zeros((0, data.shape[1]), dtype=data.dtype)

    out = np.concatenate(segments, axis=0)

    # At each splice junction, linearly interpolate `pad` samples
    # from the last sample before the cut to the first sample after it.
    # This eliminates any step discontinuity in ~0.09 ms — completely inaudible.
    pos = 0
    for seg in segments[:-1]:
        pos += len(seg)
        cf = min(pad, pos, len(out) - pos)
        if cf < 2:
            continue
        for ch in range(out.shape[1]):
            v0 = float(out[pos - 1, ch])
            v1 = float(out[pos,     ch])
            ramp = np.linspace(v0, v1, cf + 2)[1:-1].astype(np.float32)
            # Blend ramp into the `cf` samples straddling the join
            half = cf // 2
            out[pos - half : pos,        ch] = ramp[:half]
            out[pos         : pos + half, ch] = ramp[half:]

    return out


def repair_clicks_spectral_gate(data, clicks, sr, n_fft=256, hop=64):
    """
    Per-bin magnitude gate in the STFT domain.

    A click inflates the STFT magnitude at EVERY frequency bin in the
    affected frame.  For each click frame, this function computes the
    expected per-bin magnitude by interpolating the two nearest clean
    frames, then applies a per-bin gain  g[f] = min(1, expected[f] / actual[f])
    that clips the inflated bins back down to the surrounding level.

    The ORIGINAL PHASE is preserved — there is no phase interpolation,
    so there is no 'gulp' from phase cancellation.  The gain only
    suppresses — it never amplifies — so legitimate transients that are
    genuinely louder than their neighbours are left untouched.

    Multiple passes converge quickly: if a click is 10× the underlying
    signal, pass 1 brings it to 1×, pass 2 to ~0.1×, etc.
    """
    if not clicks:
        return data.copy()

    out  = data.copy()
    n    = len(data)
    half = n_fft // 2

    for centre, start, end, fsim, novelty in sorted(clicks, key=lambda x: x[0]):
        ctx   = n_fft * 8
        win_s = max(0, start - ctx)
        win_e = min(n,  end   + ctx)

        for ch in range(data.shape[1]):
            seg = out[win_s:win_e, ch].astype(np.float64)
            if len(seg) < n_fft * 2:
                continue

            _, _, S = stft(seg, fs=sr, nperseg=n_fft,
                           noverlap=n_fft - hop, window='hann', boundary='even')

            rel_s    = start - win_s
            rel_e    = end   - win_s
            n_frames = S.shape[1]
            click_frames = [t for t in range(n_frames)
                            if t * hop + half >= rel_s and t * hop - half <= rel_e]
            if not click_frames:
                continue

            cf_set = set(click_frames)
            clean  = [t for t in range(n_frames) if t not in cf_set]
            if not clean:
                continue

            S_rep = S.copy()
            for t in click_frames:
                pre  = [i for i in clean if i < t]
                post = [i for i in clean if i > t]

                if pre and post:
                    p, q = pre[-1], post[0]
                    a = (t - p) / (q - p)
                    exp_mag = (1 - a) * np.abs(S[:, p]) + a * np.abs(S[:, q])
                elif pre:
                    exp_mag = np.abs(S[:, pre[-1]])
                else:
                    exp_mag = np.abs(S[:, post[0]])

                cur_mag = np.abs(S[:, t])
                # Gain: suppress where click inflated the bin; never amplify
                gain      = np.minimum(1.0, (exp_mag + 1e-10) / (cur_mag + 1e-10))
                S_rep[:, t] = S[:, t] * gain   # phase untouched

            _, rep = istft(S_rep, fs=sr, nperseg=n_fft,
                           noverlap=n_fft - hop, window='hann', boundary='even')
            rep = rep.astype(np.float32)

            fade  = hop
            a_s   = max(win_s, start - fade)
            a_e   = min(n,     end   + fade)
            rel_as = a_s - win_s
            rel_ae = a_e - win_s
            length = rel_ae - rel_as
            if length < 1 or rel_ae > len(rep):
                continue

            env = np.ones(length, dtype=np.float32)
            fl  = min(fade, length // 4)
            if fl > 0:
                env[:fl]  = np.linspace(0, 1, fl)
                env[-fl:] = np.linspace(1, 0, fl)

            out[a_s:a_e, ch] = (out[a_s:a_e, ch] * (1 - env) +
                                 rep[rel_as:rel_ae] * env)
    return out


def repair_clicks_spectral(data, clicks, sr, n_fft=256, hop=64):
    """
    Spectral repair: for each detected click, identify the STFT frames that
    overlap the click spike and replace them by interpolating between the
    nearest clean frames on each side.  The underlying music is preserved
    because we reconstruct both magnitude and phase from the clean context —
    no audio is invented.
    """
    if not clicks:
        return data

    out  = data.copy()
    n    = len(data)
    half = n_fft // 2

    for centre, start, end, fsim, novelty in sorted(clicks, key=lambda x: x[0]):
        ctx   = n_fft * 8                        # samples of clean context each side
        win_s = max(0, start - ctx)
        win_e = min(n,  end   + ctx)

        for ch in range(data.shape[1]):
            seg = out[win_s:win_e, ch].astype(np.float64)
            if len(seg) < n_fft * 2:
                continue

            _, _, S = stft(seg, fs=sr, nperseg=n_fft,
                           noverlap=n_fft - hop, window='hann', boundary='even')

            # Frames whose centre sample overlaps [start, end] relative to win_s
            rel_s = start - win_s
            rel_e = end   - win_s
            n_frames = S.shape[1]
            click_frames = [
                t for t in range(n_frames)
                if t * hop + half >= rel_s and t * hop - half <= rel_e
            ]

            if not click_frames:
                continue

            cf_set = set(click_frames)
            clean  = [t for t in range(n_frames) if t not in cf_set]
            if not clean:
                continue

            S_rep = S.copy()
            for t in click_frames:
                pre  = [i for i in clean if i < t]
                post = [i for i in clean if i > t]
                if pre and post:
                    p, q = pre[-1], post[0]
                    a = (t - p) / (q - p)
                    S_rep[:, t] = (1 - a) * S[:, p] + a * S[:, q]
                elif pre:
                    S_rep[:, t] = S[:, pre[-1]]
                else:
                    S_rep[:, t] = S[:, post[0]]

            _, rep = istft(S_rep, fs=sr, nperseg=n_fft,
                           noverlap=n_fft - hop, window='hann', boundary='even')
            rep = rep.astype(np.float32)

            # Crossfade only the click region ± one hop back into the output
            fade  = hop
            a_s   = max(win_s, start - fade)
            a_e   = min(n,     end   + fade)
            rel_as = a_s - win_s
            rel_ae = a_e - win_s
            length = rel_ae - rel_as
            if length < 1 or rel_ae > len(rep):
                continue

            env = np.ones(length, dtype=np.float32)
            fl  = min(fade, length // 4)
            if fl > 0:
                env[:fl]  = np.linspace(0, 1, fl)
                env[-fl:] = np.linspace(1, 0, fl)

            out[a_s:a_e, ch] = (out[a_s:a_e, ch] * (1 - env) +
                                 rep[rel_as:rel_ae] * env)
    return out


def _erase_vocal_clicks(vocal, clicks, sr, tail_pad=12, fade=6):
    """
    Zero the click region in a vocal residual signal, with a short linear
    crossfade on each side to prevent step discontinuities.

    tail_pad : samples to extend beyond the detected click bounds (covers tails)
    fade     : crossfade ramp length on each edge (samples)
    """
    out = vocal.copy()
    n   = len(vocal)
    for centre, start, end, fsim, novelty in clicks:
        s = max(0,     start - tail_pad)
        e = min(n - 1, end   + tail_pad)
        for ch in range(out.shape[1]):
            # Ramp down into silence at s, ramp up back at e
            f0 = min(fade, s)
            f1 = min(fade, n - 1 - e)
            if f0 > 0:
                out[s - f0:s, ch] *= np.linspace(1.0, 0.0, f0).astype(np.float32)
            out[s:e + 1, ch] = 0.0
            if f1 > 0:
                out[e + 1:e + 1 + f1, ch] *= np.linspace(0.0, 1.0, f1).astype(np.float32)
    return out


def repair_clicks_median_instrumental(data, clicks, instr, sr):
    """
    Median-filter click repair on the vocal residual only.

    The full mix is split into:
        vocal residual  =  data − instrumental  (contains click + true vocal)
        backing         =  instrumental          (clean — untouched)

    The median filter is run on just the vocal residual.  Because the residual
    is a single voice rather than a dense mix, its neighbourhood median is the
    true vocal waveform (not near-zero), so the click is replaced with the
    correct underlying audio, not silence.

    Reconstruction:  output = median_repaired_vocal_residual + instrumental
    """
    if not clicks:
        return data.copy()

    n = min(len(data), len(instr))
    data_n  = data[:n]
    instr_n = instr[:n]

    off = _global_offset(data_n, instr_n, sr)
    instr_al = _align_to(instr_n, off, n)
    if len(instr_al) < n:
        instr_al = np.pad(instr_al, ((0, n - len(instr_al)), (0, 0)))
    instr_al = instr_al[:n].astype(np.float32)

    rms_mix   = float(np.sqrt(np.mean(data_n**2)   + 1e-12))
    rms_instr = float(np.sqrt(np.mean(instr_al**2) + 1e-12))
    gain = min(rms_mix / (rms_instr + 1e-12) if rms_instr > 1e-8 else 1.0, 2.0)
    instr_scaled = instr_al * gain

    vocal = data_n - instr_scaled
    # Zero the click in the vocal residual (with a 6-sample crossfade on each
    # side).  At the click position the output will be purely instrumental —
    # the vocal is silent for ~0.3 ms, which is inaudible, while the
    # instrumental plays through cleanly.  This achieves true blue (silence)
    # in the vocal layer rather than a residual-energy red.
    vocal_repaired = _erase_vocal_clicks(vocal, clicks, sr, tail_pad=12, fade=6)

    out = vocal_repaired + instr_scaled
    if len(out) < len(data):
        out = np.pad(out, ((0, len(data) - len(out)), (0, 0)))
    return out[:len(data)]


def repair_clicks_spectral_instrumental(data, clicks, instr, sr, n_fft=256, hop=64):
    """
    Instrumental-guided spectral repair.

    The full mix is split into:
        vocal residual  =  data  −  instrumental
        backing         =  instrumental  (bass, drums, chords — untouched)

    Only the vocal residual STFT frames at click positions are interpolated.
    The instrumental passes through unchanged, so bass notes, hi-hats and chord
    harmonics are never re-synthesised — they come straight from the reference.

    Reconstruction:  output = repaired_vocal_residual + instrumental
    """
    if not clicks:
        return data.copy()

    n = min(len(data), len(instr))
    data_n  = data[:n]
    instr_n = instr[:n]

    # Global alignment (handles any remaining offset between the two files)
    off = _global_offset(data_n, instr_n, sr)
    instr_al = _align_to(instr_n, off, n)
    if len(instr_al) < n:
        instr_al = np.pad(instr_al, ((0, n - len(instr_al)), (0, 0)))
    instr_al = instr_al[:n].astype(np.float32)

    # Level-match instrumental to mix (simple RMS gain)
    rms_mix   = float(np.sqrt(np.mean(data_n **2) + 1e-12))
    rms_instr = float(np.sqrt(np.mean(instr_al**2) + 1e-12))
    gain = rms_mix / (rms_instr + 1e-12) if rms_instr > 1e-8 else 1.0
    # Don't over-amplify; cap gain at 2× to guard against silence segments
    gain = min(gain, 2.0)
    instr_scaled = instr_al * gain

    vocal = data_n - instr_scaled
    vocal_repaired = repair_clicks_spectral(vocal, clicks, sr, n_fft=n_fft, hop=hop)

    out = vocal_repaired + instr_scaled
    # Preserve original length
    if len(out) < len(data):
        out = np.pad(out, ((0, len(data) - len(out)), (0, 0)))
    return out[:len(data)]


# ══════════════════════════════════════════════════════════════════════════════
#  REFERENCE CONFIRMATION  (detection aid only — no reference audio used in repair)
# ══════════════════════════════════════════════════════════════════════════════

def _global_offset(ref, other, sr, search_sec=20):
    """Cross-correlate mono envelopes to find sample offset of other vs ref."""
    n = int(search_sec * sr)
    r = np.abs(ref[:n].mean(axis=1))
    o = np.abs(other[:n].mean(axis=1))
    corr = correlate(r, o, mode='full')
    return int(np.argmax(corr)) - (n - 1)

def _align_to(arr, offset, n):
    if offset >= 0: return arr[offset:][:n]
    return np.pad(arr, ((-offset, 0), (0, 0)))[:n]

def _build_click_severity_envelope(clicks, n_samples, sr, sigma_ms=300.0):
    """
    Build a smooth [0, 1] severity signal from detected click positions.

    Each click deposits an impulse weighted by its fingerprint similarity (fsim).
    A wide Gaussian smoothing then spreads each click into a gradual transition
    zone, so dense clusters merge into one broad region and isolated clicks
    produce a smaller bump.  The result is normalised by its own peak so the
    most heavily-clicked region always reaches 1.0.
    """
    from scipy.ndimage import gaussian_filter1d

    impulse = np.zeros(n_samples, dtype=np.float64)
    for centre, start, end, fsim, novelty in clicks:
        # fsim as weight floors at 0.3 so even weak matches register
        impulse[max(0, centre - 1):min(n_samples, centre + 2)] += max(0.3, float(fsim))

    sigma = max(1, int(sigma_ms / 1000 * sr))
    env   = gaussian_filter1d(impulse, sigma=sigma).astype(np.float32)

    peak = float(env.max())
    if peak > 1e-8:
        env /= peak          # densest cluster → 1.0, clean regions → 0.0
    return env


def blend_clicks_proportional(data, clicks, sr,
                               ref_single=None, ref_double=None,
                               sigma_ms=300.0):
    """
    Three-way proportional blend: original / single-export / double-export.

    Builds a smooth severity envelope from all detected click positions, then
    continuously mixes the three sources according to that envelope.  No point
    splices — the crossfade region is as wide as the click cluster (controlled
    by sigma_ms, default 300 ms).

      severity = 0.0  →  100 % original
      severity = 0.5  →  100 % single-export  (or linear blend toward double if
                          no single is given)
      severity = 1.0  →  100 % double-export  (falls back to single if no double)

    All three sources are globally aligned to the original via cross-correlation
    before blending, so there is no timing offset between layers.
    """
    if not clicks or (ref_single is None and ref_double is None):
        return data.copy()

    n        = len(data)
    severity = _build_click_severity_envelope(clicks, n, sr, sigma_ms=sigma_ms)

    # Align whichever references exist
    def _load_aligned(ref):
        if ref is None:
            return None
        off = _global_offset(data, ref, sr)
        al  = _align_to(ref, off, n)
        if len(al) < n:
            al = np.pad(al, ((0, n - len(al)), (0, 0)))
        return al[:n]

    al_single = _load_aligned(ref_single)
    al_double = _load_aligned(ref_double)

    # If only one reference is available, treat it as the upper tier
    if al_single is None and al_double is not None:
        al_single = al_double
        al_double = None
    if al_double is None:
        # Two-way blend: original ↔ single
        alpha = severity                     # shape (n,)
    else:
        # Three-way blend via two linear stages
        # Stage 1 (sev 0→0.5): original → single   (alpha1 goes 0→1)
        # Stage 2 (sev 0.5→1): single  → double    (alpha2 goes 0→1)
        pass   # handled per-channel below

    out = np.zeros_like(data)
    for ch in range(data.shape[1]):
        orig_ch = data[:, ch]

        def _ch(arr, c):
            if arr is None: return None
            return arr[:, c] if arr.shape[1] > c else arr[:, 0]

        s_ch = _ch(al_single, ch)
        d_ch = _ch(al_double, ch)

        if d_ch is None:
            # Two-way
            out[:, ch] = orig_ch * (1 - severity) + s_ch * severity
        else:
            # Three-way
            alpha1 = np.clip(severity * 2, 0.0, 1.0)       # original → single
            alpha2 = np.clip((severity - 0.5) * 2, 0.0, 1.0)  # single → double
            out[:, ch] = (orig_ch * (1.0 - alpha1) +
                          s_ch    * (alpha1 * (1.0 - alpha2)) +
                          d_ch    * alpha2)
    return out


def repair_clicks_median(data, clicks, sr, window=31):
    """
    Impulse removal via per-sample median filter.

    For each detected click, the click samples are replaced by the running
    median of their immediate neighbours.  A spike is eliminated because
    median([clean…, spike, clean…]) = clean — clean samples always outvote
    the spike as long as window > 2 × click_duration.
    A consonant transient (which ramps up over many ms) survives because the
    median of its neighbourhood is still the consonant shape — it never becomes
    a minority in the window.

    window: odd number of samples.  Must be larger than 2 × max_click_duration
            but much smaller than a consonant (consonant ≈ 5 ms = 220 samples).
            Default 31 samples ≈ 0.70 ms — handles clicks up to 15 samples wide,
            which covers all Suno artifact clicks (typically 8–13 samples).
    """
    from scipy.signal import medfilt

    if not clicks:
        return data.copy()

    out = data.copy()
    n   = len(data)
    pad = window * 4          # context each side so the median is stable

    for centre, start, end, fsim, novelty in clicks:
        s = max(0, start - pad)
        e = min(n, end   + pad)

        for ch in range(data.shape[1]):
            seg     = out[s:e, ch].copy()
            cleaned = medfilt(seg, kernel_size=window)

            # Crossfade repaired region back in over ± 4 samples at the edges
            fade   = 4
            rel_s  = start - s
            rel_e  = end   - s
            a_s    = max(0,       rel_s - fade)
            a_e    = min(len(seg), rel_e + fade)
            length = a_e - a_s
            if length < 1:
                continue

            env = np.ones(length, dtype=np.float32)
            fl  = min(fade, length // 3)
            if fl > 0:
                env[:fl]  = np.linspace(0.0, 1.0, fl)
                env[-fl:] = np.linspace(1.0, 0.0, fl)

            out[s + a_s:s + a_e, ch] = (
                out[s + a_s:s + a_e, ch] * (1.0 - env) +
                cleaned[a_s:a_e]          * env
            )
    return out


def _spectral_eq_match(source, target, sr, n_fft=2048, smooth_bins=20):
    """
    Shape source's spectral/dynamic character to match target's.

    Primary path: matchering 2.0 — matches frequency response, RMS,
    stereo width, and peak amplitude all at once (industry-standard approach
    used when releasing edited stems).

    Fallback (if matchering is unavailable): overlap-add FFT magnitude ratio
    — corrects average tonal balance only.
    """
    # ── Primary: matchering ───────────────────────────────────────────────────
    try:
        import matchering as mg
        import logging, io
        logging.getLogger('matchering').setLevel(logging.ERROR)

        tmp_src = tempfile.mktemp(suffix='_stems.wav')
        tmp_tgt = tempfile.mktemp(suffix='_ref.wav')
        tmp_out = tempfile.mktemp(suffix='_matched.wav')
        try:
            sf.write(tmp_src, source, sr, subtype='PCM_24')
            sf.write(tmp_tgt, target, sr, subtype='PCM_24')
            mg.process(target=tmp_src, reference=tmp_tgt,
                       results=[mg.pcm24(tmp_out)])
            result, _ = sf.read(tmp_out)
            if result.ndim == 1:
                result = result[:, np.newaxis]
            # Pad/trim to original length
            n = len(source)
            if len(result) < n:
                result = np.pad(result, ((0, n - len(result)), (0, 0)))
            return result[:n].astype(np.float32)
        finally:
            for f in (tmp_src, tmp_tgt, tmp_out):
                if os.path.exists(f): os.unlink(f)

    except Exception:
        pass  # fall through to manual implementation

    # ── Fallback: overlap-add FFT magnitude ratio ─────────────────────────────
    from scipy.ndimage import uniform_filter1d

    n   = min(len(source), len(target), 60 * sr)
    hop = n_fft // 2
    win = np.hanning(n_fft).astype(np.float64)

    src_mono = source[:n].mean(axis=1).astype(np.float64)
    tgt_mono = target[:n].mean(axis=1).astype(np.float64)

    src_m = np.zeros(n_fft // 2 + 1)
    tgt_m = np.zeros(n_fft // 2 + 1)
    count = 0
    for i in range(0, n - n_fft, hop):
        src_m += np.abs(np.fft.rfft(src_mono[i:i + n_fft] * win))
        tgt_m += np.abs(np.fft.rfft(tgt_mono[i:i + n_fft] * win))
        count += 1
    if count == 0:
        return source.copy()
    src_m /= count; tgt_m /= count

    gain = (tgt_m + 1e-8) / (src_m + 1e-8)
    gain = uniform_filter1d(gain, size=smooth_bins)
    gain = np.clip(gain, 0.1, 10.0)

    out = np.zeros_like(source)
    for ch in range(source.shape[1]):
        sig    = source[:, ch].astype(np.float64)
        result = np.zeros(len(sig))
        norm   = np.zeros(len(sig))
        for i in range(0, len(sig) - n_fft, hop):
            seg = sig[i:i + n_fft] * win
            S   = np.fft.rfft(seg)
            S  *= gain
            s   = np.fft.irfft(S, n=n_fft)
            result[i:i + n_fft] += s * win
            norm[i:i + n_fft]   += win ** 2
        norm   = np.maximum(norm, 1e-8)
        out[:, ch] = (result / norm).astype(np.float32)
    return out


def blend_stems_at_clicks(declicked, clicks, sr, stems_mix,
                           alpha=0.30, fade_ms=15.0):
    """
    At each repaired click position, blend in EQ-matched stems.

    A raised-cosine envelope peaks at `alpha` at the click centre and
    tapers to zero over `fade_ms` on each side.

      output = (1 − env) × declicked + env × stems_mix

    Because the stems don't contain the click artifact, this reintroduces
    real vocal/instrumental content at the repaired positions rather than
    the silence left by the erase step.

    alpha   : peak blend fraction (0.0 = no blend, 1.0 = full stems)
    fade_ms : half-width of the raised-cosine window in ms
    """
    if not clicks or alpha <= 0.0:
        return declicked.copy()

    n      = min(len(declicked), len(stems_mix))
    out    = declicked[:n].copy()
    stems  = stems_mix[:n]
    half   = int(fade_ms / 1000 * sr)

    for centre, start, end, fsim, novelty in clicks:
        s      = max(0, centre - half)
        e      = min(n, centre + half + 1)
        length = e - s
        if length < 2:
            continue
        t   = np.linspace(0.0, np.pi, length, dtype=np.float32)
        env = (0.5 - 0.5 * np.cos(t)) * alpha   # 0 → alpha → 0

        for ch in range(out.shape[1]):
            s_ch = stems[:, ch] if ch < stems.shape[1] else stems[:, 0]
            out[s:e, ch] = ((1.0 - env) * out[s:e, ch] + env * s_ch[s:e])

    if len(declicked) > n:
        out = np.concatenate([out, declicked[n:]], axis=0)
    return out


def confirm_clicks_from_reference(clicks, data, ref, sr,
                                   confirm_threshold=1.3,
                                   search_ms=25.0):
    """
    Use a reference recording (e.g. Suno Studio remaster) purely as a detection
    aid.  For each detected click, find the same moment in `ref` via
    cross-correlation.  If the reference has significantly lower amplitude there
    (ratio >= confirm_threshold), the click is confirmed as a genuine artifact.
    Unconfirmed clicks are still kept — reference confirmation is additive, not
    a filter.

    Returns a list of (centre, start, end, fsim, novelty, ref_confirmed) tuples.
    The caller always repairs spectrally; reference audio is never mixed into the
    output, so there is zero sonic mismatch between original and repaired audio.
    """
    if not clicks or ref is None:
        return [(c, s, e, f, v, False) for c, s, e, f, v in clicks]

    global_off = _global_offset(data, ref, sr)
    n          = len(data)
    ref_al     = _align_to(ref, global_off, n)
    if len(ref_al) < n:
        ref_al = np.pad(ref_al, ((0, n - len(ref_al)), (0, 0)))

    data_m = np.abs(data.mean(axis=1))
    ref_m  = np.abs(ref_al.mean(axis=1))
    search  = int(search_ms / 1000 * sr)
    ref_win = int(80.0      / 1000 * sr)

    result = []
    for centre, start, end, fsim, novelty in sorted(clicks, key=lambda x: x[0]):
        # Local alignment within ±search_ms
        rc_s      = max(0, centre - ref_win // 2)
        rc_e      = min(n, rc_s + ref_win)
        ref_chunk = data_m[rc_s:rc_e]
        ss        = max(0, centre - search)
        se        = min(n, centre + ref_win + search)
        srch      = ref_m[ss:se]

        if len(srch) >= len(ref_chunk) and len(ref_chunk) > 0:
            corr      = correlate(srch, ref_chunk, mode='valid')
            local_off = int(np.argmax(corr)) - search
        else:
            local_off = 0

        pad  = 10
        s0   = max(0, start - pad); s1 = min(n, end + pad)
        rs0  = max(0, s0 + local_off); rs1 = min(n, s1 + local_off)
        orig_peak = data_m[s0:s1].max()
        ref_peak  = ref_m[rs0:rs1].max() if rs1 > rs0 else orig_peak
        confirmed = (orig_peak / (ref_peak + 1e-10)) >= confirm_threshold

        result.append((centre, start, end, fsim, novelty, confirmed))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ARTIFACT REMOVER
# ══════════════════════════════════════════════════════════════════════════════

def remove_artifacts(mix, wet, instr, sr,
                     window_ms=4.0, ratio_threshold=3.5,
                     min_dur_ms=20.0, merge_ms=30.0, fade_ms=12.0):
    """
    Detect and repair sustained mix-level distortion bursts: regions where
    mix RMS is more than ratio_threshold times the stem sum RMS.

    The stem sum (wet + instr) represents the clean source. A high ratio
    means the mix has excess energy not present in any stem — a Suno
    mix-level clipping/distortion artifact.

    ratio_threshold=3.5 sits above the 90th-percentile of normal mix/stem
    variation caused by mastering, so only genuine spikes are caught.
    """
    n = min(len(mix), len(wet), len(instr))
    mix   = mix[:n].astype(np.float32)
    wet   = wet[:n].astype(np.float32)
    instr = instr[:n].astype(np.float32)

    win  = int(window_ms / 1000 * sr)
    step = max(1, win // 4)

    mix_m  = mix.mean(axis=1)
    stem_m = wet.mean(axis=1) + instr.mean(axis=1)

    def _rms(sig, i):
        chunk = sig[i * step: i * step + win]
        return float(np.sqrt(np.mean(chunk ** 2) + 1e-12))

    n_frames = max(1, (n - win) // step)
    mix_rms  = np.array([_rms(mix_m,  i) for i in range(n_frames)])
    stem_rms = np.array([_rms(stem_m, i) for i in range(n_frames)])

    ratio    = mix_rms / (stem_rms + 1e-8)

    art_frames = np.where(ratio > ratio_threshold)[0]
    if len(art_frames) == 0:
        return mix.copy(), []

    # Merge nearby artifact frames, then filter by minimum duration
    merge_gap  = max(1, int(merge_ms  / 1000 * sr / step))
    min_frames = max(1, int(min_dur_ms / 1000 * sr / step))

    groups, g = [], [art_frames[0]]
    for f in art_frames[1:]:
        if f - g[-1] <= merge_gap: g.append(f)
        else: groups.append(g); g = [f]
    groups.append(g)
    groups = [g for g in groups if len(g) >= min_frames]

    if not groups:
        return mix.copy(), []

    # Scale: in frames where both mix and stems are active but ratio is normal,
    # estimate the typical mix/stem gain so the repair matches the mix level.
    normal = np.where((ratio > 0.5) & (ratio < 2.0) & (stem_rms > 1e-4))[0]
    if len(normal) > 20:
        scale = float(np.clip(np.median(mix_rms[normal] / stem_rms[normal]), 0.3, 3.0))
    else:
        scale = 1.0

    target = (wet + instr) * scale
    out    = mix.copy()
    fade   = int(fade_ms / 1000 * sr)
    regions = []

    for g in groups:
        s = max(0, g[0]  * step - fade)
        e = min(n, g[-1] * step + win + fade)
        regions.append((s, e, float(ratio[g].mean())))

        length = e - s
        env    = np.ones(length, dtype=np.float32)
        fl     = min(fade, length // 4)
        if fl > 0:
            env[:fl]  = np.linspace(0, 1, fl)
            env[-fl:] = np.linspace(1, 0, fl)

        for ch in range(mix.shape[1]):
            out[s:e, ch] = (mix[s:e, ch] * (1 - env) +
                            target[s:e, ch] * env)

    return out, regions


# ══════════════════════════════════════════════════════════════════════════════
#  A/B COMPARATOR  (inline — no external script needed)
# ══════════════════════════════════════════════════════════════════════════════

def show_comparison(orig, enh, sr, original_path, enhanced_path):
    """
    3-panel full-song spectrogram + click-by-click audio preview.

    orig / enh  — in-memory numpy arrays (avoids MP3 quantisation noise
                  contaminating the diff panel).
    original_path / enhanced_path — used only for saving the image.

    Top    — original (clicks visible as bright vertical lines)
    Middle — diff (original − enhanced): shows ONLY what was removed
    Bottom — declicked (clean)

    Then plays the top-N loudest click regions back-to-back:
    original first, then the same segment from the declicked version.
    """
    try:
        import sounddevice as sd
        _have_sd = True
    except ImportError:
        _have_sd = False

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from scipy.signal import stft as _stft
    except ImportError:
        print("  (matplotlib not available — skipping visual comparison)")
        return

    n = min(len(orig), len(enh))
    orig, enh = orig[:n], enh[:n]
    diff = orig - enh                       # what was removed — only the clicks

    orig_m = orig.mean(axis=1).astype(np.float64)
    enh_m  = enh.mean(axis=1).astype(np.float64)
    diff_m = diff.mean(axis=1).astype(np.float64)

    nperseg = 2048
    hop     = 512
    f_max   = 8000

    def _stft_mag(mono):
        freqs, times, Zxx = _stft(mono, fs=sr, nperseg=nperseg,
                                   noverlap=nperseg - hop, boundary='even')
        mag  = 20 * np.log10(np.abs(Zxx) + 1e-8)
        mask = freqs <= f_max
        return times, freqs[mask], mag[mask]

    def _norm(mag, vmax, vmin):
        return np.clip((mag - vmin) / (vmax - vmin), 0.0, 1.0)

    print("  [Comparison] Computing spectrograms (this takes a moment)...")
    times, freqs, mag_orig = _stft_mag(orig_m)
    _,     _,     mag_enh  = _stft_mag(enh_m)
    _,     _,     mag_diff = _stft_mag(diff_m)

    # Use the original's dynamic range as reference so the diff panel
    # shows only energy that is large relative to the original signal.
    vmax = float(np.percentile(mag_orig, 99))
    vmin = vmax - 60

    S_orig = _norm(mag_orig, vmax, vmin)
    S_enh  = _norm(mag_enh,  vmax, vmin)
    S_diff = _norm(mag_diff, vmax, vmin)   # dark unless click-sized energy

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10), facecolor='#080808')
    gs  = GridSpec(3, 1, figure=fig, hspace=0.08)
    ax_orig = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1])
    ax_enh  = fig.add_subplot(gs[2])

    kHz = freqs / 1000
    panels = [
        (ax_orig, S_orig, 'Original  (clicks = bright vertical lines)', 'inferno'),
        (ax_diff, S_diff, 'Removed   (diff = original − declicked)',    'hot'),
        (ax_enh,  S_enh,  'Declicked (clean)',                          'inferno'),
    ]
    for ax, S, title, cmap in panels:
        ax.pcolormesh(times, kHz, S, cmap=cmap, shading='gouraud', vmin=0, vmax=1)
        ax.set_facecolor('#080808')
        ax.set_ylabel('kHz', color='#aaa', fontsize=8)
        ax.set_title(title, color='white', fontsize=10, loc='left', pad=3)
        ax.tick_params(colors='#666', labelsize=7)
        ax.set_ylim(0, f_max / 1000)
        for sp in ax.spines.values():
            sp.set_edgecolor('#222')

    ax_enh.set_xlabel('time (s)', color='#aaa', fontsize=8)
    ax_orig.set_xticklabels([])
    ax_diff.set_xticklabels([])

    img_path = os.path.splitext(enhanced_path)[0] + '_comparison.png'
    plt.savefig(img_path, dpi=120, facecolor='#080808', bbox_inches='tight')
    plt.close()
    print(f"  [Comparison] Saved → {os.path.basename(img_path)}")
    try:
        subprocess.run(['xdg-open', img_path], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # ── Audio playback — full declicked song ─────────────────────────────────
    if not _have_sd:
        print("  (sounddevice not installed — skipping audio playback)")
        return

    print("\n  Playing declicked song — press Ctrl+C to stop.\n")
    try:
        sd.play(enh.copy(), sr, blocking=True)
    except KeyboardInterrupt:
        sd.stop()
        print("\n  Stopped early.\n")


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

def _pick(prompt, files, directory=".", exclude=None):
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
            if m: return os.path.join(directory, m[0])
        except (ValueError, EOFError):
            pass
        print("  Invalid — try again.")

def _yn(prompt):
    return input(f"  {prompt} (y/N): ").strip().lower() == 'y'

def _output_format(paths):
    return "mp3" if all(p.lower().endswith(".mp3") for p in paths if p) else "wav"

def _save_output(wav_data, sr, out_path, fmt):
    if fmt == "wav":
        sf.write(out_path, wav_data, sr, subtype="PCM_24")
    else:
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            sf.write(tmp, wav_data, sr, subtype="PCM_24")
            subprocess.run(["ffmpeg", "-y", "-i", tmp, "-codec:a", "libmp3lame",
                            "-qscale:a", "0", out_path],
                           capture_output=True, stdin=subprocess.DEVNULL, check=True)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

def _mp3s_in(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp3"))


def _wait_for_exactly(folder, count, action_hint):
    while True:
        files = _mp3s_in(folder)
        n = len(files)
        if n == count:
            return files
        if n < count:
            print(f"\n  ! {n} file(s) in folder — need {count}.  "
                  f"({count - n} more to add: {action_hint})")
        else:
            print(f"\n  ! {n} file(s) in folder — only {count} should be there.  "
                  f"Remove {n - count} extra file(s).")
        input("  [Press Enter to check again]\n")


def _interactive():
    """Guided wizard — returns (original, wet_vocal, instrumental, dry_vocal,
    session_dir, out_path) with repair_mode already decided as 'median'."""
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
    input("  [Press Enter to continue]\n")

    # ── Step 1 — Instrumental export ─────────────────────────────────────────
    print("  ┌─ Step 1 of 3 — Export the instrumental from Suno Studio ─┐")
    print()
    print("  1. Open your song in Suno Studio.")
    print("  2. IMPORTANT: disable beat grid snap before exporting.")
    print("     (Snap ON causes timing drift that misaligns the stems.)")
    print("  3. Export  Instrumental  (not the full song).")
    print("  4. Download the file.")
    print()
    input("  Done downloading the instrumental?\n  [Press Enter to continue]\n")

    while True:
        name = input("  Song name (used for folder name): ").strip()
        if name:
            break
        print("  Please enter a name.")

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    os.makedirs(folder, exist_ok=True)
    print(f"\n  Folder: {folder}")
    try:
        subprocess.run(["xdg-open", folder], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    print()
    print("  Place ONLY the instrumental file into that folder.")
    print()
    input("  [Press Enter when it is in the folder]\n")
    files_after_1    = _wait_for_exactly(folder, 1, "the instrumental file")
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

    # ── Step 3 — Confirm ─────────────────────────────────────────────────────
    print()
    print("  ┌─ Step 3 of 3 — Declicking ────────────────────────────────┐")
    print()
    print(f"  Original     : {original_name}")
    print(f"  Instrumental : {instrumental_name}")
    print()
    input("  Press Enter to start — this may take a minute.\n  [Press Enter to continue]\n")

    out_path = os.path.join(folder, f"{name}_declicked.mp3")
    return original, None, instrumental, None, folder, out_path


def _run_declicker_pass(data, sr, pass_num, total_passes,
                         dry_vocal, ref_data, threshold, similarity,
                         repair_mode="spectral", instrumental=None):
    print(f"  [Pass {pass_num}/{total_passes}] Declicker")
    clicks = detect_clicks(data, sr, threshold, similarity, dry_vocal=dry_vocal)
    if not clicks:
        print("        No Suno artifact clicks detected.\n")
        return data

    # Optionally use reference to confirm each click (detection aid only —
    # reference audio is never used for repair to avoid sonic mismatch).
    if ref_data is not None:
        confirmed_clicks = confirm_clicks_from_reference(clicks, data, ref_data, sr)
        n_conf = sum(1 for *_, ok in confirmed_clicks if ok)
        print(f"        Found {len(clicks)} click(s)  ({n_conf} reference-confirmed):\n")
        print(f"  {'#':>4}  {'Time':>8}  {'Dur':>7}  {'Match':>7}  {'Ref':>5}")
        print(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*5}")
        for i, (c, s, e, fsim, nov, ok) in enumerate(confirmed_clicks):
            t = c / sr; m, sec = divmod(t, 60)
            tag = "✓" if ok else " "
            print(f"  {i+1:>4}  {int(m):02d}:{sec:05.2f}  {(e-s)/sr*1000:>5.2f}ms  {fsim:>7.3f}  {tag:>5}")
        # Strip the confirmed flag before passing to repair
        clicks_for_repair = [(c, s, e, f, v) for c, s, e, f, v, _ in confirmed_clicks]
    else:
        print(f"        Found {len(clicks)} click(s):\n")
        print(f"  {'#':>4}  {'Time':>8}  {'Dur':>7}  {'Match':>7}")
        print(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*7}")
        for i, (c, s, e, fsim, nov) in enumerate(clicks):
            t = c / sr; m, sec = divmod(t, 60)
            print(f"  {i+1:>4}  {int(m):02d}:{sec:05.2f}  {(e-s)/sr*1000:>5.2f}ms  {fsim:>7.3f}")
        clicks_for_repair = clicks

    if repair_mode == "gate":
        print("\n        Repairing with per-bin spectral magnitude gate...")
        data = repair_clicks_spectral_gate(data, clicks_for_repair, sr)
    elif repair_mode == "ar":
        print("\n        Repairing with AR prediction (forward+backward blend)...")
        data = repair_clicks_ar(data, clicks_for_repair, sr)
    elif repair_mode == "fill":
        print("\n        Erasing click tails and filling gap with spline...")
        data = erase_and_fill_clicks(data, clicks_for_repair, sr)
    elif repair_mode == "delete":
        print("\n        Deleting click samples and stitching...")
        data = delete_clicks(data, clicks_for_repair, sr)
        removed = sum(e - s for _, s, e, *_ in clicks_for_repair)
        print(f"        Removed {removed} samples ({removed/sr*1000:.2f} ms total).\n")
    elif repair_mode == "median":
        if instrumental is not None:
            print("\n        Repairing with instrumental-guided median filter...")
            data = repair_clicks_median_instrumental(data, clicks_for_repair, instrumental, sr)
        else:
            print("\n        Repairing with median impulse filter...")
            data = repair_clicks_median(data, clicks_for_repair, sr)
    elif instrumental is not None:
        print("\n        Repairing with instrumental-guided spectral interpolation...")
        data = repair_clicks_spectral_instrumental(data, clicks_for_repair, instrumental, sr)
    else:
        print("\n        Repairing with spectral interpolation...")
        data = repair_clicks_spectral(data, clicks_for_repair, sr)
    print("        Done.\n")
    return data


def main():
    ap = argparse.ArgumentParser(description="Suno AI audio enhancement pipeline.")
    ap.add_argument("input",          nargs="?", default=None)
    ap.add_argument("--vocal",        default=None, help="Wet vocal stem (with effects)")
    ap.add_argument("--dry-vocal",    default=None, help="Dry vocal stem (no effects) — used to verify long artifacts aren't music")
    ap.add_argument("--instrumental", default=None)
    ap.add_argument("--reference",    default=None,
                    help="Single Suno Studio export — moderate-severity blend tier")
    ap.add_argument("--reference2",   default=None,
                    help="Double Suno Studio export — extreme-severity blend tier")
    ap.add_argument("--sigma",        type=float, default=300.0,
                    help="Blend envelope smoothing width in ms (default: 300)")
    ap.add_argument("--out",          default=None)
    ap.add_argument("--threshold",    type=float, default=5.0)
    ap.add_argument("--similarity",   type=float, default=0.70)
    ap.add_argument("--passes",       type=int,   default=3,
                    help="Spectral-repair passes when no references given (default: 3)")
    ap.add_argument("--repair-mode",  default="spectral",
                    choices=["spectral", "median", "delete", "fill", "ar", "gate"],
                    help="spectral=STFT interpolation (default), median=impulse filter, "
                         "ar=autoregressive prediction (iZotope-style, no external file)")
    ap.add_argument("--stem-blend-alpha", type=float, default=0.0,
                    help="Blend stems into repaired click positions. "
                         "0.0=off, 0.3=30%% stems at click centre. "
                         "Requires --vocal + --instrumental.")
    ap.add_argument("--stem-blend-fade",  type=float, default=15.0,
                    help="Half-width of stem blend window in ms (default: 15)")
    ap.add_argument("--no-compare",   action="store_true")
    args = ap.parse_args()

    session_dir  = "."
    wet_vocal    = args.vocal
    dry_vocal    = args.dry_vocal
    reference    = args.reference
    reference2   = args.reference2

    if not args.input:
        (args.input, wet_vocal, args.instrumental,
         dry_vocal, session_dir, args.out) = _interactive()
        args.repair_mode = "median"

    for p in filter(None, [args.input, wet_vocal, args.instrumental,
                            dry_vocal, reference, reference2]):
        if not os.path.isfile(p):
            sys.exit(f"File not found: {p}")

    all_inputs = [p for p in [args.input, wet_vocal, args.instrumental,
                              dry_vocal, reference, reference2] if p]
    fmt        = _output_format(all_inputs)
    stem       = os.path.splitext(os.path.basename(args.input))[0]
    out_path   = args.out or os.path.join(session_dir, stem + "_enhanced." + fmt)

    if os.path.abspath(out_path) == os.path.abspath(args.input):
        sys.exit("Output path matches input — refusing to overwrite.")

    use_blend = bool(reference or reference2)

    print(f"\n{'━'*54}")
    print(f"  suno-enhance")
    print(f"{'━'*54}")
    print(f"  Input        : {os.path.basename(args.input)}")
    if wet_vocal:    print(f"  Vocal        : {os.path.basename(wet_vocal)}")
    if args.instrumental: print(f"  Instrumental : {os.path.basename(args.instrumental)}")
    if dry_vocal:    print(f"  Dry vocal    : {os.path.basename(dry_vocal)}")
    if reference:    print(f"  Ref (single) : {os.path.basename(reference)}")
    if reference2:   print(f"  Ref (double) : {os.path.basename(reference2)}")
    has_stems = bool(wet_vocal and args.instrumental)
    if use_blend:
        tiers = []
        if reference:  tiers.append("single-export")
        if reference2: tiers.append("double-export")
        steps_str = f"detect → proportional blend ({'+'.join(tiers)}, σ={args.sigma}ms)"
    else:
        steps_str = f"declicker × {args.passes}"
    if has_stems: steps_str += " → artifact remover"
    print(f"  Steps        : {steps_str}")
    print(f"  Output       : {os.path.basename(out_path)}")
    print(f"{'━'*54}\n")

    data, sr = load(args.input)
    orig_data = data.copy()   # keep unmodified original for comparison (avoids MP3 re-encode diff)
    print(f"  {sr} Hz | {data.shape[1]}ch | {len(data)/sr:.1f}s\n")

    vocal_data   = None
    instr        = None
    dry_voc_data = None
    ref_single   = None
    ref_double   = None

    if args.instrumental:
        instr, _ = load(args.instrumental)
    if wet_vocal and args.instrumental:
        vocal_data, _ = load(wet_vocal)
    if dry_vocal:
        dry_voc_data, _ = load(dry_vocal)
    if reference:
        print(f"  Loading reference (single export)...")
        ref_single, _ = load(reference)
        print()
    if reference2:
        print(f"  Loading reference (double export)...")
        ref_double, _ = load(reference2)
        print()

    if use_blend:
        # ── proportional blend mode ───────────────────────────────────────────
        # Detect clicks once (on the original), then build a severity envelope
        # and blend all sources proportionally.  No iterative repair — the wide
        # Gaussian naturally covers clusters.
        print(f"  [Declicker — detection pass]")
        clicks = detect_clicks(data, sr, args.threshold, args.similarity,
                               dry_vocal=dry_voc_data)
        if not clicks:
            print("        No Suno artifact clicks detected.\n")
        else:
            print(f"        Found {len(clicks)} click(s).\n")
            print(f"  [Blend] Building severity envelope (σ={args.sigma}ms)...")
            data = blend_clicks_proportional(data, clicks, sr,
                                             ref_single=ref_single,
                                             ref_double=ref_double,
                                             sigma_ms=args.sigma)
            print(f"        Done.\n")
    else:
        # ── iterative spectral repair mode ────────────────────────────────────
        for pass_n in range(1, args.passes + 1):
            data = _run_declicker_pass(data, sr, pass_n, args.passes,
                                       dry_voc_data, None,
                                       args.threshold, args.similarity,
                                       repair_mode=args.repair_mode,
                                       instrumental=instr)

    # ── stem blend at click positions ────────────────────────────────────────
    if (args.stem_blend_alpha > 0.0
            and vocal_data is not None and instr is not None):
        print(f"  [Stem blend]  α={args.stem_blend_alpha:.2f}  "
              f"fade={args.stem_blend_fade:.0f}ms")

        # Re-detect on original so we know where every click was
        orig_data, _ = load(args.input)
        blend_clicks = detect_clicks(orig_data, sr,
                                     args.threshold, args.similarity)
        if not blend_clicks:
            print("        No clicks found on original — skipping stem blend.\n")
        else:
            print(f"        Using {len(blend_clicks)} click position(s).")

            # Align and EQ-match the stem mix to the declicked output
            n = len(data)
            off_v = _global_offset(data, vocal_data, sr)
            off_i = _global_offset(data, instr,      sr)
            v_al  = _align_to(vocal_data, off_v, n)
            i_al  = _align_to(instr,      off_i, n)
            if len(v_al) < n:
                v_al = np.pad(v_al, ((0, n - len(v_al)), (0, 0)))
            if len(i_al) < n:
                i_al = np.pad(i_al, ((0, n - len(i_al)), (0, 0)))
            stems_raw = (v_al[:n] + i_al[:n]).astype(np.float32)

            print("        EQ-matching stems to original...")
            stems_eq = _spectral_eq_match(stems_raw, data, sr)
            print(f"        Blending at {len(blend_clicks)} position(s)...")
            data = blend_stems_at_clicks(data, blend_clicks, sr, stems_eq,
                                         alpha=args.stem_blend_alpha,
                                         fade_ms=args.stem_blend_fade)
            print("        Done.\n")

    # ── artifact remover ─────────────────────────────────────────────────────
    # Skip when stem blend is active — the blend already handles repair and
    # the artifact remover needs a wet (processed) vocal which may not be
    # the same file provided for the stem blend.
    if vocal_data is not None and instr is not None and args.stem_blend_alpha <= 0.0:
        print(f"  [Artifact remover]")
        data, regions = remove_artifacts(data, vocal_data, instr, sr)
        if not regions:
            print("        No sustained artifacts detected.\n")
        else:
            print(f"        Found {len(regions)} artifact region(s):\n")
            print(f"  {'#':>4}  {'Start':>8}  {'End':>8}  {'Residual':>9}")
            print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*9}")
            for i, (s, e, r) in enumerate(regions):
                def _fmt(smp): m, sec = divmod(smp / sr, 60); return f"{int(m):02d}:{sec:05.2f}"
                print(f"  {i+1:>4}  {_fmt(s):>8}  {_fmt(e):>8}  {r:>9.3f}")
            print()

    print(f"  Saving as {fmt.upper()}...")
    _save_output(data, sr, out_path, fmt)
    print(f"\n  Original untouched : {args.input}")
    print(f"  Enhanced copy saved: {out_path}")
    print(f"{'━'*54}\n")

    if not args.no_compare:
        ans = input("  Play and show clicks? (y/N): ").strip().lower()
        if ans == 'y':
            show_comparison(orig_data, data, sr, args.input, out_path)


if __name__ == "__main__":
    main()
