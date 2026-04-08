"""
suno-deshimmer — removes AI generation noise floor (hiss/shimmer) from Suno songs,
then restores natural spaciousness with a clean reverb tail in the filtered band.

The artifact: a broadband noise floor that Suno's AI generation process bakes into
the full mix — sounds like white noise / microphone hiss / static, present from the
very first second, modulating slightly in amplitude under the music.

Key insight: Suno's own stem extractor removes this noise as a byproduct of separation.
The extracted instrumental stem has essentially zero noise floor. Therefore:

  mix - instrumental (at vocal-absent frames) = pure noise fingerprint

We use the first pre-vocal seconds of the song (where mix ≈ instrumental + noise,
no vocals yet) to extract an exact per-bin noise profile, then subtract it
from the full mix with a floor to protect musical content.

Fallback (no instrumental): minimum-statistics noise floor estimation on mix alone.

Usage:
    python deshimmer.py song.wav --instrumental instr.mp3
    python deshimmer.py song.wav --instrumental instr.mp3 --strength 0.85 --floor 0.80
    python deshimmer.py song.wav                           # fallback: min-stats only
"""

import sys, os, argparse, tempfile, subprocess
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, fftconvolve, butter, sosfiltfilt


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


# ── noise fingerprint from instrumental reference ──────────────────────────────

def noise_fingerprint_from_stem(mix, instr, sr, ref_seconds=4.0, n_fft=2048, hop=512):
    """
    Extract noise fingerprint using instrumental stem as reference.

    Uses the first ref_seconds of the song where vocals haven't started yet.
    At that point: mix ≈ instrumental + noise  →  mix - instr ≈ noise

    Cross-correlates the two signals first to handle any small sample offset.
    Returns a 1D array (n_freqs,) — the mean noise magnitude per frequency bin.
    """
    n_ref = int(ref_seconds * sr)
    n_ref = min(n_ref, len(mix), len(instr))

    # mono for alignment
    mx = mix[:n_ref].mean(axis=1)
    ix = instr[:n_ref].mean(axis=1)

    # coarse alignment via cross-correlation (handles Suno's fixed export offset)
    corr   = np.correlate(mx, ix, mode='full')
    offset = int(np.argmax(np.abs(corr)) - (len(ix) - 1))
    offset = np.clip(offset, -sr // 4, sr // 4)   # max ±0.25s search
    if offset > 0:
        ix = np.pad(ix, (offset, 0))[:n_ref]
    elif offset < 0:
        ix = ix[-offset:]
        ix = np.pad(ix, (0, n_ref - len(ix)))

    # noise = mix - instrumental, averaged across channels
    fingerprint = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    count = 0
    for ch in range(mix.shape[1]):
        m = mix[:n_ref, ch]
        # apply same offset to this channel's instrumental
        i_ch = instr[:n_ref, ch] if ch < instr.shape[1] else instr[:n_ref, 0]
        if offset > 0:
            i_ch = np.pad(i_ch, (offset, 0))[:n_ref]
        elif offset < 0:
            i_ch = i_ch[-offset:]
            i_ch = np.pad(i_ch, (0, n_ref - len(i_ch)))
        noise_sig = m - i_ch
        _, _, N = stft(noise_sig, fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, window='hann', boundary='even')
        fingerprint += np.abs(N).mean(axis=1)
        count += 1

    fingerprint /= count
    print(f"  Peak noise bin : {np.argmax(fingerprint) * sr / n_fft:.0f} Hz")
    print(f"  Alignment offset used: {offset} samples ({offset/sr*1000:.1f} ms)")
    return fingerprint.astype(np.float32)


# ── fallback: minimum-statistics noise floor ──────────────────────────────────

def estimate_noise_floor(mag, window_frames=50):
    """Per-bin rolling minimum — the noise floor that's always present."""
    n_freqs, n_frames = mag.shape
    noise = np.empty_like(mag)
    half  = window_frames // 2
    for t in range(n_frames):
        lo = max(0, t - half)
        hi = min(n_frames, t + half + 1)
        noise[:, t] = mag[:, lo:hi].min(axis=1)
    return noise


# ── deshimmer ─────────────────────────────────────────────────────────────────

def deshimmer(song, sr, fingerprint=None, strength=0.85, floor=0.80,
              band_lo=500, n_fft=2048, hop=512, noise_window=50):
    """
    Subtract noise fingerprint from song.

    If fingerprint is provided (from instrumental stem), subtract that fixed
    profile from every frame — scaled by strength, floored at floor*original.

    If fingerprint is None, estimate per-frame noise floor via rolling minimum
    (fallback when no instrumental is available).

    band_lo: don't touch anything below this Hz — protects bass and body.
    """
    n     = len(song)
    out   = np.zeros_like(song)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    # frequency mask with soft fade at band_lo boundary
    freq_mask = (freqs >= band_lo).astype(np.float32)
    lo_idx = np.searchsorted(freqs, band_lo)
    fade   = 10
    for i in range(fade):
        idx = lo_idx - fade + i
        if 0 <= idx < len(freq_mask):
            freq_mask[idx] = (i + 1) / (fade + 1)

    for ch in range(song.shape[1]):
        _, _, S = stft(song[:, ch], fs=sr, nperseg=n_fft,
                       noverlap=n_fft - hop, window='hann', boundary='even')
        mag   = np.abs(S).astype(np.float32)
        phase = np.angle(S)

        if fingerprint is not None:
            # fixed fingerprint: same profile subtracted from every frame
            subtraction = strength * fingerprint[:, None] * freq_mask[:, None]
        else:
            # fallback: rolling minimum noise floor per frame
            noise_floor  = estimate_noise_floor(mag, window_frames=noise_window)
            subtraction  = strength * noise_floor * freq_mask[:, None]

        hard_floor = mag * floor
        mag_clean  = np.maximum(mag - subtraction, hard_floor)

        Z_clean = mag_clean * np.exp(1j * phase)
        _, rep  = istft(Z_clean, fs=sr, nperseg=n_fft,
                        noverlap=n_fft - hop, window='hann', boundary='even')

        rep = rep[:n] if len(rep) >= n else np.pad(rep, (0, n - len(rep)))
        out[:, ch] = rep.astype(np.float32)

    return out


# ── plate reverb IR + application ─────────────────────────────────────────────

def make_plate_ir(sr, rt60=1.5, pre_delay_ms=20.0):
    """
    Synthetic plate reverb impulse response — smooth, dense, no resonant peaks.

    Method:
      1. Exponentially decaying Gaussian noise (dense, no discrete echoes)
      2. Frequency-domain magnitude smoothing (kills spectral resonances)
      3. Frequency shaping: HF rolloff at 7kHz (warmth), LF rolloff at 200Hz
         (keeps kick/snare reverb imperceptible)
      4. Pre-delay (separation between dry transient and reverb onset)

    rt60         : reverberation time in seconds (time to decay 60 dB)
    pre_delay_ms : delay before reverb onset in ms
    """
    from scipy.ndimage import uniform_filter1d

    ir_len = int(rt60 * sr * 1.3)     # 30% extra for complete tail
    t      = np.arange(ir_len) / sr

    # dense Gaussian noise × exponential decay envelope
    rng   = np.random.default_rng(42)
    noise = rng.standard_normal(ir_len)
    env   = np.exp(-6.908 * t / rt60)  # exactly -60 dB at rt60
    ir    = noise * env

    # short build-up (5ms) — avoids click at onset
    build = int(0.005 * sr)
    ir[:build] *= np.linspace(0, 1, build)

    # frequency-domain smoothing — critical step: removes spectral resonances
    # without this the reverb has a colored, resonant character
    IR    = np.fft.rfft(ir)
    mag   = np.abs(IR)
    phase = np.angle(IR)
    mag   = uniform_filter1d(mag, size=80)      # smooth over ~80 bins
    mag   = uniform_filter1d(mag, size=80)      # second pass for extra smoothness
    ir    = np.fft.irfft(mag * np.exp(1j * phase), n=ir_len)

    # frequency shaping
    # HF rolloff at 7kHz: warm tail, no metallic highs
    sos_lp = butter(3, 7000 / (sr / 2), btype='low',  output='sos')
    ir = sosfiltfilt(sos_lp, ir)
    # LF rolloff at 200Hz: kick and bass stay dry
    sos_hp = butter(3,  200 / (sr / 2), btype='high', output='sos')
    ir = sosfiltfilt(sos_hp, ir)

    # pre-delay
    pre = int(pre_delay_ms / 1000 * sr)
    ir  = np.pad(ir, (pre, 0))

    # normalize to unit peak
    ir /= np.abs(ir).max() + 1e-10
    return ir.astype(np.float32)


def add_reverb_tail(signal, sr, band_lo=500, rt60=1.5, pre_delay_ms=20.0, wet=0.10):
    """
    Apply plate reverb to the signal above band_lo Hz and mix at wet level.

    Derives reverb only from frequencies above band_lo — kick/bass fundamentals
    are excluded so their reverb contribution is inaudible at 10% wet.

    wet=0.10 means 100% dry + 10% reverb, exactly as user described.
    """
    n   = len(signal)
    ir  = make_plate_ir(sr, rt60=rt60, pre_delay_ms=pre_delay_ms)
    sos = butter(4, band_lo / (sr / 2), btype='high', output='sos')
    out = np.zeros_like(signal)

    for ch in range(signal.shape[1]):
        # reverb source: only high-passed signal (avoids kick/bass reverb)
        src = sosfiltfilt(sos, signal[:, ch])
        rev = fftconvolve(src, ir)[:n]
        out[:, ch] = signal[:, ch] + wet * rev.astype(np.float32)

    # safety clip
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Remove AI generation noise floor from Suno songs using instrumental stem.")
    ap.add_argument("input",            nargs="?", default=None, help="Full song mix")
    ap.add_argument("--instrumental",   default=None, help="Instrumental stem (Suno-extracted, noise-free)")
    ap.add_argument("--out",            default=None)
    ap.add_argument("--ref-seconds",    type=float, default=4.0,
                    help="Seconds of pre-vocal audio to use for noise fingerprint (default 4.0)")
    ap.add_argument("--strength",       type=float, default=0.85,
                    help="Subtraction strength 0–1 (default 0.85)")
    ap.add_argument("--floor",          type=float, default=0.80,
                    help="Min magnitude floor as fraction of original (default 0.80)")
    ap.add_argument("--band-lo",        type=float, default=500,
                    help="Don't touch below this Hz (default 500)")
    ap.add_argument("--noise-window",   type=int,   default=50,
                    help="Fallback rolling minimum window in frames (default 50)")
    ap.add_argument("--reverb-wet",     type=float, default=0.10,
                    help="Reverb mix level 0–1 (default 0.10 = 10%%)")
    ap.add_argument("--reverb-time",    type=float, default=1.5,
                    help="Reverb decay RT60 in seconds (default 1.5)")
    ap.add_argument("--no-reverb",      action="store_true",
                    help="Skip reverb restoration step")
    args = ap.parse_args()

    if not args.input:
        files = list_audio_files()
        if not files:
            sys.exit("No audio files found in the current directory.")
        args.input = pick_file("Select the full song mix:", files)

    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")
    if args.instrumental and not os.path.isfile(args.instrumental):
        sys.exit(f"File not found: {args.instrumental}")

    out_path = args.out or (os.path.splitext(args.input)[0] + "_deshimmered.wav")
    if os.path.abspath(out_path) == os.path.abspath(args.input):
        sys.exit("Output path matches input — refusing to overwrite.")

    print(f"\nsuno-deshimmer")
    print(f"  Input        : {os.path.basename(args.input)}")
    if args.instrumental:
        print(f"  Instrumental : {os.path.basename(args.instrumental)}  (noise reference)")
    else:
        print(f"  Mode         : fallback minimum-statistics (no stem provided)")
    print(f"  Strength     : {args.strength}  floor={args.floor}  band-lo={args.band_lo:.0f} Hz")
    print(f"  Output       : {os.path.basename(out_path)}\n")

    song,  sr = load(args.input)
    print(f"  {sr} Hz | {song.shape[1]}ch | {len(song)/sr:.1f}s")

    fingerprint = None
    if args.instrumental:
        instr, _ = load(args.instrumental)
        print(f"  Extracting noise fingerprint from first {args.ref_seconds:.1f}s (pre-vocal)...")
        fingerprint = noise_fingerprint_from_stem(song, instr, sr,
                                                  ref_seconds=args.ref_seconds)

    print("  Subtracting noise floor...")
    result = deshimmer(song, sr,
                       fingerprint=fingerprint,
                       strength=args.strength,
                       floor=args.floor,
                       band_lo=args.band_lo,
                       noise_window=args.noise_window)

    if not args.no_reverb:
        print(f"  Adding plate reverb (wet={args.reverb_wet}  RT60={args.reverb_time}s  pre={20}ms)...")
        result = add_reverb_tail(result, sr,
                                 band_lo=args.band_lo,
                                 rt60=args.reverb_time,
                                 wet=args.reverb_wet)

    save(out_path, result, sr)
    print(f"\n  Original untouched : {args.input}")
    print(f"  Clean copy saved   : {out_path}\n")


if __name__ == "__main__":
    main()
