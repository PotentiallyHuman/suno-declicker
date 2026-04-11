#!/usr/bin/env python3
"""
Real-time looping spectrogram viewer — full screen, pygame display.

Usage:
    python3 spectrogram_loop.py audio.mp3 start_sec end_sec
"""

import sys, os, tempfile, subprocess
import numpy as np
import soundfile as sf
from scipy.signal import stft as scipy_stft
from scipy.ndimage import zoom
import pygame
import sounddevice as sd


def load_wav(path, sr=44100):
    if path.lower().endswith('.mp3'):
        tmp = tempfile.mktemp(suffix='.wav')
        subprocess.run(['ffmpeg','-y','-i',path,'-ar',str(sr),tmp],
                       capture_output=True, check=True)
        data, _ = sf.read(tmp)
        os.unlink(tmp)
    else:
        data, _ = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


if len(sys.argv) < 4:
    print(__doc__); sys.exit(1)

audio_path = sys.argv[1]
t0, t1     = float(sys.argv[2]), float(sys.argv[3])

print(f"Loading {os.path.basename(audio_path)}  [{t0}s – {t1}s]...")
full, sr = load_wav(audio_path)
phrase   = full[int(t0*sr):int(t1*sr)].copy()
n_phrase = len(phrase)
dur      = n_phrase / sr

# ── Spectrogram ───────────────────────────────────────────────────────────────
print("Computing spectrogram...")
freqs, times, Zxx = scipy_stft(phrase, fs=sr, nperseg=512, noverlap=480)
mag_db  = 20 * np.log10(np.abs(Zxx) + 1e-8)
vmax    = mag_db.max()
vmin    = vmax - 60
fmask   = freqs <= 12000
spec    = mag_db[fmask]               # shape: (freq_bins, time_bins)

# Normalise to 0–255
spec_norm = np.clip((spec - vmin) / (vmax - vmin), 0, 1)
spec_norm = (spec_norm * 255).astype(np.uint8)
spec_norm = spec_norm[::-1]           # flip: low freq at bottom

# ── Pygame init ───────────────────────────────────────────────────────────────
pygame.init()
info   = pygame.display.Info()
W, H   = info.current_w, info.current_h
screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
pygame.display.set_caption("Spectrogram")

# Resize spectrogram to screen size
print(f"Rendering to {W}×{H}...")
freq_bins, time_bins = spec_norm.shape
zy = H / freq_bins
zx = W / time_bins
spec_scaled = zoom(spec_norm, (zy, zx), order=1)

# Apply inferno colormap manually (R,G,B)
def inferno(v):
    # Simplified inferno: black→purple→red→orange→yellow→white
    v = np.clip(v / 255.0, 0, 1)
    r = np.clip(v * 2.5,          0, 1)
    g = np.clip((v - 0.4) * 2.5,  0, 1)
    b = np.clip(1 - v * 3,        0, 1) + np.clip((v - 0.7) * 3, 0, 1)
    b = np.clip(b, 0, 1)
    return (r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)

rgb = np.zeros((H, W, 3), dtype=np.uint8)
rgb[:,:,0], rgb[:,:,1], rgb[:,:,2] = inferno(spec_scaled)

# Build pygame surface from RGB array
surf_spec = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))

# ── Audio playback ────────────────────────────────────────────────────────────
state = {'pos': 0}

def audio_callback(outdata, frames, time_info, status):
    pos = state['pos']
    end = pos + frames
    if end <= n_phrase:
        chunk = phrase[pos:end]
        state['pos'] = end
    else:
        rem   = n_phrase - pos
        chunk = np.concatenate([phrase[pos:], phrase[:frames - rem]])
        state['pos'] = frames - rem
    outdata[:, 0] = chunk
    if outdata.shape[1] > 1:
        outdata[:, 1] = chunk

stream = sd.OutputStream(samplerate=sr, channels=2,
                          blocksize=512, callback=audio_callback)
stream.start()

# ── Main loop ─────────────────────────────────────────────────────────────────
clock  = pygame.time.Clock()
font   = pygame.font.SysFont('monospace', 16)
CYAN   = (0, 220, 220)
WHITE  = (255, 255, 255)

print("Running — press Q or Esc to quit.")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

    screen.blit(surf_spec, (0, 0))

    # Cursor: x pixel position proportional to playback position
    x_px = int((state['pos'] / n_phrase) * W)
    x_px = max(0, min(W - 1, x_px))
    pygame.draw.line(screen, CYAN, (x_px, 0), (x_px, H), 2)

    # Time label
    t_now = state['pos'] / sr
    label = font.render(f"{t0 + t_now:.3f}s", True, WHITE)
    screen.blit(label, (x_px + 4, 10))

    pygame.display.flip()
    clock.tick(60)

stream.stop()
pygame.quit()
