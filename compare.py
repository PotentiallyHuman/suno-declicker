"""
compare — A/B audio comparator.

Plays two files in perfect sync. Press SPACE to switch between them.
Press 1-9 to jump to that tenth of the song. Q to quit.

Usage:
    python compare.py a.wav b.wav [timestamp1 timestamp2 ...]
    python compare.py a.wav b.wav --clicks 60.74 63.93 65.00
"""

import sys, os, threading, tempfile, subprocess
import numpy as np
import soundfile as sf
import sounddevice as sd


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


def fmt(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python compare.py file_a file_b")

    # parse --clicks timestamps
    clicks = []
    args = sys.argv[1:]
    if '--clicks' in args:
        ci = args.index('--clicks')
        clicks = [float(x) for x in args[ci+1:]]
        args = args[:ci]

    print(f"\n  Loading A: {os.path.basename(args[0])}...")
    a, sr_a = load(args[0])
    print(f"  Loading B: {os.path.basename(args[1])}...")
    b, sr_b = load(args[1])

    if sr_a != sr_b:
        sys.exit(f"Sample rates differ: {sr_a} vs {sr_b}")
    sr   = sr_a
    n    = min(len(a), len(b))
    a    = a[:n]; b = b[:n]
    ch   = max(a.shape[1], b.shape[1])
    if a.shape[1] < ch: a = np.tile(a, (1, ch))[:, :ch]
    if b.shape[1] < ch: b = np.tile(b, (1, ch))[:, :ch]

    label_a = os.path.basename(args[0])
    label_b = os.path.basename(args[1])
    dur     = n / sr

    # brief dip buffer to confirm switch audibly
    dip_len  = int(0.03 * sr)   # 30ms silence on switch
    state = {
        'pos':  0,
        'src':  'A',
        'dip':  0,        # samples of dip remaining
        'done': False,
    }

    def status_line():
        t = state['pos'] / sr
        src = state['src']
        label = label_a if src == 'A' else label_b
        print(f"\r  [{fmt(t)}]  ▶ {src}: {label}          ", end='', flush=True)

    def callback(outdata, frames, time_info, status):
        pos  = state['pos']
        end  = min(pos + frames, n)
        size = end - pos
        if size <= 0:
            outdata[:] = 0
            state['done'] = True
            raise sd.CallbackStop

        src = a if state['src'] == 'A' else b
        chunk = src[pos:end].copy()

        # apply dip at start of chunk if switch just happened
        dip = state['dip']
        if dip > 0:
            d = min(dip, size)
            chunk[:d] *= 0
            state['dip'] = dip - d

        outdata[:size] = chunk
        if size < frames:
            outdata[size:] = 0
            state['done'] = True
        state['pos'] = end

    def key_listener():
        import tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not state['done']:
                key = sys.stdin.read(1)

                if key == ' ':
                    state['src'] = 'B' if state['src'] == 'A' else 'A'
                    state['dip'] = dip_len
                    status_line()

                # 1-9: seek to that tenth of song
                elif key in '123456789':
                    frac = int(key) / 10
                    state['pos'] = int(frac * n)
                    state['dip'] = 0
                    status_line()

                # c + digit: jump to click number (1-based)
                elif key == 'c' and clicks:
                    print(f"\r  Jump to click #: ", end='', flush=True)
                    num = sys.stdin.read(1)
                    if num.isdigit():
                        idx = int(num) - 1
                        if 0 <= idx < len(clicks):
                            # jump 1s before click
                            state['pos'] = max(0, int((clicks[idx] - 1.0) * sr))
                            state['dip'] = 0
                            status_line()

                elif key in ('q', 'Q', '\x03'):
                    state['done'] = True
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print(f"\n  A — {label_a}")
    print(f"  B — {label_b}")
    print(f"\n  {sr} Hz | {ch}ch | {dur:.1f}s")
    if clicks:
        print(f"\n  Clicks: ", end='')
        print('  '.join(f"{i+1}:{fmt(t)}" for i, t in enumerate(clicks)))
    print(f"\n  SPACE = switch A/B")
    print(f"  1–9   = jump to 10%–90% of song")
    if clicks:
        print(f"  c+N   = jump to 1s before click N")
    print(f"  Q     = quit\n")

    status_line()

    t = threading.Thread(target=key_listener, daemon=True)
    t.start()

    with sd.OutputStream(samplerate=sr, channels=ch, callback=callback):
        while not state['done']:
            sd.sleep(50)

    print(f"\n\n  Done.\n")


if __name__ == "__main__":
    main()
