#!/usr/bin/env python3
"""Make 65536qam_12k_192k.wav and symbols.csv.

One seamless loop: a 64-symbol QPSK preamble, then every one of the 65536
constellation points exactly once. README.md describes the format.

    python3 generate.py

Needs numpy.
"""
import struct
import numpy as np

FS = 192_000              # samples per second
SPS = 16                  # samples per symbol, so 12000 symbols per second
ALPHA = 0.35              # root raised cosine roll-off
N_PREAMBLE = 64
PREAMBLE_LEVEL = 147      # the four preamble points are (+-147, +-147)
CENTRE_HZ = 30_000_000    # a label for SDR software; this file never went through a radio
PEAK = 10 ** (-1 / 20)    # largest I or Q sample: 1 dB under full scale

WAV = "65536qam_12k_192k.wav"
KEY = "symbols.csv"


def payload_words():
    """All 65536 sixteen-bit words, each exactly once, in scrambled order.

    A 16-bit shift register (x^16 + x^14 + x^13 + x^11 + 1, start 0xACE1) is
    clocked 16 times per symbol and its contents are the word. That visits
    every word but zero, so zero is sent last.
    """
    s, words = 0xACE1, []
    for _ in range(65535):
        for _ in range(16):
            bit = (s ^ (s >> 2) ^ (s >> 3) ^ (s >> 5)) & 1
            s = (s >> 1) | (bit << 15)
        words.append(s)
    words.append(0)
    assert sorted(words) == list(range(65536))
    return words


def preamble_bits(n):
    """A 9-bit shift register (period 511), started with all ones."""
    s, bits = 0x1FF, []
    for _ in range(n):
        bit = (s ^ (s >> 4)) & 1
        s = (s >> 1) | (bit << 8)
        bits.append(bit)
    return bits


def level(bits):
    """Eight Gray-coded bits to one of the 256 levels -255, -253 ... +255."""
    n = bits
    n ^= n >> 1
    n ^= n >> 2
    n ^= n >> 4
    return 2 * n - 255


def rrc(n, sps, alpha):
    """Root raised cosine, as a frequency response over n samples."""
    f = np.abs(np.fft.fftfreq(n)) * sps          # in symbol rates
    lo, hi = (1 - alpha) / 2, (1 + alpha) / 2
    h = np.zeros(n)
    h[f <= lo] = 1.0
    edge = (f > lo) & (f < hi)
    h[edge] = np.sqrt(0.5 * (1 + np.cos(np.pi / alpha * (f[edge] - lo))))
    return h


def main():
    bits = preamble_bits(2 * N_PREAMBLE)
    pre = [(PREAMBLE_LEVEL * (1 - 2 * bits[2 * k]), PREAMBLE_LEVEL * (1 - 2 * bits[2 * k + 1]))
           for k in range(N_PREAMBLE)]
    words = payload_words()
    pay = [(level(w >> 8), level(w & 0xFF)) for w in words]

    with open(KEY, "w") as f:
        f.write("n,I,Q,word\n")
        for n, (i, q) in enumerate(pre):
            f.write("%d,%d,%d,\n" % (n, i, q))
        for n, ((i, q), w) in enumerate(zip(pay, words)):
            f.write("%d,%d,%d,%04X\n" % (N_PREAMBLE + n, i, q, w))

    sym = np.array([complex(i, q) for i, q in pre + pay])
    # Shaped over the whole loop at once: the tails of the last symbols wrap
    # round into the start, so the file joins up with itself exactly.
    x = np.zeros(len(sym) * SPS, complex)
    x[::SPS] = sym
    x = np.fft.ifft(np.fft.fft(x) * rrc(len(x), SPS, ALPHA))
    top = max(np.abs(x.real).max(), np.abs(x.imag).max())
    iq = np.empty(2 * len(x), "<i2")
    iq[0::2] = np.rint(x.real / top * PEAK * 32767)
    iq[1::2] = np.rint(x.imag / top * PEAK * 32767)
    data = iq.tobytes()

    # 16-bit stereo WAV, I on the left and Q on the right, with the "auxi"
    # chunk that SpectraFlux, SpectraVue, CuteSDR and others read for the
    # centre frequency.
    start = struct.pack("<8H", 2026, 11, 1, 23, 0, 0, 0, 0)
    ms = round(1000 * len(x) / FS)
    stop = struct.pack("<8H", 2026, 11, 1, 23, 0, 0, ms // 1000, ms % 1000)
    auxi = start + stop + struct.pack("<4I", CENTRE_HZ, CENTRE_HZ, 0, FS) + bytes(24)
    fmt = struct.pack("<HHIIHH", 1, 2, FS, FS * 4, 4, 16)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"auxi" + struct.pack("<I", len(auxi)) + auxi
            + b"data" + struct.pack("<I", len(data)) + data)
    with open(WAV, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)

    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    print("%s: %d symbols, %d samples, %.3f s, %.2f MB" % (
        WAV, len(sym), len(x), len(x) / FS, (len(body) + 8) / 1e6))
    print("peak to average %.2f dB, rms %.1f dB under full scale" % (
        20 * np.log10(np.abs(x).max() / rms), -20 * np.log10(rms / top * PEAK / np.sqrt(2))))
    print("%s: %d rows" % (KEY, len(sym)))


if __name__ == "__main__":
    main()
