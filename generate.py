#!/usr/bin/env python3
"""Make the three signal files and symbols.csv.

    65536qam_12k_192k.wav   level 1: clean
    65536qam_level2.wav     level 2: carrier, timing and phase are off, but fixed
    65536qam_level3.wav     level 3: they move, and there is an echo and noise

All three carry the same symbols: a 64-symbol QPSK preamble, then every one
of the 65536 constellation points exactly once. Each file is one seamless
loop. README.md describes the format.

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
CENTRE_HZ = 30_000_000    # a label for SDR software; these files never went through a radio
PEAK = 10 ** (-1 / 20)    # largest I or Q sample: 1 dB under full scale

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


def delayed(x, samples):
    """x, later by a number of samples that need not be whole, around the loop."""
    shift = np.exp(-2j * np.pi * np.fft.fftfreq(len(x)) * samples)
    return np.fft.ifft(np.fft.fft(x) * shift)


def turns(n, cycles, start_deg=0.0):
    """A phase in radians that goes round `cycles` whole times in one loop."""
    return 2 * np.pi * cycles * np.arange(n) / n + np.radians(start_deg)


def level2(x):
    """Everything is off, and nothing moves."""
    x = delayed(x, 0.37 * SPS)                     # symbols no longer sit on samples
    return x * np.exp(1j * turns(len(x), 203, 213))   # +37.13 Hz, started at 213 degrees


def level3(x, symbol_power):
    """What a real link does to it, kept small enough to be undone."""
    rng = np.random.default_rng(3)
    # A second path, 56 dB down and 1.6 symbols late.
    x = x + 0.0015 * np.exp(1j * np.radians(70)) * delayed(x, 1.6 * SPS)
    x = delayed(x, 0.61 * SPS)                     # a different timing offset
    n = len(x)
    # Carrier 58.17 Hz low, with slow phase wander on top (about half a degree rms).
    phase = turns(n, -318, 131)
    for cycles, degrees in ((3, 0.6), (7, 0.4), (16, 0.25), (29, 0.15)):
        phase += np.radians(degrees) * np.sin(turns(n, cycles, rng.uniform(0, 360)))
    # The level drifts by a tenth of a dB either way, once per loop.
    gain = 10 ** (0.1 * np.sin(turns(n, 1, 45)) / 20)
    x = x * gain * np.exp(1j * phase)
    # Noise for 60 dB MER after a matched filter.
    sigma = np.sqrt(symbol_power / SPS / 10 ** (60 / 10) / 2)
    return x + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def write_wav(path, x):
    """16-bit stereo WAV, I on the left and Q on the right, with the "auxi"
    chunk that SpectraFlux, SpectraVue, CuteSDR and others read for the
    centre frequency."""
    top = max(np.abs(x.real).max(), np.abs(x.imag).max())
    iq = np.empty(2 * len(x), "<i2")
    iq[0::2] = np.rint(x.real / top * PEAK * 32767)
    iq[1::2] = np.rint(x.imag / top * PEAK * 32767)
    data = iq.tobytes()
    ms = round(1000 * len(x) / FS)
    start = struct.pack("<8H", 2026, 11, 1, 23, 0, 0, 0, 0)
    stop = struct.pack("<8H", 2026, 11, 1, 23, 0, 0, ms // 1000, ms % 1000)
    auxi = start + stop + struct.pack("<4I", CENTRE_HZ, CENTRE_HZ, 0, FS) + bytes(24)
    fmt = struct.pack("<HHIIHH", 1, 2, FS, FS * 4, 4, 16)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"auxi" + struct.pack("<I", len(auxi)) + auxi
            + b"data" + struct.pack("<I", len(data)) + data)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    print("%s: %d samples, %.3f s, %.2f MB, peak to average %.2f dB" % (
        path, len(x), len(x) / FS, (len(body) + 8) / 1e6, 20 * np.log10(np.abs(x).max() / rms)))


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
    print("%s: %d symbols" % (KEY, len(pre) + len(pay)))

    sym = np.array([complex(i, q) for i, q in pre + pay])
    # Shaped over the whole loop at once: the tails of the last symbols wrap
    # round into the start, so the file joins up with itself exactly.
    x = np.zeros(len(sym) * SPS, complex)
    x[::SPS] = sym
    x = np.fft.ifft(np.fft.fft(x) * rrc(len(x), SPS, ALPHA))

    write_wav("65536qam_12k_192k.wav", x)
    write_wav("65536qam_level2.wav", level2(x))
    write_wav("65536qam_level3.wav", level3(x, np.mean(np.abs(sym) ** 2)))


if __name__ == "__main__":
    main()
