#!/usr/bin/env python3
"""Demodulate 65536qam_12k_192k.wav and count the errors against symbols.csv.

    python3 demod.py [file.wav]

This is the easy way, and it only works on a clean file: it relies on there
being no carrier offset and on a symbol landing on every 16th sample, starting
with the first. A real receiver has to find all of that out for itself.

Needs numpy.
"""
import struct
import sys
import numpy as np

SPS, ALPHA, N_PREAMBLE = 16, 0.35, 64


def read_wav(path):
    raw = open(path, "rb").read()
    at, rate = 12, None
    while at + 8 <= len(raw):
        name, size = raw[at:at + 4], struct.unpack("<I", raw[at + 4:at + 8])[0]
        if name == b"fmt ":
            rate = struct.unpack("<I", raw[at + 12:at + 16])[0]
        if name == b"data":
            iq = np.frombuffer(raw[at + 8:at + 8 + size], "<i2").astype(float)
            return rate, iq[0::2] + 1j * iq[1::2]
        at += 8 + size + (size & 1)
    raise SystemExit("no data chunk in " + path)


def rrc(n, sps, alpha):
    f = np.abs(np.fft.fftfreq(n)) * sps
    lo, hi = (1 - alpha) / 2, (1 + alpha) / 2
    h = np.zeros(n)
    h[f <= lo] = 1.0
    edge = (f > lo) & (f < hi)
    h[edge] = np.sqrt(0.5 * (1 + np.cos(np.pi / alpha * (f[edge] - lo))))
    return h


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "65536qam_12k_192k.wav"
    rate, x = read_wav(path)

    key = [line.strip().split(",") for line in open("symbols.csv")][1:]
    sent = np.array([complex(int(r[1]), int(r[2])) for r in key])
    words = np.array([int(r[3], 16) for r in key[N_PREAMBLE:]])

    # Matched filter. The file is a loop, so the filter wraps round its ends.
    y = np.fft.ifft(np.fft.fft(x) * rrc(len(x), SPS, ALPHA))
    s = y[::SPS]

    # The preamble is known, so it gives the gain and the phase, and with the
    # phase which of the four ways round the constellation is.
    p = sent[:N_PREAMBLE]
    s = s / (np.vdot(p, s[:N_PREAMBLE]) / np.vdot(p, p))

    # Nearest point: levels are the odd numbers -255 ... +255.
    def index(v):
        return np.clip(np.rint((v + 255) / 2), 0, 255).astype(int)
    ii, qi = index(s.real), index(s.imag)
    decided = (2 * ii - 255) + 1j * (2 * qi - 255)

    # Back to bits: Gray code per axis, I in the top eight bits.
    got = (((ii ^ (ii >> 1)) << 8) | (qi ^ (qi >> 1)))[N_PREAMBLE:]

    wrong = int(np.sum(decided != sent))
    bits = int(sum(bin(int(v)).count("1") for v in (got ^ words)))
    err = s - sent
    mer = 10 * np.log10(np.sum(np.abs(sent) ** 2) / np.sum(np.abs(err) ** 2))

    print("%s: %d samples at %d S/s, %d symbols" % (path, len(x), rate, len(s)))
    print("symbol errors: %d of %d" % (wrong, len(s)))
    print("bit errors:    %d of %d" % (bits, 16 * len(words)))
    print("different points in the payload: %d" % len(set(got.tolist())))
    print("MER: %.1f dB" % mer)


if __name__ == "__main__":
    main()
