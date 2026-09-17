"""Spectral (in-signal) watermarking for synthesized audio.

This is the robust provenance layer: a low-amplitude sinusoidal tone at an
inaudible frequency (18 kHz), amplitude-modulated by a PRNG sequence seeded
from the watermark_id; detection uses keyed correlation.

This is distinct from — and complementary to — the cryptographic provenance
layer in ``c2pa_sign``, which embeds a signed C2PA manifest into the file.

The synthesis API applies both:
- C2PA manifest for cryptographic provenance
- spectral watermark for in-signal provenance
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from math import gcd

import numpy as np

logger = logging.getLogger(__name__)

# An 18 kHz carrier is only inaudible if the sample rate can carry it
# comfortably above 16 kHz.
MIN_INAUDIBLE_CARRIER_HZ = 16000.0


def _clamp_carrier(carrier_hz: float, sr: int) -> float:
    """Clamp carrier below Nyquist."""
    nyquist = sr / 2
    return min(carrier_hz, nyquist - 100.0)


def ensure_carrier_sr(
    audio: np.ndarray,
    sr: int,
    carrier_hz: float = 18000.0,
    min_carrier_hz: float = MIN_INAUDIBLE_CARRIER_HZ,
) -> tuple[np.ndarray, int]:
    """Resample audio up if the sample rate is too low for the carrier.

    Kokoro normally produces 24 kHz audio. An 18 kHz carrier cannot be
    represented safely at 24 kHz, so it is resampled to 48 kHz before
    watermark embedding.
    """
    if sr / 2 - 100.0 >= min_carrier_hz:
        return np.asarray(audio, dtype=np.float32), sr

    target_sr = max(int(2 * (carrier_hz + 2000.0)), 48000)

    from scipy.signal import resample_poly

    g = gcd(target_sr, sr)

    out = resample_poly(
        np.asarray(audio, dtype=np.float32),
        target_sr // g,
        sr // g,
    )

    return out.astype(np.float32), target_sr


def _prng_sequence(seed: str, length: int) -> np.ndarray:
    """Create a reproducible +/-1 spreading code from a string seed."""
    rng = np.random.default_rng(
        int(
            hashlib.sha256(seed.encode()).hexdigest(),
            16,
        )
        % (2**32)
    )

    return rng.choice(
        [-1.0, 1.0],
        size=length,
    ).astype(np.float32)


def embed(
    audio: np.ndarray,
    sr: int = 22050,
    watermark_id: str | None = None,
    amplitude: float = 0.002,
    carrier_hz: float = 18000.0,
) -> tuple[np.ndarray, str]:
    """Embed a spectral watermark into audio.

    Args:
        audio: 1-D float32 PCM samples.
        sr: Sample rate.
        watermark_id: Unique watermark ID. Generated automatically if None.
        amplitude: Watermark amplitude.
        carrier_hz: Carrier frequency in Hz.

    Returns:
        Tuple containing:
            watermarked audio
            watermark ID
    """
    if watermark_id is None:
        watermark_id = str(uuid.uuid4())

    carrier_hz = _clamp_carrier(
        carrier_hz,
        sr,
    )

    if carrier_hz < MIN_INAUDIBLE_CARRIER_HZ:
        logger.warning(
            "Watermark carrier clamped to %.0f Hz at sr=%d Hz — "
            "this is AUDIBLE; call ensure_carrier_sr() before embed().",
            carrier_hz,
            sr,
        )

    t = np.arange(
        len(audio),
        dtype=np.float32,
    ) / sr

    carrier = np.sin(
        2 * np.pi * carrier_hz * t
    )

    code = _prng_sequence(
        watermark_id,
        len(audio),
    )

    watermark = (
        amplitude
        * carrier
        * code
    )

    watermarked = (
        np.asarray(audio, dtype=np.float32)
        + watermark
    ).clip(
        -1.0,
        1.0,
    )

    return watermarked, watermark_id


def detect(
    audio: np.ndarray,
    sr: int = 22050,
    watermark_id: str = "",
    carrier_hz: float = 18000.0,
    threshold: float = 0.001,
) -> tuple[bool, float]:
    """Detect a watermark for the supplied watermark ID.

    The old detector normalized against the complete speech waveform.
    Because speech contains much more energy than the low-amplitude watermark,
    this diluted the correlation and produced values around 0.0002 even when
    the watermark was actually embedded.

    This detector instead estimates the amplitude of the keyed watermark
    directly by correlating the audio with the same carrier/spreading code
    used during embedding.

    Returns:
        (detected, score)

        ``detected`` is True when the estimated watermark amplitude reaches
        the configured threshold.

        ``score`` is the estimated watermark amplitude.
    """
    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    carrier_hz = _clamp_carrier(
        carrier_hz,
        sr,
    )

    # Recreate the exact carrier used during embedding.
    t = np.arange(
        len(audio),
        dtype=np.float32,
    ) / sr

    carrier = np.sin(
        2 * np.pi * carrier_hz * t
    )

    # Recreate the exact keyed spreading sequence.
    code = _prng_sequence(
        watermark_id,
        len(audio),
    )

    # Same reference signal used by embed().
    reference = carrier * code

    # Estimate the amplitude of the reference signal contained in audio.
    #
    # If:
    #
    #     audio = speech + 0.003 * reference
    #
    # then:
    #
    #     dot(audio, reference) / dot(reference, reference)
    #
    # estimates approximately 0.003.
    corr = float(
        np.dot(
            audio,
            reference,
        )
    )

    reference_energy = float(
        np.dot(
            reference,
            reference,
        )
        + 1e-12
    )

    estimated_amplitude = (
        corr / reference_energy
    )

    # Watermark polarity is not important for provenance detection.
    score = float(
        abs(estimated_amplitude)
    )

    detected = score >= threshold

    return detected, score