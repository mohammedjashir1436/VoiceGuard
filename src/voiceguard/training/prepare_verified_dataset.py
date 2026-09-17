"""
Prepare independently verified VoiceGuard audio for training.

Source:
    data/verified_training/
        real/
        fake/

Output:
    data/prepared_training/
        real/
        fake/

The script:
- reads only explicitly promoted verified samples
- converts audio to 16 kHz mono
- validates that usable audio exists
- saves normalized WAV files
- does NOT modify the production detector
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


DEFAULT_SOURCE = Path("data/verified_training")
DEFAULT_OUTPUT = Path("data/prepared_training")

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
}


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio as mono float32."""
    audio, sample_rate = librosa.load(
        str(path),
        sr=16000,
        mono=True,
    )

    audio = np.asarray(audio, dtype=np.float32)

    return audio, 16000


def validate_audio(audio: np.ndarray) -> tuple[bool, str]:
    """Check that an audio sample is usable for training."""

    if audio.size == 0:
        return False, "empty audio"

    duration = audio.size / 16000.0

    if duration < 0.8:
        return False, f"too short ({duration:.2f}s)"

    if not np.isfinite(audio).all():
        return False, "contains NaN or infinity"

    peak = float(np.max(np.abs(audio)))

    if peak < 1e-5:
        return False, "near silent"

    return True, f"{duration:.2f}s"


def prepare_class(
    source_dir: Path,
    output_dir: Path,
    label: str,
) -> tuple[int, int]:
    """Prepare one REAL or FAKE class."""

    source_class = source_dir / label
    output_class = output_dir / label

    output_class.mkdir(parents=True, exist_ok=True)

    if not source_class.exists():
        print(f"[{label.upper()}] source directory does not exist")
        return 0, 0

    files = sorted(
        path
        for path in source_class.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"[{label.upper()}] no supported audio files found")
        return 0, 0

    prepared = 0
    rejected = 0

    for index, source_path in enumerate(files, start=1):
        print(
            f"[{label.upper()}] "
            f"{index}/{len(files)}: {source_path.name}"
        )

        try:
            audio, sample_rate = load_audio(source_path)

            valid, reason = validate_audio(audio)

            if not valid:
                print(f"  REJECTED: {reason}")
                rejected += 1
                continue

            destination = output_class / (
                source_path.stem + ".wav"
            )

            sf.write(
                str(destination),
                audio,
                sample_rate,
                subtype="PCM_16",
            )

            print(
                f"  OK: {reason} -> {destination}"
            )

            prepared += 1

        except Exception as exc:
            print(f"  REJECTED: {exc}")
            rejected += 1

    return prepared, rejected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare verified VoiceGuard training audio."
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Verified training source directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Prepared training output directory.",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the prepared output directory before processing.",
    )

    args = parser.parse_args()

    source_dir = args.source
    output_dir = args.output

    print("=" * 70)
    print("VoiceGuard — Verified Dataset Preparation")
    print("=" * 70)
    print(f"Source : {source_dir}")
    print(f"Output : {output_dir}")
    print()

    if not source_dir.exists():
        raise SystemExit(
            f"ERROR: source directory not found: {source_dir}"
        )

    if args.clean and output_dir.exists():
        print(f"Cleaning: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_prepared = 0
    total_rejected = 0

    for label in ("real", "fake"):
        prepared, rejected = prepare_class(
            source_dir=source_dir,
            output_dir=output_dir,
            label=label,
        )

        total_prepared += prepared
        total_rejected += rejected

        print()

    real_count = len(
        list((output_dir / "real").glob("*.wav"))
    ) if (output_dir / "real").exists() else 0

    fake_count = len(
        list((output_dir / "fake").glob("*.wav"))
    ) if (output_dir / "fake").exists() else 0

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"REAL prepared : {real_count}")
    print(f"FAKE prepared : {fake_count}")
    print(f"Total prepared: {total_prepared}")
    print(f"Rejected      : {total_rejected}")
    print()

    if real_count == 0 or fake_count == 0:
        print(
            "WARNING: Both REAL and FAKE samples are required "
            "before training."
        )
    else:
        print("Dataset contains both classes.")
        print("Ready for the next training-preparation stage.")

    print("=" * 70)


if __name__ == "__main__":
    main()