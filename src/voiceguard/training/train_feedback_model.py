"""
VoiceGuard - Verified Feedback Model Training

Trains a candidate v2 Hugging Face audio-classification model
from verified REAL/FAKE recordings.

IMPORTANT:
- Does NOT modify the production model.
- Reads only data/verified_training/{real,fake}
- Saves candidate model under data/model_candidates/
- Uses librosa for audio decoding so TorchCodec is not required.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_linear_schedule_with_warmup,
)


MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"

SAMPLE_RATE = 16000
MAX_SECONDS = 5.0
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_audio(path: Path) -> torch.Tensor:
    """
    Load audio as mono 16 kHz float32.

    Uses librosa instead of torchaudio so the training script
    does not depend on TorchCodec.
    """

    audio, _ = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    max_samples = int(
        SAMPLE_RATE * MAX_SECONDS
    )

    if audio.size > max_samples:
        audio = audio[:max_samples]

    if audio.size == 0:
        raise ValueError(
            f"Audio file is empty: {path}"
        )

    return torch.from_numpy(audio)


class VerifiedAudioDataset(Dataset):
    def __init__(
        self,
        samples,
        feature_extractor,
    ):
        self.samples = samples
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        audio = load_audio(path)

        inputs = self.feature_extractor(
            audio.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        input_values = inputs["input_values"].squeeze(0)

        return {
            "input_values": input_values,
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


def collate_fn(batch):
    input_values = [
        item["input_values"]
        for item in batch
    ]

    labels = torch.stack(
        [
            item["labels"]
            for item in batch
        ]
    )

    max_len = max(
        audio.shape[0]
        for audio in input_values
    )

    padded = torch.zeros(
        len(input_values),
        max_len,
        dtype=torch.float32,
    )

    attention_mask = torch.zeros(
        len(input_values),
        max_len,
        dtype=torch.long,
    )

    for i, audio in enumerate(input_values):
        length = audio.shape[0]

        padded[i, :length] = audio
        attention_mask[i, :length] = 1

    return {
        "input_values": padded,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collect_samples(data_dir: Path):
    samples = []

    label_directories = (
        ("real", 0),
        ("fake", 1),
    )

    supported_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    }

    for label_name, label in label_directories:

        directory = data_dir / label_name

        if not directory.exists():
            raise FileNotFoundError(
                f"Missing dataset directory: {directory}"
            )

        for path in sorted(directory.iterdir()):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in supported_extensions
            ):
                continue

            samples.append(
                (path, label)
            )

    return samples


def split_samples(
    samples,
    val_ratio=0.2,
):
    real = [
        item
        for item in samples
        if item[1] == 0
    ]

    fake = [
        item
        for item in samples
        if item[1] == 1
    ]

    random.shuffle(real)
    random.shuffle(fake)

    real_val_count = max(
        1,
        int(len(real) * val_ratio),
    )

    fake_val_count = max(
        1,
        int(len(fake) * val_ratio),
    )

    val_samples = (
        real[:real_val_count]
        + fake[:fake_val_count]
    )

    train_samples = (
        real[real_val_count:]
        + fake[fake_val_count:]
    )

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    correct = 0
    total = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    losses = []

    for batch in loader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        if outputs.loss is not None:
            losses.append(
                float(outputs.loss.detach().cpu())
            )

        predictions = (
            outputs.logits.argmax(dim=-1)
        )

        labels = batch["labels"]

        correct += int(
            (predictions == labels).sum()
        )

        total += labels.numel()

        tp += int(
            (
                (predictions == 1)
                & (labels == 1)
            ).sum()
        )

        tn += int(
            (
                (predictions == 0)
                & (labels == 0)
            ).sum()
        )

        fp += int(
            (
                (predictions == 1)
                & (labels == 0)
            ).sum()
        )

        fn += int(
            (
                (predictions == 0)
                & (labels == 1)
            ).sum()
        )

    accuracy = (
        correct / max(total, 1)
    )

    precision = (
        tp / max(tp + fp, 1)
    )

    recall = (
        tp / max(tp + fn, 1)
    )

    f1_denominator = (
        precision + recall
    )

    if f1_denominator > 0:
        f1 = (
            2
            * precision
            * recall
            / f1_denominator
        )
    else:
        f1 = 0.0

    return {
        "loss": (
            float(np.mean(losses))
            if losses
            else 0.0
        ),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fake_recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a VoiceGuard candidate model "
            "from verified feedback data."
        )
    )

    parser.add_argument(
        "--data",
        default="data/verified_training",
    )

    parser.add_argument(
        "--output",
        default=(
            "data/model_candidates/"
            "wav2vec2_feedback_v2"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    set_seed()

    data_dir = Path(args.data)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("VoiceGuard Feedback Training")
    print("=" * 70)

    print(
        f"Base model : {MODEL_ID}"
    )

    print(
        f"Dataset    : {data_dir}"
    )

    print(
        f"Output     : {output_dir}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device     : {device}"
    )

    # ---------------------------------------------------------
    # Load the exact Hugging Face production model architecture
    # as a separate candidate copy.
    # ---------------------------------------------------------

    print(
        "\nLoading feature extractor..."
    )

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_ID
        )
    )

    print(
        "Loading model..."
    )

    model = (
        AutoModelForAudioClassification
        .from_pretrained(
            MODEL_ID
        )
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Labels:",
        model.config.id2label,
    )

    # Confirm expected REAL/FAKE mapping.
    label_map = {
        int(key): str(value).lower()
        for key, value
        in model.config.id2label.items()
    }

    if label_map != {
        0: "real",
        1: "fake",
    }:
        raise RuntimeError(
            "Unexpected model label mapping. "
            f"Expected {{0: 'real', 1: 'fake'}}, "
            f"got {label_map}"
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    samples = collect_samples(
        data_dir
    )

    real_count = sum(
        1
        for _, label in samples
        if label == 0
    )

    fake_count = sum(
        1
        for _, label in samples
        if label == 1
    )

    print("\nDataset:")
    print(
        f"REAL : {real_count}"
    )
    print(
        f"FAKE : {fake_count}"
    )
    print(
        f"TOTAL: {len(samples)}"
    )

    if real_count < 2:
        raise RuntimeError(
            "Need at least 2 REAL samples."
        )

    if fake_count < 2:
        raise RuntimeError(
            "Need at least 2 FAKE samples."
        )

    train_samples, val_samples = (
        split_samples(
            samples,
            args.val_ratio,
        )
    )

    print("\nSplit:")
    print(
        f"Train: {len(train_samples)}"
    )
    print(
        f"Val  : {len(val_samples)}"
    )

    train_real = sum(
        1
        for _, label in train_samples
        if label == 0
    )

    train_fake = sum(
        1
        for _, label in train_samples
        if label == 1
    )

    val_real = sum(
        1
        for _, label in val_samples
        if label == 0
    )

    val_fake = sum(
        1
        for _, label in val_samples
        if label == 1
    )

    print(
        f"Train REAL: {train_real}"
    )
    print(
        f"Train FAKE: {train_fake}"
    )
    print(
        f"Val REAL  : {val_real}"
    )
    print(
        f"Val FAKE  : {val_fake}"
    )

    # ---------------------------------------------------------
    # Datasets / loaders
    # ---------------------------------------------------------

    train_dataset = VerifiedAudioDataset(
        train_samples,
        feature_extractor,
    )

    val_dataset = VerifiedAudioDataset(
        val_samples,
        feature_extractor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = (
        len(train_loader)
        * args.epochs
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(
                1,
                total_steps // 10,
            ),
            num_training_steps=total_steps,
        )
    )

    best_f1 = -1.0
    history = []

    for epoch in range(args.epochs):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"EPOCH {epoch + 1}/{args.epochs}"
        )

        print(
            "=" * 70
        )

        model.train()

        running_loss = 0.0

        for step, batch in enumerate(
            train_loader
        ):

            batch = {
                key: value.to(device)
                for key, value
                in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                **batch
            )

            loss = outputs.loss

            if loss is None:
                raise RuntimeError(
                    "Model did not return a training loss."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            loss_value = float(
                loss.detach().cpu()
            )

            running_loss += loss_value

            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"Step {step + 1}/{len(train_loader)} "
                f"Loss={loss_value:.4f}",
                flush=True,
            )

        train_loss = (
            running_loss
            / max(len(train_loader), 1)
        )

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        metrics["epoch"] = (
            epoch + 1
        )

        metrics["train_loss"] = (
            train_loss
        )

        history.append(metrics)

        print("\nValidation:")
        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{metrics['loss']:.4f}"
        )

        print(
            f"Accuracy   : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision  : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall     : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"FAKE Recall: "
            f"{metrics['fake_recall']:.4f}"
        )

        print(
            f"F1         : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"TP={metrics['tp']} "
            f"TN={metrics['tn']} "
            f"FP={metrics['fp']} "
            f"FN={metrics['fn']}"
        )

        # -----------------------------------------------------
        # Save best candidate
        # -----------------------------------------------------

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            print(
                "\nSaving best candidate model..."
            )

            model.save_pretrained(
                output_dir
            )

            feature_extractor.save_pretrained(
                output_dir
            )

            with open(
                output_dir / "metrics.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    metrics,
                    file,
                    indent=2,
                )

    # ---------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------

    with open(
        output_dir / "history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )

    # Save dataset information.
    dataset_info = {
        "base_model": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "max_seconds": MAX_SECONDS,
        "seed": SEED,
        "total_samples": len(samples),
        "real_samples": real_count,
        "fake_samples": fake_count,
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "train_real": train_real,
        "train_fake": train_fake,
        "validation_real": val_real,
        "validation_fake": val_fake,
        "best_validation_f1": best_f1,
    }

    with open(
        output_dir / "dataset_info.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dataset_info,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Candidate model saved to:\n"
        f"{output_dir}"
    )

    print(
        f"\nBest validation F1: "
        f"{best_f1:.4f}"
    )

    print(
        "\nIMPORTANT: "
        "This candidate has NOT replaced "
        "the production VoiceGuard detector."
    )


if __name__ == "__main__":
    main()