"""Zero-shot voice-cloning engines.

IndexTTS-2 runs on the Windows host through its dedicated HTTP service because
the VoiceGuard API itself runs inside Docker/Linux.

XTTS continues to use the existing subprocess/warm-server implementation.
The API layer remains responsible for authorization, consent, quota,
watermarking, and C2PA provenance.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib import error, request

import numpy as np
import soundfile as sf

from voiceguard.synthesis.base import SynthEngine


SYNTH_HOME = Path(
    os.environ.get(
        "VG_SYNTH_HOME",
        str(Path.home() / ".voiceguard" / "synth"),
    )
)

_WORKER = Path(__file__).with_name("clone_worker.py")
_MIN_REF_SEC = 3.0

# Existing optional warm-server ports.
_WARM_PORTS = {
    "xtts": 8801,
}

# IndexTTS-2 is installed on the Windows host and exposed through this service.
# Docker Desktop provides host.docker.internal for reaching the host.
INDEXTTS2_SERVICE_URL = os.environ.get(
    "VG_INDEXTTS2_URL",
    "http://host.docker.internal:8802",
).rstrip("/")


class CloneEngine(SynthEngine):
    """Generic subprocess/warm-server cloning engine used by XTTS."""

    requires_reference = True
    worker_key: str = ""

    def _warm_generate(
        self,
        text: str,
        ref: Path,
        out_path: Path,
        language: str,
    ) -> bool:
        """Try the existing persistent warm server."""

        import urllib.request

        port = _WARM_PORTS.get(self.name)
        if not port:
            return False

        body = json.dumps(
            {
                "text": text,
                "ref": str(ref),
                "out": str(out_path),
                "language": language,
            }
        ).encode()

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
                if resp.status != 200:
                    return False
        except Exception:
            return False

        return out_path.exists() and out_path.stat().st_size > 0

    def _venv_python(self) -> Path:
        return SYNTH_HOME / self.name / "venv" / "bin" / "python"

    def _weights_dir(self) -> Path:
        env = os.environ.get(f"VG_{self.name.upper()}_WEIGHTS")
        return Path(env) if env else SYNTH_HOME / self.name / "weights"

    def is_available(self) -> bool:
        return self._venv_python().exists() and self._weights_dir().exists()

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        reference_wav: str | Path | None = None,
        language: str = "en",
    ) -> tuple[np.ndarray, int]:
        if reference_wav is None:
            raise ValueError(
                f"{self.label} requires a reference audio clip"
            )

        ref = Path(reference_wav)

        if not ref.exists():
            raise ValueError(
                f"Reference audio file does not exist: {ref}"
            )

        dur = sf.info(str(ref)).duration

        if dur < _MIN_REF_SEC:
            raise ValueError(
                f"Reference audio too short ({dur:.1f}s); "
                f"need >= {_MIN_REF_SEC}s"
            )

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tf:
            out_path = Path(tf.name)

        try:
            # Existing warm-server path for engines such as XTTS.
            if self._warm_generate(
                text,
                ref,
                out_path,
                language,
            ):
                audio, sr = sf.read(
                    str(out_path),
                    dtype="float32",
                    always_2d=False,
                )

                if getattr(audio, "ndim", 1) == 2:
                    audio = audio.mean(axis=1)

                return np.asarray(audio, dtype=np.float32), int(sr)

            # Existing subprocess fallback.
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = (
                "/tmp/nvml_fix:"
                + env.get("LD_LIBRARY_PATH", "")
            )
            env["COQUI_TOS_AGREED"] = "1"

            cmd = [
                str(self._venv_python()),
                str(_WORKER),
                "--engine",
                self.worker_key,
                "--text",
                text,
                "--ref",
                str(ref),
                "--out",
                str(out_path),
                "--weights",
                str(self._weights_dir()),
                "--language",
                language,
            ]

            proc = subprocess.run(  # noqa: S603
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            if (
                proc.returncode != 0
                or not out_path.exists()
                or out_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    f"{self.label} worker failed "
                    f"(rc={proc.returncode}): "
                    f"{proc.stderr[-500:]}"
                )

            audio, sr = sf.read(
                str(out_path),
                dtype="float32",
                always_2d=False,
            )

            if getattr(audio, "ndim", 1) == 2:
                audio = audio.mean(axis=1)

            return np.asarray(audio, dtype=np.float32), int(sr)

        finally:
            out_path.unlink(missing_ok=True)


class IndexTTS2Engine(CloneEngine):
    """IndexTTS-2 client for the Windows-hosted synthesis service."""

    name = "indextts2"
    label = "IndexTTS-2 (zero-shot voice cloning)"
    worker_key = "indextts2"
    languages = ["en"]
    description = (
        "High-quality zero-shot cloning from a short reference clip."
    )

    def is_available(self) -> bool:
        """Check whether the host IndexTTS-2 service is reachable."""

        try:
            req = request.Request(
                f"{INDEXTTS2_SERVICE_URL}/health",
                method="GET",
            )

            with request.urlopen(req, timeout=5) as resp:  # noqa: S310
                if resp.status != 200:
                    return False

                data = json.loads(
                    resp.read().decode("utf-8")
                )

            return (
                data.get("status") == "ok"
                and data.get("engine") == "indextts2"
                and data.get("model_exists") is True
            )

        except Exception as exc:
            print(
                "IndexTTS2 availability check failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        reference_wav: str | Path | None = None,
        language: str = "en",
    ) -> tuple[np.ndarray, int]:
        """Send text + reference audio to the Windows IndexTTS-2 service."""

        if reference_wav is None:
            raise ValueError(
                f"{self.label} requires a reference audio clip"
            )

        ref = Path(reference_wav)

        if not ref.exists():
            raise ValueError(
                f"Reference audio file does not exist: {ref}"
            )

        dur = sf.info(str(ref)).duration

        if dur < _MIN_REF_SEC:
            raise ValueError(
                f"Reference audio too short ({dur:.1f}s); "
                f"need >= {_MIN_REF_SEC}s"
            )

        # Build multipart/form-data without adding another dependency
        # to the Docker backend image.
        boundary = (
            "----VoiceGuardIndexTTS2"
            + uuid.uuid4().hex
        )

        reference_bytes = ref.read_bytes()

        parts: list[bytes] = []

        def add_field(
            name: str,
            value: str,
        ) -> None:
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; '
                    f'name="{name}"\r\n'
                    f"\r\n"
                    f"{value}\r\n"
                ).encode("utf-8")
            )

        add_field("text", text)
        add_field("language", language)

        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; '
                f'name="reference"; filename="reference.wav"\r\n'
                f"Content-Type: audio/wav\r\n"
                f"\r\n"
            ).encode("utf-8")
        )

        parts.append(reference_bytes)
        parts.append(b"\r\n")

        parts.append(
            f"--{boundary}--\r\n".encode("utf-8")
        )

        body = b"".join(parts)

        req = request.Request(
            f"{INDEXTTS2_SERVICE_URL}/synthesize",
            data=body,
            method="POST",
            headers={
                "Content-Type": (
                    f"multipart/form-data; boundary={boundary}"
                ),
                "Content-Length": str(len(body)),
            },
        )

        try:
            # IndexTTS-2 is running on CPU on this machine and can take
            # considerably longer than 15 minutes for voice cloning.
            # Allow up to 2 hours for a genuine synthesis request.
            with request.urlopen(
                req,
                timeout=7200,
            ) as resp:  # noqa: S310
                if resp.status != 200:
                    raise RuntimeError(
                        f"IndexTTS-2 service returned HTTP "
                        f"{resp.status}"
                    )

                audio_bytes = resp.read()

        except error.HTTPError as exc:
            try:
                detail = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                detail = str(exc)

            raise RuntimeError(
                f"IndexTTS-2 service returned HTTP "
                f"{exc.code}: {detail[-1000:]}"
            ) from exc

        except error.URLError as exc:
            raise RuntimeError(
                "Could not connect to the IndexTTS-2 service at "
                f"{INDEXTTS2_SERVICE_URL}: {exc}"
            ) from exc

        except TimeoutError as exc:
            raise RuntimeError(
                "IndexTTS-2 synthesis timed out after 2 hours."
            ) from exc

        if not audio_bytes:
            raise RuntimeError(
                "IndexTTS-2 service returned an empty audio response."
            )

        # Save the returned WAV temporarily so soundfile can decode it.
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as tf:
            response_path = Path(tf.name)
            tf.write(audio_bytes)

        try:
            audio, sr = sf.read(
                str(response_path),
                dtype="float32",
                always_2d=False,
            )

            if getattr(audio, "ndim", 1) == 2:
                audio = audio.mean(axis=1)

            if len(audio) == 0:
                raise RuntimeError(
                    "IndexTTS-2 returned an empty WAV."
                )

            return np.asarray(
                audio,
                dtype=np.float32,
            ), int(sr)

        finally:
            response_path.unlink(missing_ok=True)


class XTTSEngine(CloneEngine):
    """Coqui XTTS v2 cloning engine."""

    name = "xtts"
    label = "Coqui XTTS v2 (zero-shot voice cloning)"
    worker_key = "xtts"
    languages = [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "ar",
        "zh",
        "ja",
    ]

    description = (
        "Multilingual zero-shot cloning from a short reference clip."
    )

    def is_available(self) -> bool:
        # XTTS auto-downloads weights, but we still gate on an explicit
        # weights directory so an unverified installation is not advertised.
        return (
            self._venv_python().exists()
            and self._weights_dir().exists()
        )