from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_jobs: dict[str, dict[str, Any]] = {}


def create_job(user: str, engine: str) -> str:
    job_id = uuid.uuid4().hex

    _jobs[job_id] = {
        "job_id": job_id,
        "user": user,
        "engine": engine,
        "status": "queued",
        "audio_url": None,
        "watermark_id": None,
        "synthesis_latency_ms": None,
        "error": None,
        "created_at": time.time(),
    }

    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def update_job(job_id: str, **values: Any) -> None:
    job = _jobs.get(job_id)

    if job is not None:
        job.update(values)


def run_synthesis_job(
    *,
    job_id: str,
    user: str,
    engine: str,
    text: str,
    voice: str,
    language: str,
    ref_path: str | None,
    media_dir: Path,
) -> None:
    """
    Long-running synthesis worker.

    This runs outside the original HTTP request so CPU-heavy IndexTTS-2
    generation cannot cause a browser/Nginx 504 timeout.
    """
    from voiceguard.synthesis.registry import registry as synth_registry
    from voiceguard.watermark import c2pa_sign
    from voiceguard.watermark.c2pa_watermark import embed, ensure_carrier_sr
    import soundfile as sf

    started = time.perf_counter()

    update_job(
        job_id,
        status="processing",
        error=None,
    )

    logger.info(
        "synthesis job started job=%s user=%s engine=%s",
        job_id,
        user,
        engine,
    )

    try:
        eng = synth_registry.get(engine)

        if eng is None:
            raise RuntimeError(
                f"Synthesis engine '{engine}' was not found."
            )

        if not eng.is_available():
            raise RuntimeError(
                f"Synthesis engine '{engine}' is not available."
            )

        # This is intentionally synchronous inside the background worker.
        # IndexTTS-2 may take a long time on CPU.
        audio, sr = eng.synthesize(
            text,
            voice=voice,
            reference_wav=ref_path,
            language=language,
        )

        # Keep the same watermark/provenance pipeline as the original
        # synchronous endpoint.
        wm_audio, wm_sr = ensure_carrier_sr(
            np.asarray(audio, dtype=np.float32),
            sr,
        )

        watermarked, watermark_id = embed(
            wm_audio,
            sr=wm_sr,
            amplitude=0.003,
        )

        fname = f"vg_{uuid.uuid4().hex}_{engine}.wav"
        out_path = media_dir / fname

        sf.write(
            str(out_path),
            watermarked,
            wm_sr,
        )

        # Best-effort C2PA signing.
        signed_tmp = out_path.with_suffix(".signed.wav")

        c2pa_status = c2pa_sign.sign_file(
            str(out_path),
            str(signed_tmp),
            software_agent=f"VoiceGuard/{engine}",
        )

        if c2pa_status.get("signed") and signed_tmp.exists():
            os.replace(
                str(signed_tmp),
                str(out_path),
            )
        else:
            signed_tmp.unlink(missing_ok=True)

        # Register final hash after optional C2PA signing.
        final_audio_hash: str | None = None

        try:
            final_audio_hash = hashlib.sha256(
                out_path.read_bytes()
            ).hexdigest()

            # Import lazily to avoid circular imports during application startup.
            from voiceguard.api.main import _remember_watermark

            _remember_watermark(
                final_audio_hash,
                watermark_id,
            )

            logger.info(
                "synthesis job provenance registered job=%s hash=%s watermark_id=%s",
                job_id,
                final_audio_hash,
                watermark_id,
            )

        except OSError:
            logger.warning(
                "could not register provenance for %s",
                out_path,
                exc_info=True,
            )

        # Schedule the same normal media cleanup.
        try:
            from voiceguard.api.main import _schedule_media_cleanup

            _schedule_media_cleanup(out_path)

        except Exception:
            logger.warning(
                "could not schedule media cleanup for %s",
                out_path,
                exc_info=True,
            )

        latency_ms = (time.perf_counter() - started) * 1000

        update_job(
            job_id,
            status="completed",
            audio_url=f"/api/media/{fname}",
            watermark_id=watermark_id,
            synthesis_latency_ms=round(latency_ms, 2),
            error=None,
        )

        logger.info(
            "synthesis job completed job=%s user=%s engine=%s latency_ms=%.1f",
            job_id,
            user,
            engine,
            latency_ms,
        )

    except Exception as exc:
        logger.exception(
            "synthesis job failed job=%s user=%s engine=%s",
            job_id,
            user,
            engine,
        )

        update_job(
            job_id,
            status="failed",
            error=str(exc),
            synthesis_latency_ms=round(
                (time.perf_counter() - started) * 1000,
                2,
            ),
        )

    finally:
        # Reference files are temporary and must be removed after synthesis,
        # not immediately after the HTTP request returns.
        if ref_path:
            try:
                Path(ref_path).unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "could not remove reference file %s",
                    ref_path,
                    exc_info=True,
                )