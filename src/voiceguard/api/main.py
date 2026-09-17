"""
VoiceGuard FastAPI application.

Endpoints:
    POST /token              â€” JWT token issuance
    POST /detect             â€” Upload audio, get deepfake detection result
    POST /synthesize         â€” Text-to-speech with C2PA watermarking
    POST /forensic/report    â€” Generate PDF forensic report
    WS   /ws/stream          â€” Real-time microphone streaming detection
    WS   /twilio/stream      â€” Twilio Media Stream bridge
    GET  /health             â€” Healthcheck

Security: JWT auth, slowapi 60 req/min, PDPL auto-delete â‰¤60s.
"""

import hashlib
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from livekit import api
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from voiceguard.api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    check_production_security,
    create_access_token,
    get_current_claims,
    get_current_user,
    role_for,
)
from voiceguard.api.middleware import (
    PDPLTimingMiddleware,
    limiter,
    make_temp_audio_file,
    pdpl_auto_delete,
)
from voiceguard.api.schemas import (
    DetectionResult,
    ExplanationResult,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackVerifyRequest,
    ForensicReportRequest,
    ForensicReportResult,
    HealthResponse,
    ModelType,
    StreamDetectionEvent,
    SynthesisEngineInfo,
    SynthesisResult,
    TokenResponse,
    WatermarkVerifyResult,
)
from voiceguard.forensics import result_store
from voiceguard.forensics import feedback_store
from voiceguard.api.synthesis_jobs import (
    create_job,
    get_job,
    run_synthesis_job,
)

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# â”€â”€ Model registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from voiceguard.models.registry import registry  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_production_security()  # refuse to start with a default SECRET_KEY in production
    # Use all CPU cores for inference (torch defaults to physical cores, leaving
    # vCPUs idle on cloud boxes) â€” roughly halves detection latency here.
    try:
        import torch

        torch.set_num_threads(int(os.environ.get("VG_TORCH_THREADS", os.cpu_count() or 4)))
    except Exception:
        logger.warning("could not set torch thread count", exc_info=True)
    registry.preload()  # loads any model whose env-var is set at startup
    # Warm the default hub anti-spoofing detector so the first /detect isn't slow.
    try:
        registry.load("wav2vec2_spoof")
    except Exception:
        logger.warning("could not warm wav2vec2_spoof detector", exc_info=True)
    _sweep_media_dir()  # remove stale media whose TTL timers were lost on restart
    yield


def _sweep_media_dir() -> None:
    """Delete MEDIA_DIR files older than MEDIA_TTL_S (TTL timers don't survive restart)."""
    now = time.time()
    for f in MEDIA_DIR.glob("*"):
        try:
            if f.is_file() and now - f.stat().st_mtime > MEDIA_TTL_S:
                f.unlink(missing_ok=True)
        except OSError:
            logger.warning("media sweep: could not remove %s", f, exc_info=True)


# â”€â”€ Application â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app = FastAPI(
    title="VoiceGuard API",
    version=__version__,
    description="Real-time voice deepfake detection and vishing defence API",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Allowed CORS origins: local dev + the self-hosted production domain. In the
# self-hosted deployment, Nginx serves the frontend and proxies /api on the SAME
# origin, so CORS is moot for the browser app; the domain is still allow-listed
# so direct-to-API requests work. Set VOICEGUARD_DOMAIN (e.g. "voiceguard.tech")
# to allow https://<domain> and https://www.<domain>. Extra origins (e.g. an
# ngrok tunnel) can be added via FRONTEND_ORIGINS (comma-separated); a single one
# via FRONTEND_ORIGIN.
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
_domain = os.environ.get("VOICEGUARD_DOMAIN", "").strip()
_domain_origins = [f"https://{_domain}", f"https://www.{_domain}"] if _domain else []
_single_origin = os.environ.get("FRONTEND_ORIGIN", "")
_extra_origins = os.environ.get("FRONTEND_ORIGINS", "")
ALLOWED_ORIGINS = list(
    dict.fromkeys(
        _DEFAULT_ORIGINS
        + _domain_origins
        + ([_single_origin] if _single_origin else [])
        + [o.strip() for o in _extra_origins.split(",") if o.strip()]
    )
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Auth is via Bearer token (Authorization header), not cookies, so credentialed
    # CORS is unnecessary â€” keeping it False avoids relaxing the same-origin policy.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Synthesised audio is written here and served at /media (i.e. /api/media/<f>
# once behind Nginx or the demo's /api mount). Auto-deleted after MEDIA_TTL_S.
MEDIA_DIR = Path(os.environ.get("VG_MEDIA_DIR", "/tmp/voiceguard_media"))  # noqa: S108  # nosec B108
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_TTL_S = int(os.environ.get("VG_MEDIA_TTL_S", "900"))
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ACCEPTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/flac",
    "audio/ogg",
    "application/ogg",
}
ACCEPTED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".oga"}
CONTENT_TYPE_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
_UPLOAD_CHUNK = 1024 * 1024  # 1MB
# Longest clip /detect will accept. libsndfile decode + one forward pass over up
# to VG_SCORE_SECONDS of audio run synchronously in the request, so an unbounded
# clip on CPU times out the 120s nginx proxy window â€” reject early with a clear
# message instead of a 504.
MAX_AUDIO_SECONDS = float(os.environ.get("VG_MAX_AUDIO_SECONDS", "600"))


def _sniff_is_audio(head: bytes) -> bool:
    """True when *head* starts like one of the advertised formats (WAV/FLAC/MP3/OGG).

    Extension and Content-Type are attacker-controlled; the bytes are what
    libsndfile will actually parse, so gate on them. MP3 is ID3-tagged or starts
    straight at an MPEG frame sync (0xFFEx); OGG containers always open with the
    "OggS" capture pattern.
    """
    if len(head) < 12:
        return False
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return True
    if head[:4] == b"fLaC":
        return True
    if head[:4] == b"OggS":
        return True
    if head[:3] == b"ID3":
        return True
    return head[0] == 0xFF and (head[1] & 0xE0) == 0xE0  # MPEG frame sync


async def save_upload(upload: UploadFile) -> tuple[str, str]:
    """Stream an upload to a temp file; return (path, sha256_hex).

    Rejects non-audio uploads (415) and streams in 1MB chunks â€” hashing
    incrementally and aborting with 413 as soon as the running size exceeds the
    limit â€” so a large body never lands fully in RAM.
    """
    import os as _os

    suffix = (Path(upload.filename or "audio.wav").suffix or ".wav").lower()
    ctype = (upload.content_type or "").split(";")[0].strip().lower()
    if suffix not in ACCEPTED_SUFFIXES:
        suffix = CONTENT_TYPE_SUFFIXES.get(ctype, suffix)
    if ctype not in ACCEPTED_CONTENT_TYPES and suffix not in ACCEPTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported media type â€” upload WAV, MP3, FLAC, or OGG audio.",
        )

    fd, path = make_temp_audio_file(suffix=suffix)
    sha256 = hashlib.sha256()
    total = 0
    sniffed = False
    try:
        with _os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(_UPLOAD_CHUNK):
                if not sniffed:
                    # Magic bytes, not just extension/Content-Type: the bytes are
                    # what libsndfile (CVE history) will actually parse.
                    if not _sniff_is_audio(chunk[:12]):
                        raise HTTPException(
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="File content is not WAV, MP3, FLAC, or OGG audio.",
                        )
                    sniffed = True
                total += len(chunk)
                if total > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit",
                    )
                sha256.update(chunk)
                out.write(chunk)
        if not sniffed:  # empty body never entered the loop
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Empty upload â€” send WAV, MP3, FLAC, or OGG audio.",
            )
    except HTTPException:
        Path(path).unlink(missing_ok=True)  # drop the partial file
        raise
    except OSError as exc:
        Path(path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to save upload") from exc

    PDPLTimingMiddleware.register(path)
    return path, sha256.hexdigest()


def _probe_audio(path: str) -> dict:
    """Header-only probe via soundfile: duration/sample-rate/format without a full
    decode. Rejects clips over MAX_AUDIO_SECONDS (413) before any model work â€”
    synchronous CPU inference on an unbounded clip would 504 behind nginx."""
    import soundfile as sf

    try:
        info = sf.info(path)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Could not decode audio â€” upload valid WAV, MP3, FLAC, or OGG.",
        ) from exc
    if info.duration > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Audio is {info.duration:.0f}s long; the analysis limit is "
                f"{MAX_AUDIO_SECONDS:.0f}s. Trim the clip or raise VG_MAX_AUDIO_SECONDS."
            ),
        )
    return {
        "duration_s": round(info.duration, 2),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": f"{info.format}/{info.subtype}",
    }


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    """Read an audio file to (mono float32 array, sample_rate) via soundfile.

    soundfile (libsndfile) handles WAV/FLAC/OGG and MP3 (libsndfile â‰¥ 1.1) without
    the optional TorchCodec backend that newer ``torchaudio.load`` requires â€” so
    this works identically in CI and on the deploy host.
    """
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    return np.ascontiguousarray(data, dtype=np.float32), sr
# â”€â”€ VoiceGuard generated-audio provenance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _watermark_record_path(audio_hash: str) -> Path:
    """Return the sidecar path storing the watermark ID for a generated file."""
    return MEDIA_DIR / f".wm_{audio_hash}.txt"


def _remember_watermark(audio_hash: str, watermark_id: str) -> None:
    """Persist final-audio hash -> watermark ID for later provenance detection."""
    path = _watermark_record_path(audio_hash)
    try:
        path.write_text(watermark_id, encoding="utf-8")
    except OSError:
        logger.warning(
            "could not persist watermark mapping for hash=%s",
            audio_hash,
            exc_info=True,
        )


def _watermark_id_for_hash(audio_hash: str) -> str | None:
    """Look up the VoiceGuard watermark ID associated with an audio hash."""
    path = _watermark_record_path(audio_hash)

    try:
        watermark_id = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None

    return watermark_id or None


def _verify_voiceguard_watermark(
    path: str,
    audio_hash: str,
) -> tuple[bool, float, str | None]:
    """Verify VoiceGuard provenance using spectral watermark and C2PA.

    The spectral watermark is checked when the persisted hash -> watermark-ID
    mapping is available. C2PA is also checked independently so a downloaded
    VoiceGuard-generated WAV remains provable even when its tiny sidecar file
    is not present after re-upload.

    Returns:
        (verified, correlation, watermark_id)
    """
    watermark_id = _watermark_id_for_hash(audio_hash)

    # First try the keyed spectral watermark when we have its ID.
    if watermark_id:
        try:
            from voiceguard.watermark.c2pa_watermark import detect as wm_detect

            data, sr = _read_audio(path)
            detected, correlation = wm_detect(
                data,
                sr=sr,
                watermark_id=watermark_id,
            )

            if detected:
                return True, float(correlation), watermark_id
        except Exception:
            logger.warning(
                "VoiceGuard spectral watermark verification failed for hash=%s",
                audio_hash,
                exc_info=True,
            )

    # C2PA is embedded in the final synthesized WAV. It does not depend on the
    # separate .wm_<hash>.txt sidecar. Only trust it as VoiceGuard provenance
    # when the signed manifest identifies VoiceGuard as the generating agent
    # and marks the asset as AI-generated.
    try:
        from voiceguard.watermark import c2pa_sign

        c2pa = c2pa_sign.verify_file(path)
        ai_generated = bool(c2pa.get("ai_generated"))
        software_agent = str(c2pa.get("software_agent") or "")

        if ai_generated and software_agent.lower().startswith("voiceguard/"):
            logger.info(
                "VoiceGuard C2PA provenance verified hash=%s agent=%s "
                "-> final=FAKE (ML detector skipped)",
                audio_hash,
                software_agent,
            )
            return True, 1.0, watermark_id
    except Exception:
        logger.warning(
            "VoiceGuard C2PA provenance verification failed for hash=%s",
            audio_hash,
            exc_info=True,
        )

    return False, 0.0, watermark_id

def _detect_classical(path: str) -> tuple[str, float]:
    """Run classical detection. Returns (label, confidence).

    Loads via soundfile so MP3/FLAC (not just WAV) are supported, matching the
    formats the API advertises.
    """
    from voiceguard.features.extractor import extract_features

    data, sr = _read_audio(path)
    mx = np.max(np.abs(data)) + 1e-8
    data = (data / mx).astype(np.float32)
    features = extract_features(data, sr)

    detector = registry.load("classical")
    if detector is None:
        return "real", 0.5
    return detector.predict_features(features)


# SSL models take (B, T); CNN models (DSFNet/AASIST) take (B, 1, T).
_SSL_KEYS = {
    "wav2vec2",
    "wav2vec2_v2",
    "wavlm_base_plus",
    "wavlm_large",
    "wav2vec2_large",
    "xls_r",
    "xls_r_aasist",
}
_WIN = 48000  # 3s @ 16kHz


def _score_seconds_env(name: str, default: float) -> float:
    """Parse a scoring-cap env var defensively: fall back on garbage, clamp to
    the 3s model minimum (a smaller cap would silently score zero-padding)."""
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        logger.warning("%s is not a number; using default %.0fs", name, default)
        value = default
    return max(value, _WIN / 16000)


# Longest audio scored in the single forward pass (memory/latency bound on CPU).
_SCORE_MAX_S = _score_seconds_env("VG_SCORE_SECONDS", 60.0)


def _load_wav_mono16k(path: str):
    """Load *path* as a mono (1, T) float32 tensor at 16 kHz (single file read)."""
    import torch
    import torchaudio

    data, sr = _read_audio(path)
    wav = torch.from_numpy(data).unsqueeze(0)  # (1, T)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def _guard_audio_quality(wav) -> None:
    """Reject too-short / near-silent clips â€” the detector is only meaningful on
    speech, and returns a confident (usually wrong) "fake" on silence/noise."""
    if wav.shape[-1] / 16000 < 0.8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio too short to analyse â€” please upload at least ~1 second of speech.",
        )
    if float(wav.pow(2).mean().sqrt()) < 1e-3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio is silent or near-silent â€” no speech detected to analyse.",
        )


def _model_device(model) -> str:
    """Ensure *model* is on the GPU when one is available; return its device str.

    SSL checkpoints load onto CPU (map_location='cpu'); on a GPU box we move the
    (cached) model onto the GPU once â€” subsequent calls are a no-op â€” so the
    forward pass runs on the A10G instead of the CPU.
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        p = next(model.parameters(), None)
        if p is not None and p.device.type != dev:
            model.to(dev)
    except Exception:
        return "cpu"
    return dev


def _ssl_fake_prob(model, wav, model_key: str) -> float:
    """One forward pass over `wav` (1, T) from its natural start; returns fake-prob.

    The SSL detector is only reliable on a recording scored *as recorded, from
    its start*: real audio opens with ambient lead-in while TTS starts mid-speech,
    and chunks cut from the middle of an utterance read as synthetic regardless
    of content (measured on the held-out set â€” sliding-window aggregations, max
    and mean alike, therefore flag real recordings as fake). Prepending silence
    is no fix either: it flips real fakes toward "real". So: single pass, capped
    at VG_SCORE_SECONDS; non-SSL research models keep their validated 3s crop.
    """
    import torch

    device = _model_device(model)  # move to GPU once (cached), else CPU
    w = wav[..., : int(_SCORE_MAX_S * 16000)].squeeze(0)
    if model_key not in _SSL_KEYS:
        w = w[..., :_WIN]  # research CNNs are fixed-length 3s models
    if w.shape[-1] < _WIN:
        w = torch.nn.functional.pad(w, (0, _WIN - w.shape[-1]))
    inp = w.unsqueeze(0) if model_key in _SSL_KEYS else w.reshape(1, 1, -1)
    inp = inp.to(device)
    with torch.no_grad():
        return float(torch.softmax(model(inp), dim=-1)[0, 1])
def _aasist_fake_prob(wav) -> tuple[float, float]:
    """Run official AASIST and return (fake_probability, seconds_analyzed)."""
    import torch

    model = registry.load("aasist")

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Official AASIST checkpoint not available.",
        )

    scored = wav[..., : int(_SCORE_MAX_S * 16000)]

    # Do not reject the AASIST branch independently for silence.
    # The live ensemble receives the same audio that Model 1 and Model 3
    # successfully score. An AASIST-only quality rejection raises HTTP 422
    # and aborts the complete three-model ensemble before Model 2 can return
    # its probability.
    #
    # _guard_audio_quality(scored)

    # Official AASIST expects 64,600 samples.
    # Zero-padding avoids repeating the same speech multiple times.
    x = scored.squeeze(0)
    target_len = 64600

    if x.shape[-1] < target_len:
        x = torch.nn.functional.pad(x, (0, target_len - x.shape[-1]))
    else:
        x = x[:target_len]

    x = x.unsqueeze(0)

    device = _model_device(model)
    x = x.to(device)

    with torch.no_grad():
        _, logits = model(x)
        probs = torch.softmax(logits, dim=-1)

    # Official AASIST: class 0 = spoof/fake, class 1 = bona fide/real.
    fake_prob = float(probs[0, 0])
    seconds_analyzed = round(min(scored.shape[-1], target_len) / 16000, 2)

    return fake_prob, seconds_analyzed


def _detect_aasist_tensor(wav) -> tuple[str, float, float]:
    """Run official AASIST on a mono 16-kHz waveform."""
    fake_prob, seconds_analyzed = _aasist_fake_prob(wav)
    label = "fake" if fake_prob >= 0.5 else "real"
    confidence = fake_prob if label == "fake" else 1.0 - fake_prob
    return label, round(confidence, 4), seconds_analyzed


def _detect_ensemble_tensor(
    wav,
) -> tuple[str, float, float, dict, dict, dict]:
    """Run the production three-model ensemble.

    Weights:
        - Original Wav2Vec2 Spoof: 30%
        - Official AASIST: 30%
        - Trained Wav2Vec2 v2: 40%

    All three models produce a fake probability. The final decision is made
    from the weighted probability score.
    """

    # Model 1: Original Wav2Vec2 Spoof
    wav_data = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    wav_label, wav_confidence = _detect_hf_array(
        wav_data, 16000, "wav2vec2_spoof"
    )
    wav_fake_prob = (
        wav_confidence if wav_label == "fake" else 1.0 - wav_confidence
    )

    # Model 2: Official AASIST
    aasist_fake_prob, seconds_analyzed = _aasist_fake_prob(wav)
    aasist_label = "fake" if aasist_fake_prob >= 0.5 else "real"
    aasist_confidence = (
        aasist_fake_prob
        if aasist_label == "fake"
        else 1.0 - aasist_fake_prob
    )

    # Model 3: Trained Wav2Vec2 v2
    wav2vec2_v2_model = registry.load("wav2vec2_v2")
    if wav2vec2_v2_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Trained Wav2Vec2 v2 checkpoint is not available. "
                "Expected checkpoint: checkpoints/wav2vec2_v2/model_best.pt"
            ),
        )

    wav2vec2_v2_fake_prob = _ssl_fake_prob(
        wav2vec2_v2_model,
        wav,
        "wav2vec2_v2",
    )
    wav2vec2_v2_label = (
        "fake" if wav2vec2_v2_fake_prob >= 0.5 else "real"
    )
    wav2vec2_v2_confidence = (
        wav2vec2_v2_fake_prob
        if wav2vec2_v2_label == "fake"
        else 1.0 - wav2vec2_v2_fake_prob
    )

    # Three-model weighted ensemble: 30% + 30% + 40% = 100%
    wav_weight = 0.30
    aasist_weight = 0.30
    wav2vec2_v2_weight = 0.40

    final_fake_prob = (
        wav_weight * wav_fake_prob
        + aasist_weight * aasist_fake_prob
        + wav2vec2_v2_weight * wav2vec2_v2_fake_prob
    )
    final_label = "fake" if final_fake_prob >= 0.5 else "real"
    final_confidence = (
        final_fake_prob
        if final_label == "fake"
        else 1.0 - final_fake_prob
    )

    wav2vec_result = {
        "label": wav_label,
        "confidence": round(float(wav_confidence), 4),
        "fake_probability": round(float(wav_fake_prob), 4),
        "weight": wav_weight,
    }
    aasist_result = {
        "label": aasist_label,
        "confidence": round(float(aasist_confidence), 4),
        "fake_probability": round(float(aasist_fake_prob), 4),
        "weight": aasist_weight,
    }
    wav2vec2_v2_result = {
        "label": wav2vec2_v2_label,
        "confidence": round(float(wav2vec2_v2_confidence), 4),
        "fake_probability": round(float(wav2vec2_v2_fake_prob), 4),
        "weight": wav2vec2_v2_weight,
    }

    logger.info(
        "ensemble wav2vec2_fake=%.4f weight=%.2f "
        "aasist_fake=%.4f weight=%.2f "
        "wav2vec2_v2_fake=%.4f weight=%.2f "
        "final_fake=%.4f final=%s confidence=%.4f",
        wav_fake_prob,
        wav_weight,
        aasist_fake_prob,
        aasist_weight,
        wav2vec2_v2_fake_prob,
        wav2vec2_v2_weight,
        final_fake_prob,
        final_label,
        final_confidence,
    )

    return (
        final_label,
        round(float(final_confidence), 4),
        seconds_analyzed,
        wav2vec_result,
        aasist_result,
        wav2vec2_v2_result,
    )


def _detect_ssl_tensor(wav, model_key: str) -> tuple[str, float, float]:
    """Single-pass SSL detection on the clip from its natural start.

    Returns (label, confidence, seconds_analyzed). Earlier sliding-window
    aggregations (max, then mean) both misclassified real recordings because
    mid-utterance windows are out-of-distribution for the detector; the whole
    clip in one pass matches the regime the model's official EER and held-out
    real-pass were validated under. seconds_analyzed = min(duration,
    VG_SCORE_SECONDS) so callers and forensic reports can disclose truncation.
    """
    model = registry.load(model_key)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_key}' checkpoint not found. Set {model_key.upper()}_PATH.",
        )
    # Guard the region that will actually be scored, not the whole clip â€” a long
    # recording with a near-silent first VG_SCORE_SECONDS must 422, not let the
    # model confidently call silence "fake".
    scored = wav[..., : int(_SCORE_MAX_S * 16000)]
    # AASIST branch: do not reject this audio independently for silence.
    # The shared live ensemble must allow all three models to score the same audio.
    # _guard_audio_quality(scored)
    fake_prob = _ssl_fake_prob(model, wav, model_key)
    label = "fake" if fake_prob >= 0.5 else "real"
    confidence = fake_prob if label == "fake" else 1.0 - fake_prob
    return label, round(confidence, 4), round(scored.shape[-1] / 16000, 2)


def _detect_ssl(path: str, model_key: str) -> tuple[str, float, int]:
    return _detect_ssl_tensor(_load_wav_mono16k(path), model_key)


def _first_window(wav):
    """First 3s window of `wav` (1, T) as a padded (1, _WIN) tensor for attribution."""
    import torch

    w = wav.squeeze(0)
    w = torch.nn.functional.pad(w, (0, _WIN - w.shape[-1])) if w.shape[-1] < _WIN else w[:_WIN]
    return w.unsqueeze(0)


def _build_explanation(model, wav_1xT) -> ExplanationResult | None:
    """Integrated-Gradients attribution â†’ ExplanationResult (None on failure)."""
    try:
        from voiceguard.api.schemas import AttributionSegment
        from voiceguard.xai.ssl_explain import explain_waveform

        raw = explain_waveform(model, wav_1xT)
        return ExplanationResult(
            method=raw["method"],
            baseline=raw["baseline"],
            target_class=raw["target_class"],
            frame_duration_ms=raw["frame_duration_ms"],
            attribution_frames=raw["attribution_frames"],
            top_segments=[AttributionSegment(**s) for s in raw["top_segments"]],
        )
    except Exception:
        logger.warning("attribution failed", exc_info=True)
        return None


def _explain_ssl(path: str, model_key: str) -> ExplanationResult | None:
    """Run Integrated Gradients attribution on an SSL model (loads its own file)."""
    model = registry.load(model_key)
    if model is None:
        return None
    return _build_explanation(model, _first_window(_load_wav_mono16k(path)))


def _explain_occlusion(
    path: str, scorer, max_s: float = 10.0, n_seg_cap: int = 20
) -> ExplanationResult | None:
    """Occlusion attribution â†’ ExplanationResult, for non-differentiable models.

    Each time segment is silenced in turn and the clip re-scored via *scorer*
    (a callable ``(audio, sr) -> (label, confidence)``); a segment's importance
    is how much its removal moves the fake-probability away from the predicted
    class. Output matches the Integrated-Gradients shape (normalised 10ms frames
    + top segments) so clients render both identically. ``max_s`` / ``n_seg_cap``
    bound the number of re-scoring passes â€” keep them small for slow SSL models.
    """
    try:
        from voiceguard.api.schemas import AttributionSegment

        data, sr = _read_audio(path)
        data = (data / (np.max(np.abs(data)) + 1e-8)).astype(np.float32)
        data = data[: int(max_s * sr)]  # bound re-scoring cost
        duration = len(data) / sr

        def fake_logit(x: np.ndarray) -> float:
            # Logit-space scoring: classifier probabilities saturate near 0/1,
            # flattening per-segment differences that the raw evidence retains.
            label, conf = scorer(x, sr)
            p = np.clip(conf if label == "fake" else 1.0 - conf, 1e-7, 1 - 1e-7)
            return float(np.log(p / (1.0 - p)))

        base = fake_logit(data)
        target_class = 1 if base >= 0 else 0
        n_seg = min(n_seg_cap, max(6, int(duration / 0.25)))
        seg_len = len(data) / n_seg
        drops = np.zeros(n_seg)
        for i in range(n_seg):
            occluded = data.copy()
            occluded[int(i * seg_len) : int((i + 1) * seg_len)] = 0.0
            p = fake_logit(occluded)
            drops[i] = (base - p) if target_class == 1 else (p - base)
        drops = np.maximum(drops, 0.0)
        if drops.max() > 0:
            drops = drops / drops.max()

        seg_dur = duration / n_seg
        n_frames = int(duration * 100)
        idx = np.minimum((np.arange(n_frames) * 0.01 / seg_dur).astype(int), n_seg - 1)
        top = [
            AttributionSegment(
                start_s=round(i * seg_dur, 2),
                end_s=round((i + 1) * seg_dur, 2),
                importance=round(float(drops[i]), 4),
            )
            for i in np.argsort(drops)[::-1][:5]
            if drops[i] > 0
        ]
        return ExplanationResult(
            method="occlusion",
            baseline="silence",
            target_class=target_class,
            frame_duration_ms=10,
            attribution_frames=[round(float(f), 4) for f in drops[idx]],
            top_segments=top,
        )
    except Exception:
        logger.warning("occlusion attribution failed", exc_info=True)
        return None


def _detect_hf_array(
    audio: np.ndarray, sr: int, model_key: str = "wav2vec2_spoof"
) -> tuple[str, float]:
    """Score a raw waveform with a HuggingFace anti-spoofing detector."""
    detector = registry.load(model_key)
    if detector is None:
        return "real", 0.5
    return detector.predict_array(audio, sr)


def _detect_hf(path: str, model_key: str = "wav2vec2_spoof") -> tuple[str, float]:
    """Run HuggingFace anti-spoofing detection on a file. Returns (label, conf)."""
    data, sr = _read_audio(path)
    return _detect_hf_array(data, sr, model_key)


def _explain_classical(path: str) -> ExplanationResult | None:
    """Occlusion attribution using the classical detector."""
    return _explain_occlusion(path, _detect_classical_array)


def _explain_hf(path: str, model_key: str = "wav2vec2_spoof") -> ExplanationResult | None:
    """Occlusion attribution using a HuggingFace anti-spoofing detector."""
    return _explain_occlusion(path, lambda a, s: _detect_hf_array(a, s, model_key))


def _ssl_array_scorer(model_key: str):
    """Scorer ``(audio, sr) -> (label, conf)`` backed by an SSL detector, for occlusion."""
    import torch
    import torchaudio

    model = registry.load(model_key)

    def scorer(audio: np.ndarray, sr: int) -> tuple[str, float]:
        if model is None:
            return "real", 0.5
        w = torch.as_tensor(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        fp = _ssl_fake_prob(model, w, model_key)
        return ("fake", fp) if fp >= 0.5 else ("real", 1.0 - fp)

    return scorer


def _explain_ssl_fast(path: str, model_key: str) -> ExplanationResult | None:
    """Fast occlusion attribution for a (slow, 300M-param) SSL detector.

    Integrated Gradients back-propagates through the SSL model ~25Ã— and takes
    over a minute on CPU. Occlusion over a short window is forward-only and an
    order of magnitude faster, while giving the same per-moment picture.
    """
    if registry.load(model_key) is None:
        return None
    # Score ~3s in 6 silence-one-window passes â†’ seconds, not minutes.
    return _explain_occlusion(path, _ssl_array_scorer(model_key), max_s=3.0, n_seg_cap=6)


# â”€â”€ LLM narrative (optional, via Amazon Bedrock) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_BEDROCK_KEY = os.environ.get("VG_BEDROCK_API_KEY", "").strip()
_BEDROCK_REGION = os.environ.get("VG_BEDROCK_REGION", "us-east-1")
_BEDROCK_MODEL = os.environ.get("VG_BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

_MODEL_HUMAN = {
    "xls_r_aasist": "XLS-R-300M + AASIST (the v9c production detector)",
    "wav2vec2_spoof": "a wav2vec2 anti-spoofing model",
    "aasist": "the official AASIST anti-spoofing model",
    "ensemble": "Wav2Vec2 Spoof + official AASIST ensemble",
    "classical": "the classical MFCC baseline",
}


def _llm_narrative(
    label: str,
    confidence: float,
    model_key: str,
    explanation: ExplanationResult,
    seconds_analyzed: float | None,
) -> str | None:
    """Turn the detector's own numbers into a short plain-language forensic note.

    Calls Amazon Bedrock (Claude Haiku) with a bearer API key over stdlib HTTP â€”
    no boto3, no SigV4. Returns None on any failure or when no key is configured,
    so the /explain and /detect paths degrade gracefully and never break.
    """
    if not _BEDROCK_KEY:
        return None
    import json
    import urllib.parse
    import urllib.request

    fake_p = confidence if label == "fake" else 1.0 - confidence
    moments = (
        ", ".join(
            f"{s.start_s:.1f}â€“{s.end_s:.1f}s (weight {s.importance:.2f})"
            for s in explanation.top_segments[:4]
        )
        or "no single dominant window; the signal is spread across the clip"
    )
    detector = _MODEL_HUMAN.get(model_key, model_key)
    facts = (
        f"Detector: {detector}.\n"
        f"Verdict: {label.upper()} (fake-probability {fake_p:.1%}, confidence {confidence:.1%}).\n"
        f"Seconds of audio analysed: {seconds_analyzed or 'whole clip'}.\n"
        f"Attribution method: {explanation.method}.\n"
        f"Most influential moments: {moments}."
    )
    system = (
        "You are a forensic audio analyst assistant for VoiceGuard, an AI voice "
        "deepfake detector. Given ONLY the detector's numeric output, write a short, "
        "accurate, plain-language explanation an examiner can read aloud. 2â€“3 "
        "sentences, max ~60 words. Explain what the verdict means and what the "
        "attribution moments indicate (which parts of audio most drove the decision). "
        "These detectors flag statistical artefacts of synthesis (unnatural spectral "
        "detail, prosody, phase) that are inaudible â€” do NOT invent specific "
        "acoustic measurements, transcripts, or claims you were not given. Be measured: "
        "this is decision-support, not proof. Plain text only â€” no markdown, no "
        "asterisks, no bullet points, no preamble."
    )
    body = json.dumps(
        {
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": facts}]}],
            "inferenceConfig": {"maxTokens": 200, "temperature": 0.2},
        }
    ).encode()
    model_esc = urllib.parse.quote(_BEDROCK_MODEL, safe="")
    url = f"https://bedrock-runtime.{_BEDROCK_REGION}.amazonaws.com/model/{model_esc}/converse"
    req = urllib.request.Request(  # noqa: S310  # nosec B310  (fixed https Bedrock URL)
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {_BEDROCK_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310  # nosec B310
            data = json.loads(resp.read().decode())
        parts = data["output"]["message"]["content"]
        text = " ".join(p.get("text", "") for p in parts).strip()
        text = text.replace("**", "").replace("*", "").strip()  # belt-and-braces: no markdown
        return text or None
    except Exception:
        logger.warning("bedrock narrative failed", exc_info=True)
        return None


def _attach_narrative(
    explanation: ExplanationResult | None,
    label: str,
    confidence: float,
    model_key: str,
    seconds_analyzed: float | None,
) -> ExplanationResult | None:
    """Populate explanation.narrative in place (best-effort)."""
    if explanation is None:
        return None
    explanation.narrative = _llm_narrative(
        label, confidence, model_key, explanation, seconds_analyzed
    )
    return explanation


def _narrative_from_record(record: dict) -> str | None:
    """Generate a narrative from a stored detection record (no attribution segments).

    Used by /forensic/report when the detection wasn't run with explain=true, so
    every report can still carry an AI analysis. Builds a minimal explanation
    (empty segments â†’ the helper's 'spread across the clip' phrasing) and calls
    the same Bedrock path.
    """
    stub = ExplanationResult(
        method="verdict summary",
        baseline="n/a",
        target_class=1 if record["label"] == "fake" else 0,
        frame_duration_ms=10,
        attribution_frames=[],
        top_segments=[],
    )
    return _llm_narrative(
        record["label"],
        float(record["confidence"]),
        str(record["model"]),
        stub,
        record.get("seconds_analyzed"),
    )


# â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.get("/livekit/token", tags=["livekit"])
async def get_livekit_token(
    room: str = "voiceguard-call",
    identity: str | None = None,
):
    """Create a LiveKit access token for the VoiceGuard Call tab."""

    api_key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "secret")

    participant_identity = identity or f"voiceguard-{uuid.uuid4().hex[:8]}"

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(participant_identity)
        .with_name(participant_identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    return {
        "token": token,
        "url": os.environ.get(
            "LIVEKIT_BROWSER_URL",
            "ws://localhost:7880",
        ),
        "room": room,
        "identity": participant_identity,
    }

@app.post("/token", response_model=TokenResponse, tags=["auth"])
@limiter.limit("5/minute")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(OAuth2PasswordRequestForm),
):
    if not authenticate_user(form_data.username, form_data.password):
        logger.info("login failed for user=%r", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        {"sub": form_data.username, "role": role_for(form_data.username)},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(access_token=token)


@app.post("/detect", response_model=DetectionResult, tags=["detection"])
@limiter.limit("60/minute")
async def detect(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: ModelType = ModelType.ensemble,
    explain: bool = False,
    _user: str = Depends(get_current_user),
):
    """Upload an audio file and receive a deepfake detection result.

    Accepted formats: WAV, MP3, FLAC, OGG (max 100MB). The clip is scored in a
    single forward pass from its natural start, exactly as recorded â€” sliding
    windows misclassify (mid-utterance segments read as synthetic to the model)
    â€” capped at VG_SCORE_SECONDS (default 60s) of audio; `seconds_analyzed`
    reports how much of the clip the verdict covers. Raw audio is auto-deleted
    after 60 seconds (PDPL compliance). Pass `explain=true` to include
    Integrated Gradients attribution showing which time segments drove the
    decision.
    """
    path, audio_hash = await save_upload(file)
    background_tasks.add_task(pdpl_auto_delete, path)

    audio_info = _probe_audio(path)  # also enforces MAX_AUDIO_SECONDS (413)

    t0 = time.perf_counter()

    # IMPORTANT: check VoiceGuard provenance BEFORE loading the heavy ML detector.
    # A verified VoiceGuard watermark is authoritative evidence that this exact
    # audio was generated by VoiceGuard, so it must not wait for the 1.2 GB model.
    watermark_verified, watermark_correlation, watermark_id = (
        _verify_voiceguard_watermark(path, audio_hash)
    )

    wav2vec2_spoof_result = None
    aasist_result = None
    wav2vec2_v2_result = None

    if watermark_verified and model != ModelType.ensemble:
        label = "fake"
        confidence = 1.0
        seconds_analyzed = audio_info.get("duration_s")
        explanation = None
        logger.info(
            "VoiceGuard provenance verified hash=%s watermark_id=%s corr=%.4f "
            "-> final=FAKE (ML detector skipped)",
            audio_hash,
            watermark_id,
            watermark_correlation,
        )
    elif model == ModelType.classical:
        label, confidence = _detect_classical(path)
        seconds_analyzed = audio_info.get("duration_s")
        explanation = _explain_classical(path) if explain else None

    elif model == ModelType.aasist:
        wav = _load_wav_mono16k(path)
        label, confidence, seconds_analyzed = _detect_aasist_tensor(wav)
        aasist_result = {
            "label": label,
            "confidence": confidence,
            "fake_probability": round(
                confidence if label == "fake" else 1.0 - confidence, 4
            ),
        }
        explanation = None

    elif model == ModelType.ensemble:
        wav = _load_wav_mono16k(path)
        (
            label,
            confidence,
            seconds_analyzed,
            wav2vec2_spoof_result,
            aasist_result,
            wav2vec2_v2_result,
        ) = _detect_ensemble_tensor(wav)
        explanation = None

    else:
        wav = _load_wav_mono16k(path)
        label, confidence, seconds_analyzed = _detect_ssl_tensor(
            wav, str(model)
        )
        explanation = _explain_ssl_fast(path, str(model)) if explain else None

    if explanation is not None:
        _attach_narrative(
            explanation,
            label,
            confidence,
            str(model),
            seconds_analyzed,
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    # Record the server-verified result so /forensic/report can't be forged with a
    # client-supplied verdict (P0-10).
    result_store.record(
        audio_hash,
        label=label,
        confidence=round(confidence, 4),
        model=str(model),
        user=_user,
        timestamp=time.time(),
        windows_analyzed=1,
        seconds_analyzed=seconds_analyzed,
        audio_info=audio_info,
        app_version=__version__,
        narrative=explanation.narrative if explanation is not None else None,
    )
    logger.info(
        "detect user=%s model=%s verdict=%s conf=%.3f analyzed_s=%s latency_ms=%.1f",
        _user,
        model,
        label,
        confidence,
        seconds_analyzed,
        latency_ms,
    )

    return DetectionResult(
        label=label,
        confidence=round(confidence, 4),
        model=model,
        wav2vec2_spoof=wav2vec2_spoof_result,
        aasist=aasist_result,
        wav2vec2_v2=wav2vec2_v2_result,
        latency_ms=round(latency_ms, 2),
        audio_hash=audio_hash,
        windows_analyzed=1,
        seconds_analyzed=seconds_analyzed,
        explanation=explanation,
    )

@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["feedback"],
)
@limiter.limit("30/minute")
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    _user: str = Depends(get_current_user),
):
    """Submit user feedback for a previous detection.

    Feedback is stored as unverified. It does not automatically become
    training ground truth.
    """

    feedback = body.feedback.strip().lower()

    if feedback not in {"correct", "incorrect"}:
        raise HTTPException(
            status_code=400,
            detail="Feedback must be 'correct' or 'incorrect'.",
        )

    # Only accept feedback for a detection that VoiceGuard actually
    # produced and stored on the server.
    detection = result_store.get(body.audio_hash)

    if detection is None:
        raise HTTPException(
            status_code=404,
            detail="Detection result not found for this audio hash.",
        )

    record = feedback_store.record(
        audio_hash=body.audio_hash,
        predicted_label=str(detection["label"]),
        predicted_confidence=float(detection["confidence"]),
        model=str(detection["model"]),
        feedback=feedback,
        user=_user,
        timestamp=time.time(),
    )

    return FeedbackResponse(
        feedback_id=record["feedback_id"],
        audio_hash=record["audio_hash"],
        predicted_label=record["predicted_label"],
        predicted_confidence=record["predicted_confidence"],
        model=record["model"],
        feedback=record["feedback"],
        verification_status=record["verification_status"],
    )


@app.get(
    "/feedback/unverified",
    tags=["feedback"],
)
@limiter.limit("60/minute")
async def list_unverified_feedback(
    request: Request,
    claims: tuple[str, str] = Depends(get_current_claims),
):
    """Return feedback waiting for independent admin verification."""

    _user, _role = claims

    if _role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )

    return {
        "items": feedback_store.list_unverified(limit=100),
    }


@app.post(
    "/feedback/{feedback_id}/verify",
    tags=["feedback"],
)
@limiter.limit("30/minute")
async def verify_feedback(
    request: Request,
    feedback_id: str,
    body: FeedbackVerifyRequest,
    claims: tuple[str, str] = Depends(get_current_claims),
):
    """Independently verify a feedback record for future training use."""

    _user, _role = claims

    if _role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )

    verified_label = body.verified_label.strip().lower()

    if verified_label not in {"real", "fake"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="verified_label must be 'real' or 'fake'.",
        )

    record = feedback_store.verify(
        feedback_id=feedback_id,
        verified_label=verified_label,
        verified_by=_user,
        verified_at=time.time(),
    )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found or it has already been verified.",
        )

    return record


@app.post(
    "/feedback/{feedback_id}/promote",
    tags=["feedback"],
)
@limiter.limit("10/minute")
async def promote_verified_feedback(
    request: Request,
    background_tasks: BackgroundTasks,
    feedback_id: str,
    file: UploadFile = File(...),
    claims: tuple[str, str] = Depends(get_current_claims),
):
    """Promote an explicitly re-uploaded, independently verified audio sample
    into the persistent training dataset.

    The uploaded audio must have the exact SHA-256 hash recorded in the
    verified feedback record. This prevents an unrelated audio file from
    being assigned the verified label.
    """

    _user, _role = claims

    if _role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )

    record = feedback_store.get(feedback_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback record not found.",
        )

    if record.get("verification_status") != "verified":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feedback must be independently verified before promotion.",
        )

    verified_label = str(record.get("verified_label") or "").strip().lower()

    if verified_label not in {"real", "fake"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Verified feedback does not contain a valid training label.",
        )

    path = None

    try:
        path, uploaded_hash = await save_upload(file)

        expected_hash = str(record["audio_hash"]).strip().lower()

        if uploaded_hash.lower() != expected_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Uploaded audio does not match the verified feedback audio hash.",
            )

        dataset_root = Path(
            os.environ.get(
                "VG_VERIFIED_DATASET_DIR",
                "/app/data/verified_training",
            )
        )

        destination_dir = dataset_root / verified_label
        destination_dir.mkdir(parents=True, exist_ok=True)

        suffix = (
            Path(file.filename or "audio.wav").suffix.lower()
            or ".wav"
        )

        destination = destination_dir / f"{uploaded_hash}{suffix}"

        if destination.exists():
            return {
                "status": "already_present",
                "feedback_id": feedback_id,
                "audio_hash": uploaded_hash,
                "verified_label": verified_label,
                "dataset_path": str(destination),
            }

        shutil.copy2(path, destination)

        return {
            "status": "promoted",
            "feedback_id": feedback_id,
            "audio_hash": uploaded_hash,
            "verified_label": verified_label,
            "dataset_path": str(destination),
        }

    finally:
        if path is not None:
            background_tasks.add_task(pdpl_auto_delete, path)


CLONE_QUOTA_PER_HOUR = int(os.environ.get("VG_CLONE_QUOTA_PER_HOUR", "10"))
_clone_log: dict[str, list[float]] = {}


def _claim_clone_quota(user: str) -> bool:
    """Per-user sliding-hour quota for voice cloning (the abuse-sensitive path)."""
    now = time.time()
    log = [t for t in _clone_log.get(user, []) if now - t < 3600]
    if len(log) >= CLONE_QUOTA_PER_HOUR:
        _clone_log[user] = log
        return False
    log.append(now)
    _clone_log[user] = log
    return True


def _schedule_media_cleanup(path: Path) -> None:
    """Delete a generated media file after MEDIA_TTL_S (PDPL minimisation)."""
    import threading

    def _rm() -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    # Daemon timer: a pending best-effort cleanup must never block process
    # shutdown (otherwise the API â€” and the test suite â€” hangs for MEDIA_TTL_S).
    timer = threading.Timer(MEDIA_TTL_S, _rm)
    timer.daemon = True
    timer.start()


@app.get("/synthesis/engines", response_model=list[SynthesisEngineInfo], tags=["synthesis"])
async def synthesis_engines():
    """List synthesis engines and their availability (preset TTS + voice cloning)."""
    from voiceguard.synthesis.registry import registry as synth_registry

    return [SynthesisEngineInfo(**vars(i)) for i in synth_registry.info()]


@app.post("/synthesize", tags=["synthesis"])
@limiter.limit("20/minute")
async def synthesize(
    request: Request,
    background_tasks: BackgroundTasks,
    text: str = Form(..., min_length=1, max_length=2000),
    engine: str = Form("kokoro"),
    voice: str = Form("af_heart"),
    language: str = Form("en"),
    consent: bool = Form(False),
    reference: UploadFile | None = File(None),
    claims: tuple[str, str] = Depends(get_current_claims),
):
    """Queue synthesis and return immediately with a job ID.

    `engine` selects a preset-voice TTS (Kokoro) or a zero-shot voice-cloning
    engine. Cloning engines require a `reference` audio clip. The actual
    synthesis, watermarking, and C2PA signing run in a background worker so
    slow CPU-based IndexTTS-2 generation cannot hold the HTTP request open.
    """
    from voiceguard.synthesis.registry import registry as synth_registry

    _user, _role = claims

    eng = synth_registry.get(engine)
    if eng is None or not eng.is_available():
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Synthesis engine '{engine}' is not available on this instance.",
        )

    ref_path: str | None = None

    if eng.requires_reference:
        # Voice cloning is the abuse-sensitive path: admin-only, with a per-user
        # hourly quota on top of the route's per-IP rate limit. Preset TTS stays
        # open to analysts.
        if _role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Voice cloning requires the 'admin' role.",
            )

        # Voice cloning consent is enforced server-side, not just in the UI.
        if not consent:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Voice cloning requires explicit consent that you are authorised to "
                    "clone this voice. Set consent=true to proceed."
                ),
            )

        if reference is None:
            raise HTTPException(
                status_code=422,
                detail=f"Engine '{engine}' requires a reference audio clip.",
            )

        # Claimed only once the request is otherwise valid, so rejected attempts
        # don't burn the user's quota.
        if not _claim_clone_quota(_user):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Voice-clone quota reached ({CLONE_QUOTA_PER_HOUR}/hour per user).",
            )

        ref_path, _ = await save_upload(reference)
        logger.info(
            "clone consent acknowledged user=%s engine=%s",
            _user,
            engine,
        )

    # Create the job before scheduling the worker so the client can immediately
    # start polling its status.
    job_id = create_job(_user, engine)

    background_tasks.add_task(
        run_synthesis_job,
        job_id=job_id,
        user=_user,
        engine=engine,
        text=text,
        voice=voice,
        language=language,
        ref_path=ref_path,
        media_dir=MEDIA_DIR,
    )

    logger.info(
        "synthesis job queued user=%s engine=%s job=%s",
        _user,
        engine,
        job_id,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "engine": engine,
    }


@app.get("/synthesis/jobs/{job_id}", tags=["synthesis"])
async def synthesis_job_status(
    job_id: str,
    claims: tuple[str, str] = Depends(get_current_claims),
):
    """Return the status/result of an asynchronous synthesis job."""
    _user, _role = claims

    job = get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Synthesis job not found.",
        )

    # Jobs are private to the user that created them.
    if job.get("user") != _user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Synthesis job not found.",
        )

    return job


@app.post("/forensic/report", response_model=ForensicReportResult, tags=["forensics"])
@limiter.limit("10/minute")
async def forensic_report(
    request: Request,
    body: ForensicReportRequest,
    _user: str = Depends(get_current_user),
):
    """Generate a NIST SP 800-86 compliant PDF forensic report.

    The verdict is taken from VoiceGuard's own server-side detection record for the
    audio hash (set by /detect), NOT from any client-supplied value â€” a report
    cannot be forged with a fabricated verdict. Returns 404 if no detection has
    been run for this hash.
    """
    from voiceguard.forensics.chain_of_custody import ChainOfCustody
    from voiceguard.forensics.pdf_report import generate_report

    record = result_store.get(body.audio_hash)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No server-side detection record for this audio hash â€” run /detect first.",
        )
    # Build the report from the verified record; the request's detection_result is ignored.
    detection_result = {
        "label": record["label"],
        "confidence": record["confidence"],
        "model": record["model"],
        "server_verified": True,
    }

    coc = ChainOfCustody()
    coc.add_event("evidence_received", body.analyst_name, body.audio_hash, "Audio submitted")
    coc.add_event(
        "analysis_completed",
        "VoiceGuard",
        body.audio_hash,
        f"Verdict: {record['label']} (server-verified)",
    )

    audio_meta = dict(record.get("audio_info") or {})
    if record.get("windows_analyzed") is not None:
        audio_meta["windows_analyzed"] = record["windows_analyzed"]
    if record.get("seconds_analyzed") is not None:
        audio_meta["seconds_analyzed"] = record["seconds_analyzed"]
    model_meta = {
        "model": record["model"],
        "app_version": record.get("app_version", __version__),
        # Cached after the first report; ~1 GB SSL checkpoint hashes in seconds.
        "checkpoint_sha256": registry.fingerprint(record["model"]),
    }

    # AI forensic narrative: reuse the one captured at detect time (richer â€” it
    # had attribution segments), else generate one now from the verified record.
    narrative = record.get("narrative") or _narrative_from_record(record)

    # uuid4, not a timestamp: /media is unauthenticated, so names must be unguessable
    fname = f"report_{body.audio_hash[:12]}_{uuid.uuid4().hex}.pdf"
    out_path = MEDIA_DIR / fname
    try:
        generate_report(
            audio_hash=body.audio_hash,
            detection_result=detection_result,
            chain_of_custody=coc.to_dict(),
            analyst_name=body.analyst_name,
            output_path=out_path,
            audio_meta=audio_meta or None,
            model_meta=model_meta,
            narrative=narrative,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

    _schedule_media_cleanup(out_path)
    return ForensicReportResult(
        report_url=f"/api/media/{fname}",
        chain_of_custody_hash=coc.chain_hash,
    )


class _ConnectionBudget:
    """Process-wide cap on concurrent streaming connections.

    Every stream window drives model inference on a CPU box, so an uncapped
    WebSocket is a one-client DoS. Single event loop => plain counter is safe.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0

    def acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)


_stream_budget = _ConnectionBudget(int(os.environ.get("VG_WS_MAX_CONNECTIONS", "4")))
_WS_MAX_SECONDS = float(os.environ.get("VG_WS_MAX_SECONDS", "900"))
_WS_PCM_BYTES_PER_S = 16000 * 2  # realtime int16 @ 16 kHz
_WS_RATE_SLACK = 4.0  # tolerate 4x realtime (buffered sends) before closing
_WS_BURST_BYTES = 512 * 1024


async def _ws_first_message_token(websocket: WebSocket) -> str:
    """Read the JWT from the first WS message: ``{"token": "<jwt>"}``.

    Preferred over ``?token=`` â€” query strings end up in nginx/proxy access
    logs. Returns "" on timeout, non-text frame, or malformed JSON.
    """
    import asyncio
    import json

    try:
        msg = await asyncio.wait_for(websocket.receive(), timeout=5.0)
    except TimeoutError:
        return ""
    if msg.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(code=int(msg.get("code") or 1000))
    text = msg.get("text")
    if not text:
        return ""
    try:
        return str(json.loads(text).get("token", ""))
    except (ValueError, AttributeError):
        return ""


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, token: str = ""):
    """Real-time microphone streaming detection.

    Auth: send ``{"token": "<jwt>"}`` as the first (text) message â€” the server
    replies ``{"type": "auth_ok"}`` â€” then stream raw PCM frames (int16, 16kHz,
    mono). ``?token=`` works too but is deprecated (it leaks into proxy logs).
    Server responds with JSON StreamDetectionEvent messages. Connections are
    capped globally (VG_WS_MAX_CONNECTIONS), per-session (VG_WS_MAX_SECONDS),
    and at ~4x realtime ingest.
    """
    await websocket.accept()
    if not token:
        try:
            token = await _ws_first_message_token(websocket)
        except WebSocketDisconnect:
            return
    try:
        verify_token_ws(token)
    except HTTPException:
        await websocket.close(code=1008)
        return

    if not _stream_budget.acquire():
        await websocket.close(code=1013)  # try again later â€” all slots busy
        return

    import asyncio

    # The detector is only reliable on audio scored from the recording's natural
    # start (see _ssl_fake_prob) â€” a mic stream's natural start is the moment the
    # user hit Start. So each verdict re-scores the growing prefix of the session
    # (first at 3s, then every ~2s), capped at VG_WS_SCORE_SECONDS of audio (CPU
    # cost grows with prefix length). The verdict that covers the full cap is sent
    # with final=true and scoring stops; the connection stays open only so the
    # session/rate limits keep being enforced.
    first_score_bytes = 3 * _WS_PCM_BYTES_PER_S
    stride_bytes = 2 * _WS_PCM_BYTES_PER_S
    cap_s = _score_seconds_env("VG_WS_SCORE_SECONDS", 15.0)
    # Even-align so a prefix slice is always whole int16 samples.
    score_cap_bytes = max(int(cap_s * _WS_PCM_BYTES_PER_S) & ~1, first_score_bytes)
    buffer = bytearray()
    window_id = 0
    next_score_at = first_score_bytes
    verdict_final = False
    started = time.monotonic()
    received = 0

    try:
        await websocket.send_json({"type": "auth_ok"})
        while True:
            data = await websocket.receive_bytes()
            received += len(data)
            elapsed = time.monotonic() - started
            if elapsed > _WS_MAX_SECONDS:
                await websocket.close(code=1000)
                return
            if received > _WS_BURST_BYTES + elapsed * _WS_PCM_BYTES_PER_S * _WS_RATE_SLACK:
                await websocket.close(code=1008)  # faster than any real microphone
                return
            if len(buffer) < score_cap_bytes:
                buffer.extend(data)
            if verdict_final or received < next_score_at:
                continue

            # Coalesce overdue milestones: score the freshest prefix once instead
            # of replaying every stale 2s boundary after a burst or a slow pass.
            scored_to = min(received, score_cap_bytes, len(buffer)) & ~1
            audio = (
                np.frombuffer(buffer, dtype=np.int16, count=scored_to // 2).astype(np.float32)
                / 32768.0
            )
            # to_thread: model inference must not block WS keepalives
            (
                label,
                confidence,
                used_model,
                seconds_analyzed,
                wav2vec_result,
                aasist_result,
                wav2vec2_v2_result,
            ) = await asyncio.to_thread(
                _detect_ssl_array,
                audio,
                16000,
            )
            verdict_final = scored_to >= score_cap_bytes

            event = StreamDetectionEvent(
                timestamp_ms=time.time() * 1000,
                window_id=window_id,
                label=label,
                confidence=confidence,
                model=used_model,
                final=verdict_final,
                seconds_analyzed=seconds_analyzed,
                wav2vec2=wav2vec_result,
                aasist=aasist_result,
                wav2vec2_v2=wav2vec2_v2_result,
            )
            await websocket.send_json(event.model_dump())
            window_id += 1
            next_score_at = scored_to + stride_bytes

    except WebSocketDisconnect:
        pass
    finally:
        _stream_budget.release()


def _verify_twilio_signature(websocket: WebSocket) -> bool:
    """Validate Twilio's X-Twilio-Signature on the WebSocket handshake.

    Twilio signs every request as base64(HMAC-SHA1(auth_token, public URL)).
    With TWILIO_AUTH_TOKEN unset the bridge stays open in development but is
    refused in production â€” an unauthenticated endpoint would let anyone burn
    model inference.
    """
    import base64
    import hmac

    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    if not token:
        return os.environ.get("VG_ENV", "development") != "production"
    signature = websocket.headers.get("x-twilio-signature", "")
    if not signature:
        return False
    # Rebuild the public URL Twilio signed (nginx preserves Host + sets
    # X-Forwarded-Proto; Twilio Media Streams always connects over wss).
    proto = websocket.headers.get("x-forwarded-proto", websocket.url.scheme)
    scheme = "wss" if proto in ("https", "wss") else "ws"
    host = websocket.headers.get("host", websocket.url.netloc)
    url = f"{scheme}://{host}{websocket.url.path}"
    if websocket.url.query:
        url += f"?{websocket.url.query}"
    expected = base64.b64encode(
        hmac.new(token.encode(), url.encode(), hashlib.sha1).digest()  # noqa: S324 â€” Twilio's scheme is HMAC-SHA1
    ).decode()
    return hmac.compare_digest(expected, signature)


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """Twilio Media Stream WebSocket bridge.

    Receives Î¼-law encoded 8kHz audio from Twilio and runs detection.
    Authenticated via X-Twilio-Signature when TWILIO_AUTH_TOKEN is set.
    """
    if not _verify_twilio_signature(websocket):
        await websocket.close(code=1008)  # rejects the handshake with HTTP 403
        return
    await websocket.accept()
    if not _stream_budget.acquire():
        await websocket.close(code=1013)  # all inference slots busy
        return
    try:
        from voiceguard.voip.twilio_bridge import TwilioStreamHandler

        handler = TwilioStreamHandler()
        await handler.handle(websocket)
    except WebSocketDisconnect:
        pass
    except ImportError:
        await websocket.close(code=1011)
    finally:
        _stream_budget.release()


@app.post("/watermark/verify", response_model=WatermarkVerifyResult, tags=["synthesis"])
@limiter.limit("30/minute")
async def watermark_verify(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    watermark_id: str = Form(""),
    _user: str = Depends(get_current_user),
):
    """Verify the provenance of an audio file â€” the read side of /synthesize.

    Checks two independent marks: the keyed spectral watermark (only when the
    `watermark_id` returned by /synthesize is supplied) and the embedded C2PA
    manifest (cryptographic provenance, no key needed). Closes the
    Generate â†’ Detect loop: anything VoiceGuard synthesises can be proven so.
    """
    from voiceguard.watermark import c2pa_sign
    from voiceguard.watermark.c2pa_watermark import detect as wm_detect

    path, _ = await save_upload(file)
    background_tasks.add_task(pdpl_auto_delete, path)

    result = WatermarkVerifyResult()

    if watermark_id:
        data, sr = _read_audio(path)
        detected, corr = wm_detect(data, sr=sr, watermark_id=watermark_id)
        result.spectral_checked = True
        result.spectral_detected = bool(detected)
        result.spectral_correlation = round(float(corr), 6)

    c2pa = c2pa_sign.verify_file(path)
    result.c2pa_has_manifest = bool(c2pa.get("has_manifest"))
    result.c2pa_validation_state = c2pa.get("validation_state")
    result.c2pa_ai_generated = c2pa.get("ai_generated")
    result.c2pa_software_agent = c2pa.get("software_agent")

    if result.spectral_detected:
        result.verdict = "voiceguard-generated"
    elif result.c2pa_ai_generated:
        result.verdict = "ai-generated"
    elif result.c2pa_has_manifest:
        result.verdict = "unknown"
    else:
        result.verdict = "no-provenance-found"

    logger.info(
        "watermark/verify user=%s spectral=%s c2pa=%s verdict=%s",
        _user,
        result.spectral_detected if result.spectral_checked else "skipped",
        result.c2pa_has_manifest,
        result.verdict,
    )
    return result


@app.post("/explain", response_model=ExplanationResult, tags=["detection"])
@limiter.limit("20/minute")
async def explain(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: ModelType = ModelType.xls_r_aasist,
    _user: str = Depends(get_current_user),
):
    """Return attribution for an uploaded audio file.

    Shows which time segments (10ms bins) drove the model's fake/real decision.
    SSL models (wav2vec2, wavlm_base_plus, wav2vec2_large, aasist, dsfnet*) use
    Integrated Gradients; the classical model uses occlusion (silence one
    segment at a time and measure the probability drop).
    """
    path, _ = await save_upload(file)
    background_tasks.add_task(pdpl_auto_delete, path)

    
    

@app.get("/models", tags=["ops"])
async def models_list():
    """List all registered model keys and their checkpoint availability."""
    return registry.status()


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    status_map = registry.status()
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded={k: v["available"] for k, v in status_map.items()},
    )


# â”€â”€ Internal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def verify_token_ws(token: str) -> str:
    """Verify JWT token for WebSocket connections."""
    from voiceguard.api.auth import verify_token

    if not token:
        raise HTTPException(status_code=401, detail="Token required")
    return verify_token(token)


def _detect_classical_array(audio: np.ndarray, sr: int) -> tuple[str, float]:
    from voiceguard.features.extractor import extract_features

    features = extract_features(audio, sr)
    detector = registry.load("classical")
    if detector is None:
        return "real", 0.5
    return detector.predict_features(features)


def _detect_ssl_array(
    audio: np.ndarray, sr: int, model_key: str = "wav2vec2_spoof"
) -> tuple[str, float, str, float | None, dict | None, dict | None, dict | None]:
    """
    Live-stream voice deepfake detection.

    Uses the production three-model ensemble:
    Wav2Vec2 Spoof (30%) + Official AASIST (30%) + Wav2Vec2 v2 (40%).
    """
    try:
        import torch
        import torchaudio

        wav = torch.as_tensor(
            np.asarray(audio, dtype=np.float32)
        ).reshape(1, -1)

        if sr != 16000:
            wav = torchaudio.functional.resample(
                wav,
                sr,
                16000,
            )

        (
            label,
            confidence,
            seconds_analyzed,
            wav2vec_result,
            aasist_result,
            wav2vec2_v2_result,
        ) = _detect_ensemble_tensor(wav)

        logger.info(
            "LIVE ENSEMBLE: wav2vec2_fake=%.4f "
            "aasist_fake=%.4f wav2vec2_v2_fake=%.4f "
            "label=%s confidence=%.4f",
            wav2vec_result["fake_probability"],
            aasist_result["fake_probability"],
            wav2vec2_v2_result["fake_probability"],
            label,
            confidence,
        )

        return (
            label,
            confidence,
            "ensemble",
            seconds_analyzed,
            wav2vec_result,
            aasist_result,
            wav2vec2_v2_result,
        )

    except Exception:
        logger.exception("Live ensemble detection failed")

        # Never silently classify failed inference as REAL.
        return "unknown", 0.0, "ensemble", None, None, None, None


