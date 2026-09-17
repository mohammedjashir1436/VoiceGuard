/** Deepfake detection API calls (POST /detect). */
import { apiFetch, ApiError } from '../config/apiConfig'

export type AttributionSegment = {
  start_s: number
  end_s: number
  importance: number
}

export type Explanation = {
  method: string
  baseline: string
  target_class: number
  frame_duration_ms: number
  attribution_frames: number[]
  top_segments: AttributionSegment[]
  narrative?: string | null
}

export type DetectionResult = {
  label: 'real' | 'fake'
  confidence: number
  model: string
  latency_ms: number
  audio_hash: string

  // Ensemble model details
  wav2vec2_spoof?: {
    label: 'real' | 'fake'
    confidence: number
    fake_probability: number
    weight?: number
  } | null

  aasist?: {
    label: 'real' | 'fake'
    confidence: number
    fake_probability: number
    weight?: number
  } | null

  explanation?: Explanation | null
}

export type ModelInfo = {
  key: string
  available: boolean
}

/** Human-readable labels for the detector model keys. */
export const MODEL_LABELS: Record<string, string> = {
  // Recommended production detector
  ensemble: 'Ensemble (Wav2Vec2 Spoof 30% + Official AASIST 30% + Wav2Vec2 v2 40%)',

  // Individual verified detectors
  wav2vec2_spoof: 'Wav2Vec2 Spoof',
  aasist: 'Official AASIST',

  // Existing project detectors
  xls_r_aasist: 'XLS-R-300M + AASIST "v9c" (flagship)',
  classical: 'Classical (SM2026 XGBoost)',
  xls_r: 'XLS-R-300M',
  wav2vec2_large: 'wav2vec2-large',
  wav2vec2: 'wav2vec2-base',
  wavlm_large: 'WavLM-large',
  wavlm_base_plus: 'WavLM-base+',
  dsfnet_v2: 'DSFNet v2',
  dsfnet: 'DSFNet',
}

/**
 * Fetch the detector registry and its per-model checkpoint availability.
 *
 * The ensemble is always considered available because it is composed
 * of the two detectors that are loaded separately by the backend.
 */
export const getModels = async (): Promise<ModelInfo[]> => {
  const res = await apiFetch('/models')

  if (!res.ok) {
    throw new ApiError(res.status, `Error ${res.status}`)
  }

  const raw = (await res.json()) as Record<
    string,
    { available: boolean }
  >

  const models = Object.entries(raw).map(([key, value]) => ({
    key,
    available: !!value.available,
  }))

  // Backend registry does not expose "ensemble" as a physical checkpoint.
  // Add it explicitly because the ensemble is created from the two loaded
  // detectors: Wav2Vec2 Spoof + Official AASIST.
  const hasEnsemble = models.some(
    (model) => model.key === 'ensemble',
  )

  if (!hasEnsemble) {
    models.unshift({
      key: 'ensemble',
      available: true,
    })
  }

  return models
}

/**
 * Upload an audio file for deepfake detection.
 *
 * Default detector:
 *   ensemble
 *
 * Production ensemble:
 *   60% Wav2Vec2 Spoof
 *   40% Official AASIST
 *
 * @param file audio file to analyze
 * @param explain request attribution (slower)
 * @param model detector model key (default 'ensemble')
 *
 * @throws ApiError with the HTTP status:
 *   401 unauthenticated
 *   422 unusable audio
 *   503 model unavailable
 */
export const detectAudio = async (
  file: File,
  explain = false,
  model = 'ensemble',
): Promise<DetectionResult> => {
  const form = new FormData()
  form.append('file', file)

  const res = await apiFetch(
    `/detect?explain=${explain ? 'true' : 'false'}&model=${encodeURIComponent(model)}`,
    {
      method: 'POST',
      body: form,
    },
  )

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))

    throw new ApiError(
      res.status,
      body.detail || `Error ${res.status}`,
    )
  }

  return res.json() as Promise<DetectionResult>
}

/**
 * Feedback submitted by the admin after a detection.
 *
 * The feedback starts as "unverified".
 * It must be independently verified before
 * the audio can enter the training dataset.
 */
export type FeedbackResponse = {
  feedback_id: string
  audio_hash: string
  predicted_label: string
  predicted_confidence: number
  model: string
  feedback: 'correct' | 'incorrect'
  verification_status: string
}

/**
 * Submit admin feedback for a detection result.
 *
 * This does NOT add the audio to the training dataset.
 * The feedback remains unverified until verifyFeedback()
 * is called.
 */
export const submitFeedback = async (
  audioHash: string,
  feedback: 'correct' | 'incorrect',
): Promise<FeedbackResponse> => {
  const res = await apiFetch('/feedback', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      audio_hash: audioHash,
      feedback,
    }),
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))

    throw new ApiError(
      res.status,
      body.detail || `Error ${res.status}`,
    )
  }

  return res.json() as Promise<FeedbackResponse>
}

/**
 * Verify the true label of an admin feedback record.
 *
 * This is the second step in the feedback pipeline:
 *
 *   feedback → verification → promotion
 *
 * Only a verified REAL/FAKE label can be promoted
 * into the training dataset.
 */
export type FeedbackVerifyResponse = {
  feedback_id: string
  audio_hash: string
  predicted_label: string
  predicted_confidence: number
  model: string
  feedback: string
  verification_status: string
  verified_label: string
  verified_by?: string
  verified_at?: number
}

export const verifyFeedback = async (
  feedbackId: string,
  verifiedLabel: 'real' | 'fake',
): Promise<FeedbackVerifyResponse> => {
  const res = await apiFetch(
    `/feedback/${encodeURIComponent(feedbackId)}/verify`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        verified_label: verifiedLabel,
      }),
    },
  )

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))

    throw new ApiError(
      res.status,
      body.detail || `Error ${res.status}`,
    )
  }

  return res.json() as Promise<FeedbackVerifyResponse>
}

/**
 * Promote a verified feedback sample into the training dataset.
 *
 * The backend checks that:
 * 1. The feedback exists.
 * 2. The feedback has been independently verified.
 * 3. The verified label is REAL or FAKE.
 * 4. The uploaded file hash exactly matches the
 *    original detected audio hash.
 *
 * Dataset destination:
 *
 *   data/verified_training/real/
 *   data/verified_training/fake/
 */
export type PromoteFeedbackResponse = {
  status: string
  feedback_id: string
  audio_hash: string
  verified_label: string
  dataset_path: string
}

export const promoteFeedback = async (
  feedbackId: string,
  file: File,
): Promise<PromoteFeedbackResponse> => {
  const form = new FormData()
  form.append('file', file)

  const res = await apiFetch(
    `/feedback/${encodeURIComponent(feedbackId)}/promote`,
    {
      method: 'POST',
      body: form,
    },
  )

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))

    throw new ApiError(
      res.status,
      body.detail || `Error ${res.status}`,
    )
  }

  return res.json() as Promise<PromoteFeedbackResponse>
}