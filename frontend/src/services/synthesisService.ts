/** Voice synthesis API calls (GET /synthesis/engines, POST /synthesize). */
import { apiFetch, ApiError } from '../config/apiConfig'

export type EngineInfo = {
  name: string
  label: string
  requires_reference: boolean
  available: boolean
  preset_voices: string[]
  languages: string[]
  description: string
}

export type SynthesisResult = {
  audio_url: string
  watermark_id?: string
  synthesis_latency_ms?: number
  engine?: string
}

export type SynthesizeParams = {
  text: string
  engine: string
  voice?: string
  language?: string
  reference?: File | null
  consent?: boolean
}

type SynthesisJobResponse = {
  job_id: string
  status: 'queued' | 'processing' | 'completed' | 'failed' | string
  engine?: string
  audio_url?: string | null
  watermark_id?: string | null
  synthesis_latency_ms?: number | null
  error?: string | null
}

/** List synthesis engines and their availability. */
export const getEngines = async (): Promise<EngineInfo[]> => {
  const res = await apiFetch('/synthesis/engines')

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(
      res.status,
      body.detail || `Error ${res.status}`,
    )
  }

  return res.json() as Promise<EngineInfo[]>
}

/**
 * Poll an asynchronous synthesis job until it completes.
 *
 * IndexTTS-2 can take a very long time on CPU, so the browser must not
 * expect the POST /synthesize request itself to stay open.
 */
const waitForSynthesisJob = async (
  jobId: string,
  engine: string,
): Promise<SynthesisResult> => {
  const pollIntervalMs = 3000

  // Safety timeout: 2 hours.
  // This is intentionally long because IndexTTS-2 CPU synthesis can
  // take many minutes for a short reference-based generation.
  const maxWaitMs = 2 * 60 * 60 * 1000

  const startedAt = Date.now()

  while (Date.now() - startedAt < maxWaitMs) {
    await new Promise((resolve) =>
      setTimeout(resolve, pollIntervalMs),
    )

    const res = await apiFetch(`/synthesis/jobs/${encodeURIComponent(jobId)}`)

    if (!res.ok) {
      const body = await res.json().catch(() => ({}))

      throw new ApiError(
        res.status,
        body.detail || `Error ${res.status}`,
      )
    }

    const job = (await res.json()) as SynthesisJobResponse

    if (job.status === 'completed') {
      if (!job.audio_url) {
        throw new ApiError(
          500,
          'Synthesis completed but no audio URL was returned.',
        )
      }

      return {
        audio_url: job.audio_url,
        watermark_id: job.watermark_id ?? undefined,
        synthesis_latency_ms:
          job.synthesis_latency_ms ?? undefined,
        engine: job.engine || engine,
      }
    }

    if (job.status === 'failed') {
      throw new ApiError(
        500,
        job.error || `Synthesis failed for engine '${engine}'.`,
      )
    }

    // queued / processing:
    // Continue polling.
  }

  throw new ApiError(
    408,
    `Synthesis timed out while waiting for engine '${engine}'.`,
  )
}

/**
 * Synthesise speech (preset TTS or zero-shot voice cloning).
 *
 * The backend now queues the synthesis and immediately returns a job ID.
 * This function keeps the same public return type as before, so existing
 * GenerateTab code does not need to change.
 *
 * For slow engines such as IndexTTS-2 on CPU, this function waits for the
 * background job to complete and then returns the final audio result.
 *
 * @throws ApiError — backend/API errors, failed synthesis, or timeout.
 */
export const synthesize = async (
  p: SynthesizeParams,
): Promise<SynthesisResult> => {
  const form = new FormData()

  form.append('text', p.text)
  form.append('engine', p.engine)

  if (p.voice) {
    form.append('voice', p.voice)
  }

  if (p.language) {
    form.append('language', p.language)
  }

  form.append(
    'consent',
    p.consent ? 'true' : 'false',
  )

  if (p.reference) {
    form.append('reference', p.reference)
  }

  // Step 1:
  // Queue the synthesis job.
  const res = await apiFetch('/synthesize', {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))

    throw new ApiError(
      res.status,
      body.detail || `Error ${res.status}`,
    )
  }

  const queued = (await res.json()) as {
    job_id?: string
    status?: string
    engine?: string
  }

  if (!queued.job_id) {
    throw new ApiError(
      500,
      'Synthesis request was accepted but no job ID was returned.',
    )
  }

  // Step 2:
  // Wait for the background synthesis worker.
  return waitForSynthesisJob(
    queued.job_id,
    queued.engine || p.engine,
  )
}