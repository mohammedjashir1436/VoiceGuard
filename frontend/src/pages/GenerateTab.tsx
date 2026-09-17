import { useEffect, useMemo, useState } from 'react'
import {
  getEngines,
  synthesize,
  type EngineInfo,
} from '../services/synthesisService'
import {
  detectAudio,
  type DetectionResult,
} from '../services/detectionService'
import { apiFetch, ApiError } from '../config/apiConfig'

export default function GenerateTab() {
  const [text, setText] = useState('')
  const [engines, setEngines] = useState<EngineInfo[]>([])
  const [engineName, setEngineName] = useState('kokoro')
  const [voice, setVoice] = useState('af_heart')
  const [language, setLanguage] = useState('en')

  const [reference, setReference] = useState<File | null>(null)
  const [consent, setConsent] = useState(false)

  const [loading, setLoading] = useState(false)
  const [testing, setTesting] = useState(false)

  const [error, setError] = useState<string | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [watermarkId, setWatermarkId] = useState<string | null>(null)
  const [verdict, setVerdict] = useState<DetectionResult | null>(null)

  useEffect(() => {
    getEngines()
      .then((es) => {
        setEngines(es)

        const first = es.find((e) => e.available) ?? es[0]

        if (first) {
          setEngineName(first.name)

          if (first.preset_voices[0]) {
            setVoice(first.preset_voices[0])
          }

          if (first.languages[0]) {
            setLanguage(first.languages[0])
          }
        }
      })
      .catch(() => {
        setEngines([])
      })
  }, [])

  const engine = useMemo(
    () => engines.find((e) => e.name === engineName),
    [engines, engineName],
  )

  const needsRef = !!engine?.requires_reference

  const canSubmit =
    !!text.trim() &&
    !loading &&
    (!needsRef || (!!reference && consent))

  const handleSynthesize = async () => {
    if (!canSubmit) return

    setLoading(true)
    setError(null)
    setAudioUrl(null)
    setWatermarkId(null)
    setVerdict(null)

    try {
      const data = await synthesize({
        text,
        engine: engineName,
        voice,
        language,
        reference,
        consent,
      })

      setAudioUrl(data.audio_url)
      setWatermarkId(data.watermark_id ?? null)
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          e.status === 501
            ? `The "${engine?.label ?? engineName}" engine is not installed on this instance.`
            : e.message,
        )
      } else {
        setError(
          'Could not reach the synthesis API. Please try again in a moment.',
        )
      }
    } finally {
      setLoading(false)
    }
  }

  const handleTestDetector = async () => {
    if (!audioUrl) return

    setTesting(true)
    setError(null)
    setVerdict(null)

    try {
      const audioPath = audioUrl.replace(/^\/api/, '')

      const res = await apiFetch(audioPath)

      if (!res.ok) {
        throw new Error(
          `Failed to download generated audio: ${res.status}`,
        )
      }

      const blob = await res.blob()

      const file = new File([blob], 'generated-voice.wav', {
        type: blob.type || 'audio/wav',
      })

      /*
       * Production VoiceGuard detector:
       *
       * Wav2Vec2 Spoof = 60%
       * Official AASIST = 40%
       *
       * The backend performs the actual ensemble calculation.
       */
      const result = await detectAudio(
        file,
        false,
        'ensemble',
      )

      setVerdict(result)
   } catch (e) {
  console.error(
    'VoiceGuard: generated audio detector failed:',
    e,
  )

  if (e instanceof ApiError) {
    setError(`Detector error (${e.status}): ${e.message}`)
  } else if (e instanceof Error) {
    setError(`Detector error: ${e.message}`)
  } else {
    setError('Detector returned an unexpected response.')
  }
}finally {
      setTesting(false)
    }
  }

  const wavFake = verdict?.wav2vec2_spoof?.fake_probability
  const aasistFake = verdict?.aasist?.fake_probability

  const ensembleFake =
    typeof wavFake === 'number' && typeof aasistFake === 'number'
      ? wavFake * 0.6 + aasistFake * 0.4
      : verdict
        ? verdict.label === 'fake'
          ? verdict.confidence
          : 1 - verdict.confidence
        : null

  const riskScore =
    typeof ensembleFake === 'number'
      ? Math.round(ensembleFake * 100)
      : null

  const getRiskLabel = () => {
    if (riskScore === null) {
      return 'NOT TESTED'
    }

    if (riskScore >= 75) {
      return 'CRITICAL RISK'
    }

    if (riskScore >= 60) {
      return 'HIGH RISK'
    }

    if (riskScore >= 30) {
      return 'SUSPICIOUS'
    }

    return 'LOW RISK'
  }

  const getRiskClass = () => {
    if (riskScore === null) {
      return 'text-gray-400'
    }

    if (riskScore >= 60) {
      return 'text-red-400'
    }

    if (riskScore >= 30) {
      return 'text-yellow-400'
    }

    return 'text-green-400'
  }

  const handleEngineChange = (newEngineName: string) => {
    setEngineName(newEngineName)
    setReference(null)
    setConsent(false)
    setVerdict(null)
    setError(null)

    const selected = engines.find(
      (item) => item.name === newEngineName,
    )

    if (selected?.preset_voices[0]) {
      setVoice(selected.preset_voices[0])
    }

    if (selected?.languages[0]) {
      setLanguage(selected.languages[0])
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="mb-1 text-xl font-semibold text-white">
          Synthesise Voice
        </h2>

        <p className="text-sm text-gray-400">
          Generate synthetic speech using preset voices or authorised
          zero-shot voice cloning. Generated clips are watermarked and
          can be tested independently with the VoiceGuard ensemble detector.
        </p>
      </div>

      {/* Text */}
      <div>
        <label className="mb-2 block text-sm text-gray-400">
          Text to synthesise
        </label>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          maxLength={2000}
          placeholder="Enter text here (max 2000 characters)…"
          className="w-full resize-none rounded-lg border border-gray-700 bg-gray-800 p-3 text-gray-100 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
        />

        <p className="mt-1 text-right text-xs text-gray-600">
          {text.length}/2000
        </p>
      </div>

      {/* Engine + Voice */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs text-gray-500">
            Engine
          </label>

          <select
            value={engineName}
            onChange={(e) =>
              handleEngineChange(e.target.value)
            }
            className="w-full rounded-lg border border-gray-700 bg-gray-800 p-2 text-sm text-gray-300"
          >
            {engines.map((item) => (
              <option
                key={item.name}
                value={item.name}
                disabled={!item.available}
              >
                {item.label}
                {item.available ? '' : ' — not installed'}
              </option>
            ))}
          </select>
        </div>

        {engine && engine.preset_voices.length > 0 ? (
          <div>
            <label className="mb-1 block text-xs text-gray-500">
              Voice
            </label>

            <select
              value={voice}
              onChange={(e) => setVoice(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 p-2 text-sm text-gray-300"
            >
              {engine.preset_voices.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div>
            <label className="mb-1 block text-xs text-gray-500">
              Language
            </label>

            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 p-2 text-sm text-gray-300"
            >
              {(engine?.languages ?? ['en']).map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Clone Voice */}
      {needsRef && (
        <div className="space-y-4 rounded-xl border border-indigo-900/60 bg-indigo-950/20 p-5">
          <div>
            <div className="text-sm font-semibold text-white">
              Clone Voice
            </div>

            <p className="mt-1 text-xs text-gray-400">
              Upload at least 3 seconds of authorised reference speech.
              The system uses it only to generate the requested test clip.
            </p>
          </div>

          <div>
            <label className="mb-2 block text-sm text-gray-300">
              Reference voice
            </label>

            <input
              type="file"
              accept="audio/*"
              onChange={(e) => {
                setReference(
                  e.target.files?.[0] ?? null,
                )
                setVerdict(null)
              }}
              className="block w-full text-sm text-gray-400 file:mr-3 file:rounded-lg file:border-0 file:bg-indigo-600 file:px-4 file:py-2 file:text-white hover:file:bg-indigo-700"
            />

            {reference && (
              <p className="mt-2 text-xs text-green-400">
                ✓ Reference selected: {reference.name}
              </p>
            )}
          </div>

          <label className="flex items-start gap-2 text-xs text-gray-400">
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) =>
                setConsent(e.target.checked)
              }
              className="mt-0.5"
            />

            <span>
              I am authorised to clone this voice and will use the
              watermarked output only for testing and research, not
              impersonation.
            </span>
          </label>
        </div>
      )}

      {/* Generate */}
      <button
        onClick={handleSynthesize}
        disabled={!canSubmit}
        className="w-full rounded-lg bg-indigo-600 px-6 py-3 font-medium text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-gray-700"
      >
        {loading
          ? 'Generating…'
          : needsRef
            ? 'Clone & Synthesise'
            : 'Synthesise'}
      </button>

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-yellow-700 bg-yellow-900/40 p-4 text-sm text-yellow-300">
          {error}
        </div>
      )}

      {/* Generated Audio */}
      {audioUrl && (
        <div className="space-y-5 rounded-xl border border-gray-800 bg-gray-800 p-6">
          <div>
            <p className="font-medium text-white">
              Generated Voice
            </p>

            <p className="mt-1 text-xs text-gray-500">
              Synthetic audio generated by VoiceGuard.
            </p>
          </div>

          <audio
            controls
            src={audioUrl}
            className="w-full"
          />

          {watermarkId && (
            <div className="flex items-center gap-2 rounded-lg border border-green-900 bg-green-950/20 p-3 text-xs text-green-400">
              <span>✓</span>

              <span>
                Spectral watermark embedded · ID: {watermarkId}
              </span>
            </div>
          )}

          {/* Detector */}
          <div className="border-t border-gray-700 pt-5">
            <div className="mb-3">
              <p className="font-medium text-white">
                VoiceGuard Detection
              </p>

              <p className="mt-1 text-xs text-gray-500">
                Test this generated clip using the production
                Wav2Vec2 + Official AASIST ensemble.
              </p>
            </div>

            <button
              onClick={handleTestDetector}
              disabled={testing}
              className="rounded-lg bg-gray-700 px-4 py-2 text-sm font-medium text-gray-100 transition-colors hover:bg-gray-600 disabled:opacity-60"
            >
              {testing
                ? 'Testing with ensemble…'
                : '🔎 Test against detector'}
            </button>
          </div>

          {/* Detection Result */}
          {verdict && (
            <div className="space-y-4">
              {/* Main result */}
              <div
                className={`rounded-xl border p-5 ${
                  verdict.label === 'fake'
                    ? 'border-red-800 bg-red-950/40'
                    : 'border-green-800 bg-green-950/30'
                }`}
              >
                <div className="text-center">
                  <div className="text-xs uppercase tracking-wider text-gray-500">
                    Impersonation Risk
                  </div>

                  <div
                    className={`mt-2 text-5xl font-bold ${getRiskClass()}`}
                  >
                    {riskScore !== null
                      ? `${riskScore}%`
                      : '—'}
                  </div>

                  <div
                    className={`mt-2 text-sm font-semibold ${getRiskClass()}`}
                  >
                    {getRiskLabel()}
                  </div>

                  <div className="mt-2 text-sm text-gray-300">
                    {verdict.label === 'fake'
                      ? 'AI-GENERATED / CLONED VOICE'
                      : 'LIKELY REAL VOICE'}
                  </div>
                </div>
              </div>

              {/* Model scores */}
              <div className="space-y-3">
                <div className="flex items-center justify-between rounded-xl border border-gray-700 bg-gray-950 p-4">
                  <div>
                    <div className="text-sm font-medium text-gray-300">
                      Wav2Vec2 Spoof
                    </div>

                    <div className="mt-1 text-xs text-gray-500">
                      Weight: 60%
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="font-semibold text-white">
                      {typeof wavFake === 'number'
                        ? `${Math.round(wavFake * 100)}% fake`
                        : 'Unavailable'}
                    </div>

                    {verdict.wav2vec2_spoof && (
                      <div className="text-xs text-gray-500">
                        {verdict.wav2vec2_spoof.label.toUpperCase()}
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-xl border border-gray-700 bg-gray-950 p-4">
                  <div>
                    <div className="text-sm font-medium text-gray-300">
                      Official AASIST
                    </div>

                    <div className="mt-1 text-xs text-gray-500">
                      Weight: 40%
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="font-semibold text-white">
                      {typeof aasistFake === 'number'
                        ? `${Math.round(aasistFake * 100)}% fake`
                        : 'Unavailable'}
                    </div>

                    {verdict.aasist && (
                      <div className="text-xs text-gray-500">
                        {verdict.aasist.label.toUpperCase()}
                      </div>
                    )}
                  </div>
                </div>

                {/* Ensemble */}
                <div className="rounded-xl border border-indigo-900/60 bg-indigo-950/20 p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium text-gray-200">
                        Ensemble Result
                      </div>

                      <div className="mt-1 text-xs text-gray-500">
                        Wav2Vec2 60% + Official AASIST 40%
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-xl font-bold text-indigo-300">
                        {riskScore !== null
                          ? `${riskScore}%`
                          : '—'}
                      </div>

                      <div className="text-xs text-gray-500">
                        fake probability
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recommendation */}
              {riskScore !== null && riskScore >= 60 && (
                <div className="rounded-xl border border-red-800 bg-red-950/40 p-4">
                  <div className="flex gap-3">
                    <div className="text-2xl">🚨</div>

                    <div>
                      <div className="font-semibold text-red-300">
                        High AI-Voice Risk Detected
                      </div>

                      <p className="mt-1 text-sm text-red-200/80">
                        The generated clip shows a high probability
                        of synthetic or cloned speech.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {riskScore !== null && riskScore < 60 && (
                <div className="rounded-xl border border-yellow-900 bg-yellow-950/20 p-4">
                  <div className="text-sm text-yellow-200">
                    Detection completed. The ensemble risk is{' '}
                    <b>{riskScore}%</b>.
                  </div>
                </div>
              )}

              {/* Detector metadata */}
              <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-gray-500">
                <span>
                  Model: {verdict.model}
                </span>

                <span>
                  Confidence:{' '}
                  {Math.round(verdict.confidence * 100)}%
                </span>

                {verdict.latency_ms !== undefined && (
                  <span>
                    Latency: {verdict.latency_ms} ms
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}