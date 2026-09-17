import { useState, useRef, useEffect } from 'react'

import {
  detectAudio,
  getModels,
  MODEL_LABELS,
  submitFeedback,
  verifyFeedback,
  promoteFeedback,
  type DetectionResult,
  type Explanation,
  type ModelInfo,
} from '../services/detectionService'

import { generateReport } from '../services/forensicsService'
import { addHistory } from '../services/history'
import { ApiError } from '../config/apiConfig'

function ConfidenceGauge({
  confidence,
  label,
}: {
  confidence: number
  label: string
}) {
  const pct = Math.round(confidence * 100)
  const color =
    label === 'fake' ? 'text-red-400' : 'text-green-400'
  const ring =
    label === 'fake' ? 'border-red-500' : 'border-green-500'

  return (
    <div
      className={`flex flex-col items-center gap-2 p-6 rounded-xl border-2 ${ring} bg-gray-800`}
    >
      <span
        className={`text-5xl font-bold font-mono ${color}`}
      >
        {pct}%
      </span>

      <span
        className={`text-lg font-semibold uppercase tracking-widest ${color}`}
      >
        {label}
      </span>

      <span className="text-xs text-gray-500">
        confidence
      </span>
    </div>
  )
}

function AttributionView({
  explanation,
  label,
}: {
  explanation: Explanation
  label: string
}) {
  const {
    attribution_frames: frames,
    top_segments: segments,
    method,
    frame_duration_ms,
  } = explanation

  const isFake = label === 'fake'

  const barColor = isFake
    ? 'bg-rose-500'
    : 'bg-emerald-500'

  const accentText = isFake
    ? 'text-rose-300'
    : 'text-emerald-300'

  const totalS =
    (frames.length * (frame_duration_ms || 10)) / 1000

  const methodLabel =
    method === 'integrated_gradients'
      ? 'Integrated Gradients'
      : method === 'occlusion'
        ? 'Occlusion'
        : method

  const step = Math.max(
    1,
    Math.ceil(frames.length / 140),
  )

  const bars: number[] = []

  for (let i = 0; i < frames.length; i += step) {
    bars.push(
      Math.max(...frames.slice(i, i + step)),
    )
  }

  const top = [...segments]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 5)

  const strongest = top[0]

  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <p className="text-sm font-medium text-gray-200">
          Why this verdict — moment-by-moment
        </p>

        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-gray-900 border border-gray-700 text-gray-400">
          {methodLabel} · {frame_duration_ms || 10}ms frames
        </span>
      </div>

      {explanation.narrative && (
        <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3">
          <p className="text-[11px] uppercase tracking-wider text-indigo-300/80 mb-1 flex items-center gap-1.5">
            <span aria-hidden>✨</span>
            AI analysis
          </p>

          <p className="text-sm text-gray-200 leading-relaxed">
            {explanation.narrative}
          </p>
        </div>
      )}

      {strongest && (
        <p className="text-sm text-gray-400">
          The detector weighed the whole clip and leaned{' '}
          <span
            className={`font-semibold ${accentText}`}
          >
            {label.toUpperCase()}
          </span>{' '}
          most strongly on the audio around{' '}
          <span className="font-mono text-gray-200">
            {strongest.start_s.toFixed(1)}s–
            {strongest.end_s.toFixed(1)}s
          </span>
          {top.length > 1 && (
            <>
              , with further support at{' '}
              <span className="font-mono text-gray-200">
                {top[1].start_s.toFixed(1)}–
                {top[1].end_s.toFixed(1)}s
              </span>
            </>
          )}
          .
        </p>
      )}

      <div>
        <div className="flex items-end gap-px h-20">
          {bars.map((v, i) => (
            <div
              key={i}
              className={`flex-1 rounded-sm ${barColor}`}
              style={{
                height: `${Math.max(3, v * 100)}%`,
                opacity: 0.35 + v * 0.65,
              }}
              title={`importance ${v.toFixed(2)}`}
            />
          ))}
        </div>

        <div className="flex justify-between text-[10px] font-mono text-gray-600 mt-1">
          <span>0.0s</span>
          <span>{(totalS / 2).toFixed(1)}s</span>
          <span>{totalS.toFixed(1)}s</span>
        </div>
      </div>

      {top.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[11px] uppercase tracking-wider text-gray-500">
            Most influential moments
          </p>

          {top.map((s, i) => (
            <div
              key={i}
              className="flex items-center gap-3"
            >
              <span className="text-xs font-mono text-gray-500 w-4 text-right">
                {i + 1}
              </span>

              <span className="text-xs font-mono text-gray-300 w-24 shrink-0">
                {s.start_s.toFixed(1)}–
                {s.end_s.toFixed(1)}s
              </span>

              <div className="flex-1 h-2 rounded-full bg-gray-900 overflow-hidden">
                <div
                  className={`h-full ${barColor}`}
                  style={{
                    width: `${Math.max(
                      4,
                      s.importance * 100,
                    )}%`,
                  }}
                />
              </div>

              <span
                className={`text-xs font-mono ${accentText} w-10 text-right`}
              >
                {Math.round(
                  s.importance * 100,
                )}
                %
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="text-[11px] text-gray-600">
        {methodLabel === 'Integrated Gradients'
          ? 'Integrated Gradients traces the verdict back through the model to each 10ms frame — taller/brighter and higher-ranked = more influence.'
          : 'Occlusion silences each time window in turn and measures how much the verdict shifts — bigger shift = more influence.'}
      </p>
    </div>
  )
}

export default function DetectTab() {
  const [file, setFile] = useState<File | null>(null)

  const [result, setResult] =
    useState<DetectionResult | null>(null)

  const [loading, setLoading] = useState(false)

  const [explain, setExplain] = useState(false)

  const [error, setError] =
    useState<string | null>(null)

  const [reportBusy, setReportBusy] =
    useState(false)

  const [feedback, setFeedback] =
    useState<'correct' | 'incorrect' | null>(null)

  const [feedbackBusy, setFeedbackBusy] =
    useState(false)

  const [feedbackMessage, setFeedbackMessage] =
    useState<string | null>(null)

  const [feedbackId, setFeedbackId] =
    useState<string | null>(null)

  const [verifiedLabel, setVerifiedLabel] =
    useState<'real' | 'fake' | null>(null)

  const [promoteBusy, setPromoteBusy] =
    useState(false)

  const [promoteMessage, setPromoteMessage] =
    useState<string | null>(null)

  const [models, setModels] =
    useState<ModelInfo[]>([])

  const [model, setModel] =
    useState('classical')

  const fileRef =
    useRef<HTMLInputElement>(null)

  useEffect(() => {
    getModels()
      .then((ms) => {
        const order = Object.keys(MODEL_LABELS)

        ms.sort(
          (a, b) =>
            Number(b.available) -
              Number(a.available) ||
            order.indexOf(a.key) -
              order.indexOf(b.key),
        )

        setModels(ms)

        const first = ms.find(
          (m) => m.available,
        )

        if (first) {
          setModel(first.key)
        }
      })
      .catch(() => setModels([]))
  }, [])

  const handleUpload = async () => {
    if (!file) return

    setLoading(true)

    setError(null)
    setResult(null)

    setFeedback(null)
    setFeedbackId(null)
    setFeedbackMessage(null)

    setVerifiedLabel(null)
    setPromoteMessage(null)

    try {
      const data = await detectAudio(
        file,
        explain,
        model,
      )

      setResult(data)

      addHistory(file.name, data)
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          e.status === 401
            ? 'Not authenticated — use the Log in button (top-right) first.'
            : e.message,
        )
      } else {
        setError(
          'Could not reach the detection API. Please try again in a moment.',
        )
      }
    } finally {
      setLoading(false)
    }
  }

  const handleReport = async () => {
    if (!result) return

    setReportBusy(true)

    try {
      const report =
        await generateReport(result)

      window.open(
        report.report_url,
        '_blank',
      )
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.message
          : 'Could not generate the report.',
      )
    } finally {
      setReportBusy(false)
    }
  }

  const handleFeedback = async (
    value: 'correct' | 'incorrect',
  ) => {
    if (!result || feedbackBusy) return

    setFeedbackBusy(true)

    setFeedbackMessage(null)
    setPromoteMessage(null)
    setFeedbackId(null)
    setVerifiedLabel(null)

    try {
      const response =
        await submitFeedback(
          result.audio_hash,
          value,
        )

      setFeedback(value)

      setFeedbackId(
        response.feedback_id,
      )

      if (value === 'correct') {
        setVerifiedLabel(
          result.label === 'fake'
            ? 'fake'
            : 'real',
        )

        setFeedbackMessage(
          'Feedback recorded. Confirm the true label below to add this sample to the verified training dataset.',
        )
      } else {
        setFeedbackMessage(
          'Feedback recorded. Please select the actual label below before adding this sample to the verified training dataset.',
        )
      }
    } catch (e) {
      setFeedbackMessage(
        e instanceof ApiError
          ? e.message
          : 'Could not submit feedback. Please try again.',
      )
    } finally {
      setFeedbackBusy(false)
    }
  }

  const handlePromote = async () => {
    if (
      !feedbackId ||
      !file ||
      !verifiedLabel ||
      promoteBusy
    ) {
      return
    }

    setPromoteBusy(true)
    setPromoteMessage(null)

    try {
      /*
       * STEP 1
       *
       * Independently verify the true label.
       *
       * This changes the feedback record from:
       *
       *     unverified
       *
       * to:
       *
       *     verified
       *
       * The backend only allows admins to perform
       * this operation.
       */
      await verifyFeedback(
        feedbackId,
        verifiedLabel,
      )

      /*
       * STEP 2
       *
       * Promote the exact same audio file.
       *
       * The backend compares its SHA-256 hash with
       * the original detection hash before accepting it.
       */
      const response =
        await promoteFeedback(
          feedbackId,
          file,
        )

      if (
        response.status === 'promoted' ||
        response.status === 'already_present'
      ) {
        setPromoteMessage(
          response.status === 'already_present'
            ? `Already in the verified training dataset as ${verifiedLabel.toUpperCase()}.`
            : `Verified and added to the training dataset as ${verifiedLabel.toUpperCase()}.`,
        )
      } else {
        setPromoteMessage(
          `Dataset response: ${response.status}`,
        )
      }
    } catch (e) {
      setPromoteMessage(
        e instanceof ApiError
          ? e.message
          : 'Could not verify and add this sample to the training dataset.',
      )
    } finally {
      setPromoteBusy(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white mb-1">
          Upload Audio for Analysis
        </h2>

        <p className="text-sm text-gray-400">
          Supported formats: WAV, MP3, FLAC · Max 100MB ·
          Audio auto-deleted after 60s (PDPL)
        </p>
      </div>

      {/* Upload area */}
      <div
        className="border-2 border-dashed border-gray-700 rounded-xl p-10 text-center cursor-pointer hover:border-indigo-500 transition-colors"
        onClick={() =>
          fileRef.current?.click()
        }
      >
        <input
          ref={fileRef}
          type="file"
          accept=".wav,.mp3,.flac,.ogg,audio/*"
          className="hidden"
          onChange={(e) => {
            const selectedFile =
              e.target.files?.[0] ?? null

            setFile(selectedFile)

            setResult(null)

            setFeedback(null)
            setFeedbackId(null)
            setFeedbackMessage(null)

            setVerifiedLabel(null)
            setPromoteMessage(null)

            setError(null)
          }}
        />

        {file ? (
          <div className="space-y-1">
            <p className="text-white font-medium">
              {file.name}
            </p>

            <p className="text-gray-400 text-sm">
              {(file.size / 1024).toFixed(1)} KB
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="text-4xl">
              🎙️
            </div>

            <p className="text-gray-300">
              Click to select audio file
            </p>

            <p className="text-gray-500 text-sm">
              or drag and drop · use a few seconds
              of speech
            </p>
          </div>
        )}
      </div>

      {/* Model picker */}
      <div className="space-y-1">
        <label className="text-sm text-gray-400">
          Detector model
        </label>

        <select
          value={model}
          onChange={(e) =>
            setModel(e.target.value)
          }
          className="w-full bg-gray-800 border border-gray-700 text-gray-200 rounded-lg px-3 py-2 text-sm"
        >
          {models.map((m) => (
            <option
              key={m.key}
              value={m.key}
              disabled={!m.available}
            >
              {(MODEL_LABELS[m.key] ??
                m.key) +
                (m.available
                  ? ''
                  : ' — checkpoint unavailable')}
            </option>
          ))}
        </select>
      </div>

      {/* Options + submit */}
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-gray-400 select-none cursor-pointer">
          <input
            type="checkbox"
            checked={explain}
            onChange={(e) =>
              setExplain(e.target.checked)
            }
            className="accent-indigo-500"
          />

          Explain decision
        </label>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
        >
          {loading
            ? 'Analysing…'
            : 'Detect Deepfake'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-4">
          <ConfidenceGauge
            confidence={result.confidence}
            label={result.label}
          />

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {[
              {
                k: 'Model',
                v: result.model,
              },
              {
                k: 'Latency',
                v: `${result.latency_ms.toFixed(1)}ms`,
              },
              {
                k: 'Audio Hash',
                v:
                  result.audio_hash.slice(
                    0,
                    8,
                  ) + '…',
              },
            ].map(({ k, v }) => (
              <div
                key={k}
                className="bg-gray-800 rounded-lg p-3"
              >
                <p className="text-xs text-gray-500">
                  {k}
                </p>

                <p className="text-sm text-white font-mono truncate">
                  {v}
                </p>
              </div>
            ))}
          </div>

          {/* Verdict explainer */}
          <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
            <p className="text-sm font-medium text-gray-300 mb-1">
              {result.label === 'fake'
                ? 'This audio shows synthetic-speech artifacts.'
                : 'This audio is consistent with genuine human speech.'}
            </p>

            <p className="text-xs text-gray-500">
              Scored by the {result.model}
              detector. Uploaded audio is
              deleted within 60 seconds (PDPL).
            </p>
          </div>

          {/* User feedback */}
          <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
            <p className="text-sm font-medium text-gray-200 mb-1">
              Was this detection correct?
            </p>

            <p className="text-xs text-gray-500 mb-4">
              As an admin, you can verify the
              actual label and add the sample to
              the verified training dataset.
            </p>

            <div className="flex gap-3">
              <button
                onClick={() =>
                  handleFeedback('correct')
                }
                disabled={
                  feedbackBusy ||
                  feedback !== null
                }
                className={`flex-1 border rounded-lg py-2.5 px-4 text-sm font-medium transition-colors ${
                  feedback === 'correct'
                    ? 'border-green-500 bg-green-500/10 text-green-300'
                    : 'border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-700'
                } disabled:cursor-not-allowed disabled:opacity-60`}
              >
                ✓ Correct
              </button>

              <button
                onClick={() =>
                  handleFeedback('incorrect')
                }
                disabled={
                  feedbackBusy ||
                  feedback !== null
                }
                className={`flex-1 border rounded-lg py-2.5 px-4 text-sm font-medium transition-colors ${
                  feedback === 'incorrect'
                    ? 'border-orange-500 bg-orange-500/10 text-orange-300'
                    : 'border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-700'
                } disabled:cursor-not-allowed disabled:opacity-60`}
              >
                ✕ Incorrect
              </button>
            </div>

            {feedbackMessage && (
              <p className="text-xs text-green-400 mt-3">
                {feedbackMessage}
              </p>
            )}

            {/* Verification */}
            {feedbackId && (
              <div className="mt-5 border-t border-gray-700 pt-5 space-y-4">
                <div>
                  <p className="text-sm font-medium text-gray-200">
                    Verify the actual label
                  </p>

                  <p className="text-xs text-gray-500 mt-1">
                    Only the verified label will be
                    added to the training dataset.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() =>
                      setVerifiedLabel('real')
                    }
                    disabled={promoteBusy}
                    className={`border rounded-lg py-2.5 px-4 text-sm font-medium transition-colors ${
                      verifiedLabel === 'real'
                        ? 'border-green-500 bg-green-500/10 text-green-300'
                        : 'border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-700'
                    } disabled:opacity-60`}
                  >
                    REAL
                  </button>

                  <button
                    onClick={() =>
                      setVerifiedLabel('fake')
                    }
                    disabled={promoteBusy}
                    className={`border rounded-lg py-2.5 px-4 text-sm font-medium transition-colors ${
                      verifiedLabel === 'fake'
                        ? 'border-red-500 bg-red-500/10 text-red-300'
                        : 'border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-700'
                    } disabled:opacity-60`}
                  >
                    FAKE
                  </button>
                </div>

                <button
                  onClick={handlePromote}
                  disabled={
                    !verifiedLabel ||
                    promoteBusy
                  }
                  className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
                >
                  {promoteBusy
                    ? 'Verifying & Adding…'
                    : '✓ Verify & Add to Training Dataset'}
                </button>

                {promoteMessage && (
                  <div className="rounded-lg border border-emerald-700 bg-emerald-900/20 p-3">
                    <p className="text-sm text-emerald-300">
                      {promoteMessage}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Explainability */}
          {result.explanation && (
            <AttributionView
              explanation={
                result.explanation
              }
              label={result.label}
            />
          )}

          {/* Forensic report */}
          <button
            onClick={handleReport}
            disabled={reportBusy}
            className="w-full border border-gray-700 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 text-gray-200 font-medium py-2.5 px-6 rounded-lg transition-colors text-sm"
          >
            {reportBusy
              ? 'Generating…'
              : '⬇ Download forensic PDF report'}
          </button>
        </div>
      )}
    </div>
  )
}