import { useEffect, useRef, useState } from 'react'
import {
  Room,
  RoomEvent,
  ConnectionState,
  Track,
  LocalAudioTrack,
  type RemoteTrack,
} from 'livekit-client'
import {
  API_BASE,
  getToken,
  getWsUrl,
  apiFetch,
} from '../config/apiConfig'
import type { CallDetection, CallRecording, CallSession } from '../App'

type CallStatus =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'disconnecting'
  | 'error'

type DetectionModelResult = {
  label?: string
  confidence?: number
  fake_probability?: number
  weight?: number
}

type DetectionEvent = {
  timestamp_ms?: number
  window_id?: number
  label: string
  confidence: number
  model?: string | null
  final?: boolean
  seconds_analyzed?: number | null
  wav2vec2?: DetectionModelResult | null
  aasist?: DetectionModelResult | null
  wav2vec2_v2?: DetectionModelResult | null
  final_fake_probability?: number | null
}

type LiveKitTokenResponse = {
  token: string
  url: string
  room: string
  identity: string
}

type CallTabProps = {
  session: CallSession | null
  onAccept: () => void
  onReject: () => void
  onEndCall: () => void
  detection: CallDetection
  recording: CallRecording | null
}

export default function CallTab({
  session,
  onAccept,
  onReject,
  onEndCall,
  detection,
  recording,
}: CallTabProps) {
  const caller = session?.phone || '+91 98765 43210'
  const callerName = session?.callerName || 'Unknown Caller'
  const callPurpose = session?.purpose || 'General Call'
  const isSimulatedCall = Boolean(session)
  const isCallerDemo =
    new URLSearchParams(window.location.search).get('role') === 'caller'

  const [incoming, setIncoming] = useState(true)
  const [status, setStatus] = useState<CallStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  const [riskScore, setRiskScore] = useState<number | null>(null)
  const [verdict, setVerdict] = useState<
    'real' | 'fake' | 'unknown' | null
  >(null)
  const [secondsAnalyzed, setSecondsAnalyzed] = useState<number | null>(null)

  const [wav2vecResult, setWav2vecResult] =
    useState<DetectionModelResult | null>(null)
  const [aasistResult, setAasistResult] =
    useState<DetectionModelResult | null>(null)
  const [wav2vecV2Result, setWav2vecV2Result] =
    useState<DetectionModelResult | null>(null)

  const [analysisConnected, setAnalysisConnected] = useState(false)

  const roomRef = useRef<Room | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const remoteProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const remoteSourceRef = useRef<AudioNode | null>(null)
  const remoteGainRef = useRef<GainNode | null>(null)
  const remoteAudioElementRef = useRef<HTMLAudioElement | null>(null)
  const remoteAnalyzingRef = useRef(false)

  useEffect(() => {
    if (!isCallerDemo) {
      setRiskScore(detection.riskScore)
      setVerdict(detection.verdict)
      setSecondsAnalyzed(detection.secondsAnalyzed)
      setWav2vecResult(detection.wav2vec2)
      setAasistResult(detection.aasist)
      setWav2vecV2Result(detection.wav2vec2_v2)
      setAnalysisConnected(detection.analysisConnected)
    }
  }, [detection, isCallerDemo])

  const downsampleTo16k = (
    input: Float32Array,
    inputSampleRate: number,
  ): Int16Array => {
    if (inputSampleRate === 16000) {
      const output = new Int16Array(input.length)

      for (let i = 0; i < input.length; i += 1) {
        const sample = Math.max(-1, Math.min(1, input[i]))
        output[i] = sample < 0 ? sample * 32768 : sample * 32767
      }

      return output
    }

    const ratio = inputSampleRate / 16000
    const outputLength = Math.floor(input.length / ratio)
    const output = new Int16Array(outputLength)

    let outputIndex = 0
    let inputIndex = 0

    while (
      outputIndex < outputLength &&
      inputIndex < input.length
    ) {
      const nextInputIndex = Math.min(
        Math.floor((outputIndex + 1) * ratio),
        input.length,
      )

      let sum = 0
      let count = 0

      for (let i = inputIndex; i < nextInputIndex; i += 1) {
        sum += input[i]
        count += 1
      }

      const sample = count > 0 ? sum / count : 0
      const clipped = Math.max(-1, Math.min(1, sample))

      output[outputIndex] =
        clipped < 0 ? clipped * 32768 : clipped * 32767

      outputIndex += 1
      inputIndex = nextInputIndex
    }

    return output
  }

  const sendPcmToBackend = (pcm: Int16Array) => {
    const ws = wsRef.current

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return
    }

    const buffer = new ArrayBuffer(pcm.byteLength)
    new Int16Array(buffer).set(pcm)
    ws.send(buffer)
  }

  const connectAnalysisWebSocket = async () => {
    const token = getToken()

    if (!token) {
      throw new Error(
        'Authentication token not found. Please log in again.',
      )
    }

    const wsUrl = getWsUrl(`${API_BASE}/ws/stream`)
    const ws = new WebSocket(wsUrl)

    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    await new Promise<void>((resolve, reject) => {
      let settled = false

      const timeout = window.setTimeout(() => {
        if (!settled) {
          settled = true
          ws.close()
          reject(
            new Error('Voice analysis connection timed out.'),
          )
        }
      }, 8000)

      ws.onopen = () => {
        ws.send(JSON.stringify({ token }))
      }

      ws.onmessage = (event) => {
        if (typeof event.data !== 'string') {
          return
        }

        try {
          const message = JSON.parse(event.data)

          if (message.type === 'auth_ok') {
            if (!settled) {
              settled = true
              window.clearTimeout(timeout)
              setAnalysisConnected(true)
              resolve()
            }
            return
          }

          if (
            typeof message.label !== 'string' ||
            typeof message.confidence !== 'number'
          ) {
            return
          }

          const detection = message as DetectionEvent

          if (typeof detection.seconds_analyzed === 'number') {
            setSecondsAnalyzed(detection.seconds_analyzed)
          }

          if (detection.wav2vec2) {
            setWav2vecResult(detection.wav2vec2)
          }

          if (detection.aasist) {
            setAasistResult(detection.aasist)
          }

          if (detection.wav2vec2_v2) {
            setWav2vecV2Result(detection.wav2vec2_v2)
          }

          const wav2vecFake =
            detection.wav2vec2?.fake_probability

          const aasistFake =
            detection.aasist?.fake_probability

          const wav2vecV2Fake =
            detection.wav2vec2_v2?.fake_probability

          let finalFakeProbability: number

          if (
            typeof wav2vecFake === 'number' &&
            typeof aasistFake === 'number' &&
            typeof wav2vecV2Fake === 'number'
          ) {
            finalFakeProbability =
              wav2vecFake * 0.30 +
              aasistFake * 0.30 +
              wav2vecV2Fake * 0.40
          } else if (
            typeof detection.final_fake_probability === 'number'
          ) {
            finalFakeProbability =
              detection.final_fake_probability
          } else if (typeof wav2vecFake === 'number') {
            finalFakeProbability = wav2vecFake
          } else if (typeof aasistFake === 'number') {
            finalFakeProbability = aasistFake
          } else if (typeof wav2vecV2Fake === 'number') {
            finalFakeProbability = wav2vecV2Fake
          } else {
            finalFakeProbability =
              detection.label === 'fake'
                ? detection.confidence
                : detection.label === 'real'
                  ? 1 - detection.confidence
                  : 0.5
          }

          finalFakeProbability = Math.max(
            0,
            Math.min(1, finalFakeProbability),
          )

          setRiskScore(finalFakeProbability * 100)

          if (finalFakeProbability >= 0.5) {
            setVerdict('fake')
          } else {
            setVerdict('real')
          }
        } catch (parseError) {
          console.warn(
            'Unable to parse VoiceGuard analysis event:',
            parseError,
          )
        }
      }

      ws.onerror = () => {
        if (!settled) {
          settled = true
          window.clearTimeout(timeout)
          reject(
            new Error(
              'Unable to connect to VoiceGuard voice analysis.',
            ),
          )
        }
      }

      ws.onclose = () => {
        setAnalysisConnected(false)
      }
    })
  }

  const startLocalMicrophone = async () => {
    const mediaStream =
      await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      })

    mediaStreamRef.current = mediaStream

    const room = roomRef.current

    if (!room) {
      throw new Error('LiveKit room is not available.')
    }

    const microphoneTrack = new LocalAudioTrack(
      mediaStream.getAudioTracks()[0],
    )

    await room.localParticipant.publishTrack(microphoneTrack)
  }

  const startRemoteCallerAnalysis = async (
    track: RemoteTrack,
  ) => {
    if (
      isCallerDemo ||
      remoteAnalyzingRef.current ||
      track.kind !== Track.Kind.Audio
    ) {
      return
    }

    const mediaTrack = track.mediaStreamTrack

    if (!mediaTrack) {
      setError('Remote caller audio track is unavailable.')
      return
    }

    remoteAnalyzingRef.current = true

    try {
      const AudioContextClass =
        window.AudioContext ||
        (
          window as typeof window & {
            webkitAudioContext?: typeof AudioContext
          }
        ).webkitAudioContext

      if (!AudioContextClass) {
        throw new Error(
          'Web Audio API is not supported by this browser.',
        )
      }

      const audioContext =
        audioContextRef.current ?? new AudioContextClass()

      audioContextRef.current = audioContext

      if (audioContext.state === 'suspended') {
        await audioContext.resume()
      }

      const audioElement =
        track.attach() as HTMLAudioElement

      audioElement.autoplay = true
      audioElement.controls = false
      audioElement.muted = false
      audioElement.volume = 1
      audioElement.style.display = 'none'

      document.body.appendChild(audioElement)
      remoteAudioElementRef.current = audioElement

      try {
        await audioElement.play()
      } catch (playError) {
        console.warn(
          'VoiceGuard: remote audio autoplay was blocked:',
          playError,
        )
      }

      const callerMediaStream = new MediaStream([mediaTrack])

      const source =
        audioContext.createMediaStreamSource(
          callerMediaStream,
        )

      const processor =
        audioContext.createScriptProcessor(
          4096,
          1,
          1,
        )

      const silentGain = audioContext.createGain()
      silentGain.gain.value = 0

      remoteSourceRef.current = source
      remoteProcessorRef.current = processor
      remoteGainRef.current = silentGain

      processor.onaudioprocess = (event) => {
        const input =
          event.inputBuffer.getChannelData(0)

        let sumSquares = 0

        for (let i = 0; i < input.length; i += 1) {
          const sample = input[i]
          sumSquares += sample * sample
        }

        const rms =
          input.length > 0
            ? Math.sqrt(sumSquares / input.length)
            : 0

        if (rms > 0.005) {
          const pcm = downsampleTo16k(
            input,
            audioContext.sampleRate,
          )

          sendPcmToBackend(pcm)
        }
      }

      source.connect(processor)
      processor.connect(silentGain)
      silentGain.connect(audioContext.destination)

      console.log(
        'VoiceGuard: remote caller audio connected for analysis',
      )
    } catch (err) {
      remoteAnalyzingRef.current = false

      if (remoteAudioElementRef.current) {
        try {
          remoteAudioElementRef.current.remove()
        } catch {
          // Already removed.
        }
      }

      remoteAudioElementRef.current = null
      throw err
    }
  }

  const cleanupCall = async () => {
    const ws = wsRef.current

    if (ws) {
      try {
        ws.close()
      } catch {
        // Already closed.
      }
    }

    wsRef.current = null
    setAnalysisConnected(false)

    try {
      remoteProcessorRef.current?.disconnect()
    } catch {
      // Already disconnected.
    }

    try {
      remoteSourceRef.current?.disconnect()
    } catch {
      // Already disconnected.
    }

    try {
      remoteGainRef.current?.disconnect()
    } catch {
      // Already disconnected.
    }

    remoteProcessorRef.current = null
    remoteSourceRef.current = null
    remoteGainRef.current = null
    remoteAnalyzingRef.current = false

    if (remoteAudioElementRef.current) {
      try {
        remoteAudioElementRef.current.remove()
      } catch {
        // Already removed.
      }
    }

    remoteAudioElementRef.current = null

    if (audioContextRef.current) {
      try {
        await audioContextRef.current.close()
      } catch {
        // Already closed.
      }
    }

    audioContextRef.current = null

    mediaStreamRef.current
      ?.getTracks()
      .forEach((track) => track.stop())

    mediaStreamRef.current = null

    try {
      await roomRef.current?.disconnect()
    } catch {
      // Already disconnected.
    }

    roomRef.current = null
  }

  const startCallAnalysis = async () => {
    if (
      status === 'connecting' ||
      status === 'connected'
    ) {
      return
    }

    setError(null)
    setStatus('connecting')
    setIncoming(false)

    setRiskScore(null)
    setVerdict(null)
    setSecondsAnalyzed(null)
    setWav2vecResult(null)
    setAasistResult(null)
    setWav2vecV2Result(null)
    setAnalysisConnected(false)

    try {
      const response = await apiFetch(
        '/livekit/token?room=voiceguard-call',
      )

      if (!response.ok) {
        let detail =
          `LiveKit token request failed (${response.status})`

        try {
          const body = await response.json()

          if (body?.detail) {
            detail = String(body.detail)
          }
        } catch {
          // Keep default error.
        }

        throw new Error(detail)
      }

      const data: LiveKitTokenResponse =
        await response.json()

      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
      })

      roomRef.current = room

      room.on(
        RoomEvent.ConnectionStateChanged,
        (state) => {
          if (
            state === ConnectionState.Connected
          ) {
            setStatus('connected')
          }

          if (
            state === ConnectionState.Disconnected
          ) {
            setStatus('idle')
          }
        },
      )

      room.on(
        RoomEvent.TrackSubscribed,
        (
          track: RemoteTrack,
          _publication,
          participant,
        ) => {
          if (track.kind === Track.Kind.Audio) {
            console.log(
              'LiveKit remote audio track subscribed:',
              participant.identity,
            )

            void startRemoteCallerAnalysis(
              track,
            ).catch((err) => {
              console.error(
                'Remote caller analysis failed:',
                err,
              )

              setError(
                err instanceof Error
                  ? err.message
                  : 'Unable to analyze remote caller audio.',
              )
            })
          }
        },
      )

      await connectAnalysisWebSocket()

      await room.connect(
        data.url,
        data.token,
      )

      await startLocalMicrophone()

      setStatus('connected')
    } catch (err) {
      console.error(
        'VoiceGuard call connection failed:',
        err,
      )

      await cleanupCall()

      setStatus('error')

      setError(
        err instanceof Error
          ? err.message
          : 'Unable to start the VoiceGuard call.',
      )
    }
  }

  const endCallAnalysis = async () => {
    setStatus('disconnecting')

    await cleanupCall()

    setRiskScore(null)
    setVerdict(null)
    setSecondsAnalyzed(null)
    setWav2vecResult(null)
    setAasistResult(null)
    setWav2vecV2Result(null)

    setStatus('idle')
    setIncoming(true)
  }

  useEffect(() => {
    return () => {
      void cleanupCall()
    }
  }, [])

  const connected = status === 'connected'

  const getRiskLabel = () => {
    if (riskScore === null) {
      return 'WAITING FOR CALLER AUDIO'
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

  const getVerdictLabel = () => {
    if (verdict === 'fake') {
      return 'AI-GENERATED / CLONED VOICE'
    }

    if (verdict === 'real') {
      return 'LIKELY REAL VOICE'
    }

    if (verdict === 'unknown') {
      return 'ANALYSIS UNCERTAIN'
    }

    return 'WAITING FOR CALLER AUDIO'
  }

  const wavFake =
    wav2vecResult?.fake_probability

  const aasistFake =
    aasistResult?.fake_probability

  const wav2vecV2Fake =
    wav2vecV2Result?.fake_probability

  const ensembleScore =
    typeof wavFake === 'number' &&
    typeof aasistFake === 'number' &&
    typeof wav2vecV2Fake === 'number'
      ? wavFake * 0.30 +
        aasistFake * 0.30 +
        wav2vecV2Fake * 0.40
      : null

  if (isCallerDemo) {
    return (
      <div className="max-w-3xl mx-auto">
        <div className="rounded-3xl border border-gray-800 bg-gray-900 p-8 text-center">
          <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-indigo-600/20 text-4xl">
            🎙️
          </div>

          <h2 className="mt-5 text-2xl font-bold text-white">
            Caller Demo
          </h2>

          <p className="mt-2 text-gray-400">
            This tab acts as the remote caller in the
            software-only VoIP demonstration.
          </p>

          <div className="mt-6 rounded-2xl border border-gray-800 bg-gray-950 p-5 text-left">
            <div className="text-xs uppercase tracking-wider text-gray-500">
              Caller ID
            </div>

            <div className="mt-1 text-xl font-semibold text-white">
              {caller}
            </div>

            <div className="mt-3 text-sm text-green-400">
              ● Microphone will be published to LiveKit
            </div>
          </div>

          {!connected ? (
            <button
              onClick={() => {
                void startCallAnalysis()
              }}
              disabled={status === 'connecting'}
              className="mt-6 w-full rounded-xl bg-green-600 px-5 py-3 font-semibold text-white hover:bg-green-500 disabled:opacity-50"
            >
              {status === 'connecting'
                ? 'Joining Call…'
                : 'Join Call as Caller'}
            </button>
          ) : (
            <button
              onClick={() => {
                void endCallAnalysis()
              }}
              className="mt-6 w-full rounded-xl bg-red-600 px-5 py-3 font-semibold text-white hover:bg-red-500"
            >
              End Caller
            </button>
          )}

          {error && (
            <div className="mt-4 rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
              {error}
            </div>
          )}

          <div className="mt-5 text-xs text-gray-500">
            Open the normal Call page in another tab to act
            as VoiceGuard.
          </div>
        </div>
      </div>
    )
  }

  if (session?.status === 'rejected') {
    return (
      <div className="max-w-3xl mx-auto">
        <div className="rounded-3xl border border-gray-800 bg-gray-900 p-10 text-center">
          <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-red-600/10 text-4xl">
            ✕
          </div>

          <h2 className="mt-5 text-2xl font-bold text-white">
            Call Rejected
          </h2>

          <p className="mt-2 text-gray-400">
            The incoming call from {callerName} was rejected.
          </p>

          <button
            onClick={() => {
              setError(null)
            }}
            className="mt-6 rounded-xl border border-gray-700 bg-gray-950 px-6 py-3 text-sm font-medium text-gray-300 hover:bg-gray-800"
          >
            Back
          </button>
        </div>
      </div>
    )
  }

  if (session?.status === 'ended') {
    const durationSeconds = session.startTime
      ? Math.max(
          0,
          Math.round(
            (Date.now() -
              new Date(session.startTime).getTime()) /
              1000,
          ),
        )
      : null

    const finalRisk = riskScore
    const finalWavFake = wavFake
    const finalAasistFake = aasistFake
    const finalWav2vecV2Fake = wav2vecV2Fake
    const finalEnsemble = ensembleScore
    const finalVerdict = verdict

    const riskText =
      finalRisk === null
        ? 'Analysis unavailable'
        : finalRisk >= 75
          ? 'CRITICAL RISK'
          : finalRisk >= 60
            ? 'HIGH RISK'
            : finalRisk >= 30
              ? 'SUSPICIOUS'
              : 'LOW RISK'

    return (
      <div className="max-w-5xl mx-auto">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white">
              Call Summary
            </h2>
            <p className="mt-1 text-sm text-gray-400">
              VoiceGuard completed the real-time voice security analysis.
            </p>
          </div>

          <span className="rounded-full border border-gray-700 bg-gray-900 px-4 py-2 text-xs font-semibold text-gray-400">
            CALL ENDED
          </span>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6">
            <div className="text-xs uppercase tracking-wider text-gray-500">
              Caller
            </div>

            <div className="mt-3 flex items-center gap-4">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-indigo-600/20 text-3xl">
                👤
              </div>
              <div>
                <div className="text-xl font-semibold text-white">
                  {caller}
                </div>
                <div className="text-sm text-gray-400">
                  {callerName} · VoIP
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  {callPurpose}
                </div>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-gray-800 bg-gray-950 p-4">
                <div className="text-xs text-gray-500">
                  Call status
                </div>
                <div className="mt-1 font-medium text-gray-300">
                  Completed
                </div>
              </div>
              <div className="rounded-xl border border-gray-800 bg-gray-950 p-4">
                <div className="text-xs text-gray-500">
                  Audio analyzed
                </div>
                <div className="mt-1 font-medium text-gray-300">
                  {secondsAnalyzed !== null
                    ? `${secondsAnalyzed.toFixed(2)} sec`
                    : '—'}
                </div>
              </div>
            </div>

            {durationSeconds !== null && (
              <div className="mt-4 flex justify-between rounded-xl border border-gray-800 bg-gray-950 p-4 text-sm">
                <span className="text-gray-400">
                  Call duration
                </span>
                <span className="font-medium text-white">
                  {Math.floor(durationSeconds / 60)}:
                  {String(durationSeconds % 60).padStart(2, '0')}
                </span>
              </div>
            )}

            <div className="mt-5 rounded-xl border border-gray-800 bg-gray-950 p-4">
              <div className="text-sm font-semibold text-white">
                Recorded Caller Voice
              </div>
              <p className="mt-1 text-xs text-gray-500">
                The caller audio captured during this simulated call.
              </p>

              {recording ? (
                <>
                  <audio
                    className="mt-4 w-full"
                    controls
                    preload="metadata"
                    src={recording.url}
                  />
                  <div className="mt-3 flex flex-wrap gap-3">
                    <a
                      href={recording.url}
                      download={`voiceguard-call-${caller.replace(
                        /[^a-zA-Z0-9+]/g,
                        '_',
                      )}.webm`}
                      className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500"
                    >
                      Download Recording
                    </a>
                    <span className="self-center text-xs text-gray-500">
                      {recording.durationSeconds.toFixed(1)} sec ·{' '}
                      {(recording.blob.size / 1024).toFixed(0)} KB
                    </span>
                  </div>
                </>
              ) : (
                <div className="mt-4 rounded-lg border border-gray-800 bg-gray-900 p-3 text-sm text-gray-500">
                  Preparing the call recording…
                </div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6">
            <div className="text-sm font-semibold text-white">
              Final Voice Security Result
            </div>

            <div className="mt-6 text-center">
              <div className="text-sm text-gray-400">
                Impersonation Risk
              </div>
              <div
                className={`mt-2 text-6xl font-bold ${getRiskClass()}`}
              >
                {finalRisk === null
                  ? '—'
                  : `${Math.round(finalRisk)}%`}
              </div>
              <div
                className={`mt-2 text-sm font-semibold ${getRiskClass()}`}
              >
                {riskText}
              </div>
              <div className="mt-2 text-sm text-gray-400">
                {finalVerdict === 'fake'
                  ? 'Possible AI-generated / cloned voice'
                  : finalVerdict === 'real'
                    ? 'Likely genuine voice'
                    : 'No final verdict available'}
              </div>
            </div>

            <div className="mt-6 space-y-3">
              <div className="flex justify-between rounded-xl border border-gray-800 bg-gray-950 p-4">
                <span className="text-gray-400">
                  Wav2Vec2 Spoof
                </span>
                <span className="font-medium text-white">
                  {typeof finalWavFake === 'number'
                    ? `${Math.round(finalWavFake * 100)}% fake`
                    : '—'}
                </span>
              </div>

              <div className="flex justify-between rounded-xl border border-gray-800 bg-gray-950 p-4">
                <span className="text-gray-400">
                  Official AASIST
                </span>
                <span className="font-medium text-white">
                  {typeof finalAasistFake === 'number'
                    ? `${Math.round(finalAasistFake * 100)}% fake`
                    : '—'}
                </span>
              </div>

              <div className="flex justify-between rounded-xl border border-gray-800 bg-gray-950 p-4">
                <span className="text-gray-400">
                  Wav2Vec2 v2
                </span>
                <span className="font-medium text-white">
                  {typeof finalWav2vecV2Fake === 'number'
                    ? `${Math.round(
                        finalWav2vecV2Fake * 100,
                      )}% fake`
                    : '—'}
                </span>
              </div>

              <div className="flex items-center justify-between rounded-xl border border-indigo-900/60 bg-indigo-950/20 p-4">
                <div>
                  <div className="font-medium text-gray-300">
                    Ensemble (30/30/40)
                  </div>
                  <div className="mt-1 text-xs text-gray-500">
                    Wav2Vec2 30% + AASIST 30% + Wav2Vec2 v2 40%
                  </div>
                </div>
                <span className="font-semibold text-indigo-300">
                  {finalEnsemble !== null
                    ? `${Math.round(
                        finalEnsemble * 100,
                      )}% fake`
                    : finalRisk !== null
                      ? `${Math.round(finalRisk)}% fake`
                      : '—'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {finalRisk !== null && finalRisk >= 60 && (
          <div className="mt-6 rounded-2xl border border-red-800 bg-red-950/40 p-6">
            <div className="flex items-start gap-4">
              <div className="text-3xl">🚨</div>
              <div>
                <h3 className="text-lg font-semibold text-red-300">
                  High Impersonation Risk
                </h3>
                <p className="mt-2 text-sm text-red-200/80">
                  Do not approve sensitive actions. Use secondary verification through an independent trusted channel.
                </p>
                <button
                  onClick={() =>
                    window.alert(
                      'Secondary verification recommended: verify the caller through an independent trusted channel.',
                    )
                  }
                  className="mt-4 rounded-lg bg-red-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-red-500"
                >
                  Verify Caller
                </button>
              </div>
            </div>
          </div>
        )}

        {finalRisk !== null && finalRisk < 60 && (
          <div className="mt-6 rounded-2xl border border-gray-800 bg-gray-900 p-5">
            <div className="text-sm text-gray-400">
              Recommendation
            </div>
            <div className="mt-1 font-medium text-white">
              Continue normal verification practices before sensitive actions.
            </div>
          </div>
        )}
      </div>
    )
  }

  if (incoming && !connected) {
    return (
      <div className="max-w-3xl mx-auto">
        <div className="overflow-hidden rounded-3xl border border-gray-800 bg-gray-900 shadow-2xl">
          <div className="p-10 text-center">
            <div className="mx-auto flex h-24 w-24 animate-pulse items-center justify-center rounded-full bg-indigo-600/20 text-5xl">
              📞
            </div>

            <div className="mt-6 text-sm font-semibold uppercase tracking-[0.25em] text-indigo-400">
              Incoming Call
            </div>

            <h2 className="mt-3 text-3xl font-bold text-white">
              {caller}
            </h2>

            <p className="mt-2 text-gray-400">
              {callerName}
            </p>

            {session && (
              <p className="mt-1 text-xs text-gray-500">
                {callPurpose}
              </p>
            )}

            <div className="mx-auto mt-5 inline-flex rounded-full border border-gray-700 bg-gray-950 px-4 py-2 text-sm text-gray-300">
              🔒 VoIP Call
            </div>

            <div className="mt-8 grid gap-3 sm:grid-cols-2">
              <button
                onClick={() => {
                  onAccept()

                  if (isCallerDemo || !isSimulatedCall) {
                    void startCallAnalysis()
                  }
                }}
                disabled={status === 'connecting'}
                className="rounded-xl bg-green-600 px-5 py-4 font-semibold text-white hover:bg-green-500 disabled:opacity-50"
              >
                {status === 'connecting'
                  ? 'Connecting…'
                  : '✓ Accept Call'}
              </button>

              <button
                onClick={() => {
                  setError('Call rejected.')
                  onReject()
                }}
                className="rounded-xl border border-gray-700 bg-gray-950 px-5 py-4 font-semibold text-gray-300 hover:bg-gray-800"
              >
                ✕ Reject
              </button>
            </div>

            {error && (
              <div className="mt-5 rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
                {error}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">
            Call Security
          </h2>

          <p className="text-sm text-gray-400">
            VoiceGuard analyzes the remote caller's voice
            in real time.
          </p>
        </div>

        <span className="text-xs font-semibold text-green-400">
          ● CALL CONNECTED
        </span>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6">
          <div className="text-xs uppercase tracking-wider text-gray-500">
            Caller
          </div>

          <div className="mt-2 flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-indigo-600/20 text-3xl">
              👤
            </div>

            <div>
              <div className="text-xl font-semibold text-white">
                {caller}
              </div>

              <div className="text-sm text-gray-400">
                {callerName} · VoIP
              </div>
            </div>
          </div>

          <div className="mt-7 grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-gray-800 bg-gray-950 p-4">
              <div className="text-xs text-gray-500">
                LiveKit
              </div>
              <div className="mt-1 text-green-400">
                Connected
              </div>
            </div>

            <div className="rounded-xl border border-gray-800 bg-gray-950 p-4">
              <div className="text-xs text-gray-500">
                Analysis
              </div>
              <div className="mt-1 text-green-400">
                {analysisConnected
                  ? 'Connected'
                  : 'Waiting'}
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-xl border border-indigo-900/50 bg-indigo-950/20 p-4 text-sm text-indigo-200">
            🎙️ Both participants can speak. Only the
            <strong> remote caller audio </strong>
            is sent to the deepfake detector.
          </div>

          <button
            onClick={() => {
              void endCallAnalysis()
              onEndCall()
            }}
            className="mt-6 w-full rounded-xl bg-red-600 px-5 py-3 font-semibold text-white hover:bg-red-500"
          >
            End Call
          </button>
        </div>

        <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6">
          <div className="text-sm font-semibold text-white">
            Live Caller Voice Risk
          </div>

          <div className="mt-7 text-center">
            <div className="text-sm text-gray-400">
              Impersonation Risk
            </div>

            <div
              className={`mt-2 text-6xl font-bold ${getRiskClass()}`}
            >
              {riskScore === null
                ? '—'
                : `${Math.round(riskScore)}%`}
            </div>

            <div
              className={`mt-2 text-sm font-semibold ${getRiskClass()}`}
            >
              {getRiskLabel()}
            </div>

            <div className="mt-3 text-sm text-gray-400">
              {getVerdictLabel()}
            </div>
          </div>

          <div className="mt-6 h-3 w-full overflow-hidden rounded-full bg-gray-800">
            <div
              className="h-full rounded-full bg-indigo-500 transition-all duration-500"
              style={{
                width: `${Math.min(
                  100,
                  Math.max(0, riskScore ?? 0),
                )}%`,
              }}
            />
          </div>

          <div className="mt-7 space-y-3">
            <div className="flex justify-between rounded-xl border border-gray-800 bg-gray-950 p-4">
              <span className="text-gray-400">
                Wav2Vec2 Spoof
              </span>

              <span className="font-medium text-white">
                {typeof wavFake === 'number'
                  ? `${Math.round(wavFake * 100)}% fake`
                  : 'Waiting'}
              </span>
            </div>

            <div className="flex justify-between rounded-xl border border-gray-800 bg-gray-950 p-4">
              <span className="text-gray-400">
                Official AASIST
              </span>

              <span className="font-medium text-white">
                {typeof aasistFake === 'number'
                  ? `${Math.round(aasistFake * 100)}% fake`
                  : 'Waiting'}
              </span>
            </div>

            <div className="flex justify-between rounded-xl border border-gray-800 bg-gray-950 p-4">
              <span className="text-gray-400">
                Wav2Vec2 v2
              </span>

              <span className="font-medium text-white">
                {typeof wav2vecV2Fake === 'number'
                  ? `${Math.round(
                      wav2vecV2Fake * 100,
                    )}% fake`
                  : 'Waiting'}
              </span>
            </div>

            <div className="flex items-center justify-between rounded-xl border border-indigo-900/60 bg-indigo-950/20 p-4">
              <div>
                <div className="font-medium text-gray-300">
                  Ensemble (30/30/40)
                </div>

                <div className="mt-1 text-xs text-gray-500">
                  Wav2Vec2 30% + AASIST 30% + Wav2Vec2 v2 40%
                </div>
              </div>

              <span className="font-semibold text-indigo-300">
                {ensembleScore !== null
                  ? `${Math.round(
                      ensembleScore * 100,
                    )}% fake`
                  : riskScore !== null
                    ? `${Math.round(riskScore)}% fake`
                    : 'Waiting'}
              </span>
            </div>

            {secondsAnalyzed !== null && (
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">
                  Audio analyzed
                </span>

                <span className="text-white">
                  {secondsAnalyzed.toFixed(2)} sec
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {connected &&
        riskScore !== null &&
        riskScore >= 75 && (
          <div className="mt-6 rounded-2xl border border-red-700 bg-red-950/50 p-6">
            <div className="flex items-start gap-4">
              <div className="text-3xl">🚨</div>

              <div className="flex-1">
                <h3 className="text-lg font-semibold text-red-300">
                  Critical Voice Threat Detected
                </h3>

                <p className="mt-2 text-sm text-red-200/80">
                  VoiceGuard detected a very high probability
                  of AI-generated or cloned voice activity.
                  Do not approve sensitive actions.
                </p>

                <div className="mt-4 text-sm text-red-200/70">
                  Risk score:{' '}
                  <span className="font-semibold text-red-300">
                    {Math.round(riskScore)}%
                  </span>
                </div>

                <div className="mt-5 flex flex-wrap gap-3">
                  <button
                    onClick={() =>
                      window.alert(
                        'Critical risk detected. Verify the caller through an independent trusted channel before taking any sensitive action.',
                      )
                    }
                    className="rounded-lg bg-red-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-red-500"
                  >
                    Verify Caller
                  </button>

                  <button
                    onClick={() => {
                      void endCallAnalysis()
                    }}
                    className="rounded-lg border border-red-700 px-5 py-2.5 text-sm font-medium text-red-300 hover:bg-red-900/40"
                  >
                    End Call
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

      {connected &&
        riskScore !== null &&
        riskScore >= 60 &&
        riskScore < 75 && (
          <div className="mt-6 rounded-2xl border border-orange-800 bg-orange-950/30 p-6">
            <div className="flex items-start gap-4">
              <div className="text-3xl">🚨</div>

              <div className="flex-1">
                <h3 className="text-lg font-semibold text-orange-300">
                  High-Risk Voice Detected
                </h3>

                <p className="mt-2 text-sm text-orange-200/80">
                  VoiceGuard detected a high probability
                  of AI-generated or cloned voice activity.
                  Do not approve sensitive actions without verification.
                </p>

                <div className="mt-4 text-sm text-orange-200/70">
                  Risk score:{' '}
                  <span className="font-semibold text-orange-300">
                    {Math.round(riskScore)}%
                  </span>
                </div>

                <div className="mt-5 flex flex-wrap gap-3">
                  <button
                    onClick={() =>
                      window.alert(
                        'Secondary verification recommended: verify the caller through an independent trusted channel.',
                      )
                    }
                    className="rounded-lg bg-orange-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-orange-500"
                  >
                    Verify Caller
                  </button>

                  <button
                    onClick={() => {
                      void endCallAnalysis()
                    }}
                    className="rounded-lg border border-orange-700 px-5 py-2.5 text-sm font-medium text-orange-300 hover:bg-orange-900/40"
                  >
                    End Call
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

      {connected &&
        riskScore !== null &&
        riskScore >= 30 &&
        riskScore < 60 && (
          <div className="mt-6 rounded-2xl border border-yellow-800 bg-yellow-950/30 p-6">
            <div className="flex items-start gap-4">
              <div className="text-3xl">⚠️</div>

              <div className="flex-1">
                <h3 className="text-lg font-semibold text-yellow-300">
                  Suspicious Voice Activity
                </h3>

                <p className="mt-2 text-sm text-yellow-200/80">
                  VoiceGuard detected suspicious voice characteristics.
                  The caller may be using synthetic or manipulated audio.
                </p>

                <div className="mt-4 text-sm text-yellow-200/70">
                  Risk score:{' '}
                  <span className="font-semibold text-yellow-300">
                    {Math.round(riskScore)}%
                  </span>
                </div>

                <div className="mt-5 flex flex-wrap gap-3">
                  <button
                    onClick={() =>
                      window.alert(
                        'Secondary verification recommended: verify the caller through an independent trusted channel.',
                      )
                    }
                    className="rounded-lg bg-yellow-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-yellow-500"
                  >
                    Verify Caller
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

      {connected &&
        riskScore !== null &&
        riskScore < 30 && (
          <div className="mt-6 rounded-2xl border border-green-900 bg-green-950/20 p-5">
            <div className="flex items-center gap-3">
              <div className="text-2xl">✓</div>

              <div>
                <h3 className="font-semibold text-green-300">
                  Low Impersonation Risk
                </h3>

                <p className="mt-1 text-sm text-green-200/70">
                  Current caller voice characteristics are
                  consistent with a likely genuine voice.
                </p>
              </div>
            </div>
          </div>
        )}

      {error && (
        <div className="mt-6 rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
          {error}
        </div>
      )}
    </div>
  )
}
