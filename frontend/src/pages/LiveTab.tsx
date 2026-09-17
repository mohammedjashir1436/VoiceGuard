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

type LiveTabProps = {
  session: CallSession | null
  onEndCall: () => void
  onDetectionChange?: (detection: CallDetection) => void
  onCallRecording?: (recording: CallRecording) => void
}

export default function LiveTab({
  session,
  onEndCall,
  onDetectionChange,
  onCallRecording,
}: LiveTabProps) {
  const caller = session?.phone || '+91 98765 43210'
  const callerName = session?.callerName || 'Unknown Caller'
  const isCallerDemo =
    new URLSearchParams(window.location.search).get('role') === 'caller'

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
  const [wav2vec2V2Result, setWav2vec2V2Result] =
    useState<DetectionModelResult | null>(null)

  const [analysisConnected, setAnalysisConnected] = useState(false)

  const isStandaloneLive = session === null

  const [standaloneStatus, setStandaloneStatus] =
    useState<'idle' | 'starting' | 'live' | 'ended' | 'error'>('idle')
  const [standaloneRecordingUrl, setStandaloneRecordingUrl] =
    useState<string | null>(null)
  const [standaloneWaveform, setStandaloneWaveform] =
    useState<number[]>(() => Array.from({ length: 48 }, () => 0.08))

  const standaloneStreamRef = useRef<MediaStream | null>(null)
  const standaloneAudioContextRef = useRef<AudioContext | null>(null)
  const standaloneProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const standaloneSourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const standaloneGainRef = useRef<GainNode | null>(null)
  const standaloneAnalyserRef = useRef<AnalyserNode | null>(null)
  const standaloneWaveformFrameRef = useRef<number | null>(null)
  const standaloneRecorderRef = useRef<MediaRecorder | null>(null)
  const standaloneRecorderChunksRef = useRef<Blob[]>([])
  const standaloneRecordingUrlRef = useRef<string | null>(null)

  const roomRef = useRef<Room | null>(null)
  const callerRoomRef = useRef<Room | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const remoteProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const remoteSourceRef = useRef<AudioNode | null>(null)
  const remoteGainRef = useRef<GainNode | null>(null)
  const remoteAudioElementRef = useRef<HTMLAudioElement | null>(null)
  const remoteAnalyzingRef = useRef(false)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const recorderChunksRef = useRef<Blob[]>([])
  const recordingStartedAtRef = useRef<number | null>(null)
  const cleanupPromiseRef = useRef<Promise<void> | null>(null)
  const lastAudioDebugRef = useRef(0)

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

  const stopStandaloneWaveform = () => {
    if (standaloneWaveformFrameRef.current !== null) {
      window.cancelAnimationFrame(standaloneWaveformFrameRef.current)
      standaloneWaveformFrameRef.current = null
    }

    setStandaloneWaveform(
      Array.from({ length: 48 }, () => 0.08),
    )
  }

  const startStandaloneWaveform = (analyser: AnalyserNode) => {
    standaloneAnalyserRef.current = analyser

    const data = new Uint8Array(analyser.fftSize)

    const draw = () => {
      analyser.getByteTimeDomainData(data)

      const bars = 48
      const step = Math.max(1, Math.floor(data.length / bars))
      const next = Array.from({ length: bars }, (_, index) => {
        const start = index * step
        const end = Math.min(data.length, start + step)
        let peak = 0

        for (let i = start; i < end; i += 1) {
          const amplitude = Math.abs(data[i] - 128) / 128
          if (amplitude > peak) peak = amplitude
        }

        return Math.max(0.08, Math.min(1, peak * 2.8))
      })

      setStandaloneWaveform(next)
      standaloneWaveformFrameRef.current =
        window.requestAnimationFrame(draw)
    }

    draw()
  }

  const sendPcmToBackend = (pcm: Int16Array) => {
    const ws = wsRef.current

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return false
    }

    const buffer = new ArrayBuffer(pcm.byteLength)
    new Int16Array(buffer).set(pcm)
    ws.send(buffer)

    return true
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
            setWav2vec2V2Result(detection.wav2vec2_v2)
          }

          /*
           * FINAL PRODUCTION ENSEMBLE
           * Wav2Vec2 Spoof = 30%
           * Official AASIST = 30%
           * Wav2Vec2 v2    = 40%
           */
          const wav2vecFake =
            detection.wav2vec2?.fake_probability

          const aasistFake =
            detection.aasist?.fake_probability

          const wav2vec2V2Fake =
            detection.wav2vec2_v2?.fake_probability

          let finalFakeProbability: number

          if (
            typeof wav2vecFake === 'number' &&
            typeof aasistFake === 'number' &&
            typeof wav2vec2V2Fake === 'number'
          ) {
            finalFakeProbability =
              wav2vecFake * 0.30 +
              aasistFake * 0.30 +
              wav2vec2V2Fake * 0.40
          } else if (
            typeof detection.final_fake_probability === 'number'
          ) {
            finalFakeProbability =
              detection.final_fake_probability
          } else if (typeof wav2vecFake === 'number') {
            finalFakeProbability = wav2vecFake
          } else if (typeof aasistFake === 'number') {
            finalFakeProbability = aasistFake
          } else if (typeof wav2vec2V2Fake === 'number') {
            finalFakeProbability = wav2vec2V2Fake
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

          const nextRiskScore = finalFakeProbability * 100
          const nextVerdict = finalFakeProbability >= 0.5 ? 'fake' : 'real'

          onDetectionChange?.({
            riskScore: nextRiskScore,
            verdict: nextVerdict,
            secondsAnalyzed:
              typeof detection.seconds_analyzed === 'number'
                ? detection.seconds_analyzed
                : null,
            wav2vec2: detection.wav2vec2 ?? null,
            aasist: detection.aasist ?? null,
            wav2vec2_v2: detection.wav2vec2_v2 ?? null,
            analysisConnected: true,
          })

          setRiskScore(nextRiskScore)

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

  const cleanupStandaloneResources = async () => {
    try {
      standaloneProcessorRef.current?.disconnect()
    } catch {
      // Already disconnected.
    }

    try {
      standaloneSourceRef.current?.disconnect()
    } catch {
      // Already disconnected.
    }

    try {
      standaloneGainRef.current?.disconnect()
    } catch {
      // Already disconnected.
    }

    standaloneProcessorRef.current = null
    standaloneSourceRef.current = null
    standaloneGainRef.current = null
    standaloneAnalyserRef.current = null
    stopStandaloneWaveform()

    if (standaloneAudioContextRef.current) {
      try {
        await standaloneAudioContextRef.current.close()
      } catch {
        // Already closed.
      }
    }

    standaloneAudioContextRef.current = null

    standaloneStreamRef.current
      ?.getTracks()
      .forEach((track) => track.stop())

    standaloneStreamRef.current = null

    const recorder = standaloneRecorderRef.current

    if (recorder && recorder.state !== 'inactive') {
      try {
        recorder.stop()
      } catch {
        // Already stopped.
      }
    }

    standaloneRecorderRef.current = null

    if (
      wsRef.current &&
      wsRef.current.readyState !== WebSocket.CLOSED
    ) {
      try {
        wsRef.current.close()
      } catch {
        // Already closed.
      }
    }

    wsRef.current = null
    setAnalysisConnected(false)
  }

  const startStandaloneLiveAnalysis = async () => {
    try {
      setStandaloneStatus('starting')
      setError(null)

      setRiskScore(null)
      setVerdict(null)
      setSecondsAnalyzed(null)
      setWav2vecResult(null)
      setAasistResult(null)
      setWav2vec2V2Result(null)
      setStandaloneRecordingUrl(null)
      setStandaloneWaveform(Array.from({ length: 48 }, () => 0.08))

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

      standaloneStreamRef.current = mediaStream

      if (typeof MediaRecorder === 'undefined') {
        throw new Error(
          'MediaRecorder is not supported by this browser.',
        )
      }

      const mimeCandidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
      ]

      const mimeType = mimeCandidates.find((type) =>
        MediaRecorder.isTypeSupported(type),
      )

      const recorder = mimeType
        ? new MediaRecorder(mediaStream, { mimeType })
        : new MediaRecorder(mediaStream)

      standaloneRecorderChunksRef.current = []

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          standaloneRecorderChunksRef.current.push(event.data)
        }
      }

      standaloneRecorderRef.current = recorder

      recorder.start(1000)

      await connectAnalysisWebSocket()

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

      const audioContext = new AudioContextClass()

      standaloneAudioContextRef.current = audioContext

      if (audioContext.state === 'suspended') {
        await audioContext.resume()
      }

      const source =
        audioContext.createMediaStreamSource(mediaStream)

      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      analyser.smoothingTimeConstant = 0.75

      const processor =
        audioContext.createScriptProcessor(
          4096,
          1,
          1,
        )

      const silentGain =
        audioContext.createGain()

      silentGain.gain.value = 0.00001

      processor.onaudioprocess = (event) => {
        const input =
          event.inputBuffer.getChannelData(0)

        const pcm = downsampleTo16k(
          input,
          audioContext.sampleRate,
        )

        const sent = sendPcmToBackend(pcm)

        if (sent) {
          const now = performance.now()

          if (
            !lastAudioDebugRef.current ||
            now - lastAudioDebugRef.current > 1000
          ) {
            lastAudioDebugRef.current = now

            console.log(
              'VoiceGuard: STANDALONE PCM SENT',
              {
                samples: pcm.length,
                wsState: wsRef.current?.readyState,
              },
            )
          }
        }
      }

      source.connect(analyser)
      analyser.connect(processor)
      processor.connect(silentGain)
      silentGain.connect(audioContext.destination)

      standaloneSourceRef.current = source
      standaloneProcessorRef.current = processor
      standaloneGainRef.current = silentGain
      standaloneAnalyserRef.current = analyser
      startStandaloneWaveform(analyser)

      setStandaloneStatus('live')

      console.log(
        'VoiceGuard: standalone live microphone analysis started',
      )
    } catch (err) {
      console.error(
        'VoiceGuard standalone live analysis failed:',
        err,
      )

      await cleanupStandaloneResources()

      setStandaloneStatus('error')

      setError(
        err instanceof Error
          ? err.message
          : 'Unable to start standalone live analysis.',
      )
    }
  }

  const endStandaloneLiveAnalysis = async () => {
    try {
      const recorder = standaloneRecorderRef.current

      if (recorder) {
        const blob = await new Promise<Blob | null>((resolve) => {
          const resolveOnce = () => {
            const chunks = standaloneRecorderChunksRef.current

            resolve(
              chunks.length > 0
                ? new Blob(chunks, {
                    type:
                      recorder.mimeType ||
                      'audio/webm',
                  })
                : null,
            )
          }

          recorder.addEventListener(
            'stop',
            resolveOnce,
            { once: true },
          )

          try {
            if (recorder.state !== 'inactive') {
              recorder.stop()
            } else {
              resolveOnce()
            }
          } catch {
            resolveOnce()
          }
        })

        if (blob && blob.size > 0) {
          if (standaloneRecordingUrlRef.current) {
            URL.revokeObjectURL(
              standaloneRecordingUrlRef.current,
            )
          }

          const url = URL.createObjectURL(blob)

          standaloneRecordingUrlRef.current = url
          setStandaloneRecordingUrl(url)
        }
      }

      await cleanupStandaloneResources()

      setStandaloneStatus('ended')
    } catch (err) {
      console.error(
        'VoiceGuard standalone live analysis cleanup failed:',
        err,
      )

      await cleanupStandaloneResources()
      setStandaloneStatus('ended')
    }
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


  /*
   * Simulated caller for the single-browser demo.
   *
   * This creates a SECOND LiveKit participant in the same room and
   * publishes the browser microphone from that participant. The
   * receiver room therefore sees it as a real REMOTE caller track,
   * which is what VoiceGuard analyzes.
   */
  const startSimulatedCaller = async (url: string) => {
    const response = await apiFetch(
      '/livekit/token?room=voiceguard-call',
    )

    if (!response.ok) {
      throw new Error(
        `Caller LiveKit token request failed (${response.status})`,
      )
    }

    const data: LiveKitTokenResponse = await response.json()

    const callerRoom = new Room({
      adaptiveStream: true,
      dynacast: true,
    })

    callerRoomRef.current = callerRoom

    await callerRoom.connect(
      url || data.url,
      data.token,
    )

    const callerStream =
      await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      })

    const callerTrack = new LocalAudioTrack(
      callerStream.getAudioTracks()[0],
    )

    await callerRoom.localParticipant.publishTrack(
      callerTrack,
    )

    /*
     * Keep this stream in the same ref so cleanupCall() stops it.
     */
    mediaStreamRef.current = callerStream

    console.log(
      'VoiceGuard: simulated caller published microphone to LiveKit',
    )
  }

  const startCallRecording = (stream: MediaStream) => {
    if (typeof MediaRecorder === 'undefined') {
      console.warn('VoiceGuard: MediaRecorder is not supported; post-call playback will be unavailable.')
      return
    }

    if (recorderRef.current) {
      return
    }

    const mimeCandidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
    ]
    const mimeType = mimeCandidates.find((type) =>
      MediaRecorder.isTypeSupported(type),
    )

    try {
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream)

      recorderChunksRef.current = []
      recordingStartedAtRef.current = performance.now()

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recorderChunksRef.current.push(event.data)
        }
      }

      recorder.onerror = (event) => {
        console.warn('VoiceGuard: call recording error:', event)
      }

      recorder.start(1000)
      recorderRef.current = recorder
      console.log('VoiceGuard: post-call recording started', { mimeType: recorder.mimeType })
    } catch (err) {
      console.warn('VoiceGuard: unable to start call recording:', err)
      recorderRef.current = null
    }
  }

  const finishCallRecording = async () => {
    const recorder = recorderRef.current
    if (!recorder) {
      return
    }

    const durationSeconds = recordingStartedAtRef.current
      ? Math.max(0, (performance.now() - recordingStartedAtRef.current) / 1000)
      : 0

    const blob = await new Promise<Blob | null>((resolve) => {
      const resolveOnce = () => {
        const chunks = recorderChunksRef.current
        resolve(
          chunks.length > 0
            ? new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
            : null,
        )
      }

      recorder.addEventListener('stop', resolveOnce, { once: true })

      try {
        if (recorder.state !== 'inactive') {
          recorder.stop()
        } else {
          resolveOnce()
        }
      } catch {
        resolveOnce()
      }
    })

    recorderRef.current = null
    recorderChunksRef.current = []
    recordingStartedAtRef.current = null

    if (!blob || blob.size === 0) {
      return
    }

    const url = URL.createObjectURL(blob)
    onCallRecording?.({
      url,
      blob,
      durationSeconds: Number(durationSeconds.toFixed(2)),
    })

    console.log('VoiceGuard: post-call recording ready', {
      bytes: blob.size,
      durationSeconds: Number(durationSeconds.toFixed(2)),
    })
  }

  const startRemoteCallerAnalysis = async (
    track: RemoteTrack | MediaStreamTrack | MediaStream,
  ) => {
    if (remoteAnalyzingRef.current) {
      return
    }

    const mediaTrack =
      track instanceof MediaStream
        ? track.getAudioTracks()[0]
        : 'mediaStreamTrack' in track
          ? track.mediaStreamTrack
          : track

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

      // For a real remote LiveKit track, attach it so the caller can be heard.
      // For the single-browser Simulate -> Live demo, the same microphone track
      // is analyzed directly because a second LiveKit participant is not
      // guaranteed to trigger TrackSubscribed in the same browser session.
      if ('attach' in track && typeof track.attach === 'function') {
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
      }

      const callerMediaStream =
        track instanceof MediaStream
          ? track
          : new MediaStream([mediaTrack])

      startCallRecording(callerMediaStream)

      const source =
        audioContext.createMediaStreamSource(
          callerMediaStream,
        )

      // ScriptProcessorNode is deprecated, but it is intentionally kept
      // here for broad browser compatibility. The processor must have an
      // output connection for browsers to drive onaudioprocess reliably.
      const processor =
        audioContext.createScriptProcessor(
          4096,
          1,
          1,
        )

      const silentGain = audioContext.createGain()
      // Keep the graph active without making the caller hear a feedback loop.
      silentGain.gain.value = 0.00001

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

        const pcm = downsampleTo16k(
          input,
          audioContext.sampleRate,
        )

        const sent = sendPcmToBackend(pcm)

        if (sent) {
          const now = performance.now()
          if (
            !lastAudioDebugRef.current ||
            now - lastAudioDebugRef.current > 1000
          ) {
            lastAudioDebugRef.current = now
            console.log(
              'VoiceGuard: PCM streaming active',
              {
                rms: Number(rms.toFixed(4)),
                samples: pcm.length,
                ws: wsRef.current?.readyState,
              },
            )
          }
        }
      }

      source.connect(processor)
      processor.connect(silentGain)
      silentGain.connect(audioContext.destination)

      console.log(
        'VoiceGuard: caller audio connected for analysis',
        {
          sampleRate: audioContext.sampleRate,
          trackReadyState: mediaTrack.readyState,
          trackEnabled: mediaTrack.enabled,
          trackMuted: mediaTrack.muted,
        },
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
    if (cleanupPromiseRef.current) {
      return cleanupPromiseRef.current
    }

    cleanupPromiseRef.current = (async () => {
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

    await finishCallRecording()

    mediaStreamRef.current
      ?.getTracks()
      .forEach((track) => track.stop())

    mediaStreamRef.current = null

    try {
      await callerRoomRef.current?.disconnect()
    } catch {
      // Already disconnected.
    }

    callerRoomRef.current = null

    try {
      await roomRef.current?.disconnect()
    } catch {
      // Already disconnected.
    }

    roomRef.current = null
    })()

    try {
      await cleanupPromiseRef.current
    } finally {
      cleanupPromiseRef.current = null
    }
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
  
    setRiskScore(null)
    setVerdict(null)
    setSecondsAnalyzed(null)
    setWav2vecResult(null)
    setAasistResult(null)
    setWav2vec2V2Result(null)
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
          if (track.kind !== Track.Kind.Audio) {
            return
          }

          console.log(
            'LiveKit remote audio track subscribed:',
            participant.identity,
          )

          // In the single-browser Simulate -> Live flow we already have the
          // exact caller MediaStream and feed that stream directly into the
          // detector below. Do not start a second Web Audio pipeline from
          // TrackSubscribed; that caused duplicate/stale analysis sessions.
          if (session?.mode === 'live') {
            return
          }

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
        },
      )

      /*
       * Authenticate with the detector before joining
       * the LiveKit room.
       */
      await connectAnalysisWebSocket()

      await room.connect(
        data.url,
        data.token,
      )

      /*
       * For the in-app Simulate → Live demo, create a second
       * LiveKit participant that publishes the microphone.
       * The receiver analyzes that participant as REMOTE caller audio.
       */
      if (session?.mode === 'live') {
        await startSimulatedCaller(data.url)

        const simulatedCallerStream =
          mediaStreamRef.current

        const simulatedCallerTrack =
          simulatedCallerStream?.getAudioTracks()[0]

        if (!simulatedCallerStream || !simulatedCallerTrack) {
          throw new Error(
            'Simulated caller microphone stream is unavailable.',
          )
        }

        /*
         * IMPORTANT: feed the actual microphone MediaStream directly into the
         * detector. The same stream is also published to LiveKit, so the
         * single-browser Simulate -> Live call is deterministic.
         */
        await startRemoteCallerAnalysis(simulatedCallerStream)
      } else {
        await startLocalMicrophone()
      }

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

    setStatus('idle')
    }

  useEffect(() => {
    if (session?.status === 'connected') {
      void startCallAnalysis()
    }

    return () => {
      void cleanupCall()
      void cleanupStandaloneResources()
    }
    // The Live tab is mounted for the connected call session.
    // Keep this effect stable so the LiveKit room is not restarted on
    // every detector result/state update.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  const wav2vec2V2Fake =
    wav2vec2V2Result?.fake_probability

  const ensembleScore =
    typeof wavFake === 'number' &&
    typeof aasistFake === 'number' &&
    typeof wav2vec2V2Fake === 'number'
      ? wavFake * 0.30 +
        aasistFake * 0.30 +
        wav2vec2V2Fake * 0.40
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

  if (isStandaloneLive) {
    return (
      <div className="max-w-5xl mx-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-semibold text-white">
            Live Voice Analysis
          </h2>

          <p className="mt-1 text-sm text-gray-400">
            Analyze your microphone directly with VoiceGuard's
            three-model deepfake detection ensemble.
          </p>
        </div>

        {standaloneStatus === 'idle' && (
          <div className="rounded-3xl border border-gray-800 bg-gray-900 p-10 text-center">
            <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-full bg-indigo-600/20 text-5xl">
              🎙️
            </div>

            <h3 className="mt-6 text-2xl font-bold text-white">
              Ready for Live Analysis
            </h3>

            <p className="mx-auto mt-3 max-w-xl text-sm text-gray-400">
              Start Live Analysis to use your microphone.
              VoiceGuard will continuously analyze the audio
              using Wav2Vec2 Spoof, Official AASIST, and
              Wav2Vec2 v2.
            </p>

            <button
              onClick={() => {
                void startStandaloneLiveAnalysis()
              }}
              className="mt-7 rounded-xl bg-indigo-600 px-7 py-3 font-semibold text-white hover:bg-indigo-500"
            >
              Start Live Analysis
            </button>
          </div>
        )}

        {standaloneStatus === 'starting' && (
          <div className="rounded-3xl border border-gray-800 bg-gray-900 p-10 text-center">
            <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-indigo-600/20 text-4xl">
              🎙️
            </div>

            <h3 className="mt-5 text-xl font-semibold text-white">
              Starting Live Analysis…
            </h3>

            <p className="mt-2 text-sm text-gray-400">
              Connecting microphone and VoiceGuard analysis.
            </p>
          </div>
        )}

        {standaloneStatus === 'error' && (
          <div className="rounded-3xl border border-red-900 bg-red-950/30 p-8 text-center">
            <div className="text-4xl">⚠️</div>

            <h3 className="mt-4 text-xl font-semibold text-red-300">
              Live Analysis Could Not Start
            </h3>

            <p className="mt-2 text-sm text-red-200/70">
              {error || 'Unable to start live analysis.'}
            </p>

            <button
              onClick={() => {
                setError(null)
                setStandaloneStatus('idle')
              }}
              className="mt-6 rounded-xl border border-gray-700 bg-gray-900 px-6 py-3 text-sm font-medium text-white hover:bg-gray-800"
            >
              Try Again
            </button>
          </div>
        )}

        {(standaloneStatus === 'live' ||
          standaloneStatus === 'ended') && (
          <>
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-xs uppercase tracking-wider text-gray-500">
                      Microphone
                    </div>

                    <div className="mt-2 text-xl font-semibold text-white">
                      {standaloneStatus === 'live'
                        ? 'Live Analysis Active'
                        : 'Analysis Completed'}
                    </div>

                    <div className="mt-2 flex items-center gap-2 text-sm">
                      {standaloneStatus === 'live' ? (
                        <>
                          <span className="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-red-500" />
                          <span className="font-medium text-red-300">
                            RECORDING & ANALYZING
                          </span>
                        </>
                      ) : (
                        <span className="text-gray-400">
                          Detector disconnected
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="text-4xl">
                    {standaloneStatus === 'live' ? '🎙️' : '✓'}
                  </div>
                </div>

                {standaloneStatus === 'live' && (
                  <>
                    <div className="mt-7 rounded-2xl border border-indigo-500/30 bg-gray-950 px-4 py-5">
                      <div className="mb-3 flex items-center justify-between text-xs">
                        <span className="font-medium text-gray-300">
                          Live microphone waveform
                        </span>
                        <span className="text-green-400">
                          ● Voice input active
                        </span>
                      </div>

                      <div className="flex h-24 items-center justify-center gap-1 overflow-hidden">
                        {standaloneWaveform.map((level, index) => (
                          <span
                            key={index}
                            className="w-1.5 rounded-full bg-indigo-400 transition-[height,opacity] duration-75"
                            style={{
                              height: `${Math.max(8, level * 88)}px`,
                              opacity: Math.max(0.35, level),
                            }}
                          />
                        ))}
                      </div>
                    </div>

                    <button
                      onClick={() => {
                        void endStandaloneLiveAnalysis()
                      }}
                      className="mt-6 w-full rounded-xl bg-red-600 px-5 py-4 text-base font-bold text-white shadow-lg shadow-red-950/30 hover:bg-red-500"
                    >
                      ■ End Live Analysis
                    </button>
                  </>
                )}
              </div>

              <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6">
                <div className="text-sm font-semibold text-white">
                  Live Impersonation Risk
                </div>

                <div className="mt-6 text-center">
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
              </div>
            </div>

            <div className="mt-6 rounded-2xl border border-gray-800 bg-gray-900 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-semibold text-white">
                    Three-Model Detection
                  </div>

                  <div className="mt-1 text-xs text-gray-500">
                    Production ensemble: 30% + 30% + 40%
                  </div>
                </div>

                <div className="text-xs font-semibold text-indigo-300">
                  {ensembleScore !== null
                    ? `${Math.round(ensembleScore * 100)}% fake`
                    : 'Waiting'}
                </div>
              </div>

              <div className="mt-5 grid gap-3 md:grid-cols-3">
                <div className="rounded-xl border border-gray-800 bg-gray-950 p-4">
                  <div className="text-xs text-gray-500">
                    Wav2Vec2 Spoof · 30%
                  </div>

                  <div className="mt-2 text-2xl font-bold text-white">
                    {typeof wavFake === 'number'
                      ? `${Math.round(wavFake * 100)}%`
                      : '—'}
                  </div>

                  <div className="mt-1 text-xs text-gray-500">
                    fake probability
                  </div>
                </div>

                <div className="rounded-xl border border-gray-800 bg-gray-950 p-4">
                  <div className="text-xs text-gray-500">
                    Official AASIST · 30%
                  </div>

                  <div className="mt-2 text-2xl font-bold text-white">
                    {typeof aasistFake === 'number'
                      ? `${Math.round(aasistFake * 100)}%`
                      : '—'}
                  </div>

                  <div className="mt-1 text-xs text-gray-500">
                    fake probability
                  </div>
                </div>

                <div className="rounded-xl border border-gray-800 bg-gray-950 p-4">
                  <div className="text-xs text-gray-500">
                    Wav2Vec2 v2 · 40%
                  </div>

                  <div className="mt-2 text-2xl font-bold text-white">
                    {typeof wav2vec2V2Fake === 'number'
                      ? `${Math.round(wav2vec2V2Fake * 100)}%`
                      : '—'}
                  </div>

                  <div className="mt-1 text-xs text-gray-500">
                    fake probability
                  </div>
                </div>
              </div>

              {secondsAnalyzed !== null && (
                <div className="mt-5 flex justify-between text-sm">
                  <span className="text-gray-400">
                    Audio analyzed
                  </span>

                  <span className="text-white">
                    {secondsAnalyzed.toFixed(2)} sec
                  </span>
                </div>
              )}
            </div>

            {standaloneStatus === 'ended' &&
              standaloneRecordingUrl && (
                <div className="mt-6 rounded-2xl border border-gray-800 bg-gray-900 p-6">
                  <div className="text-sm font-semibold text-white">
                    Recorded Voice
                  </div>

                  <p className="mt-1 text-xs text-gray-500">
                    Your standalone Live Analysis recording.
                  </p>

                  <audio
                    className="mt-4 w-full"
                    controls
                    src={standaloneRecordingUrl}
                  />

                  <a
                    href={standaloneRecordingUrl}
                    download="voiceguard-live-recording.webm"
                    className="mt-4 inline-block rounded-xl border border-gray-700 px-5 py-2.5 text-sm font-medium text-white hover:bg-gray-800"
                  >
                    Download Recorded Voice
                  </a>
                </div>
              )}

            {standaloneStatus === 'ended' && (
              <button
                onClick={() => {
                  if (standaloneRecordingUrlRef.current) {
                    URL.revokeObjectURL(
                      standaloneRecordingUrlRef.current,
                    )
                    standaloneRecordingUrlRef.current = null
                  }

                  setStandaloneRecordingUrl(null)
                  setStandaloneStatus('idle')
                  setRiskScore(null)
                  setVerdict(null)
                  setSecondsAnalyzed(null)
                  setWav2vecResult(null)
                  setAasistResult(null)
                  setWav2vec2V2Result(null)
                  setError(null)
                }}
                className="mt-6 rounded-xl border border-gray-700 px-6 py-3 text-sm font-medium text-white hover:bg-gray-800"
              >
                Start New Analysis
              </button>
            )}
          </>
        )}

        {error &&
          standaloneStatus !== 'error' && (
            <div className="mt-6 rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
              {error}
            </div>
          )}
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
    return (
      <div className="max-w-3xl mx-auto">
        <div className="rounded-3xl border border-gray-800 bg-gray-900 p-10 text-center">
          <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-gray-700/30 text-4xl">
            ☎
          </div>

          <h2 className="mt-5 text-2xl font-bold text-white">
            Call Ended
          </h2>

          <p className="mt-2 text-gray-400">
            The VoiceGuard call session has ended.
          </p>
        </div>
      </div>
    )
  }

  if (!connected && status !== 'error') {
    return (
      <div className="max-w-5xl mx-auto">
        <div className="rounded-2xl border border-gray-800 bg-gray-900 p-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-white">
                Live Voice Analysis
              </h2>
              <p className="mt-1 text-sm text-gray-400">
                Connecting to the accepted caller and waiting for remote audio…
              </p>
            </div>
            <span className="text-xs font-semibold text-yellow-400">
              ● CONNECTING
            </span>
          </div>
          {error && (
            <div className="mt-5 rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
              {error}
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">
            Live Voice Analysis
          </h2>

          <p className="text-sm text-gray-400">
            VoiceGuard analyzes the remote caller's voice in real time using LiveKit.
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
                {typeof wav2vec2V2Fake === 'number'
                  ? `${Math.round(wav2vec2V2Fake * 100)}% fake`
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
        riskScore >= 60 && (
          <div className="mt-6 rounded-2xl border border-red-800 bg-red-950/40 p-6">
            <div className="flex items-start gap-4">
              <div className="text-3xl">🚨</div>

              <div className="flex-1">
                <h3 className="text-lg font-semibold text-red-300">
                  High-Risk Voice Detected
                </h3>

                <p className="mt-2 text-sm text-red-200/80">
                  VoiceGuard detected a high probability
                  of AI-generated or cloned voice activity.
                </p>

                <div className="mt-5 flex flex-wrap gap-3">
                  <button
                    onClick={() =>
                      window.alert(
                        'Secondary verification recommended: verify the caller through an independent trusted channel.',
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
