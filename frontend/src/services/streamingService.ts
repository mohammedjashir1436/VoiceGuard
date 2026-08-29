/**
 * Live microphone streaming to WS /ws/stream.
 *
 * Protocol:
 * 1. Open WebSocket
 * 2. Send JWT as first message
 * 3. Wait for auth_ok
 * 4. Request microphone access
 * 5. Capture microphone using AudioWorklet
 * 6. Convert audio to mono 16 kHz PCM int16
 * 7. Send PCM binary frames to backend
 * 8. Receive real/fake detection events
 */

import {
  API_BASE,
  getWsUrl,
  getToken,
} from '../config/apiConfig'

export type StreamEvent = {
  timestamp_ms: number
  window_id: number
  label: 'real' | 'fake'
  confidence: number
  model: string | null
  final?: boolean
}

export type StreamStatus =
  | 'idle'
  | 'connecting'
  | 'authenticating'
  | 'live'
  | 'busy'
  | 'closed'
  | 'error'

export type StreamHandlers = {
  onEvent: (event: StreamEvent) => void
  onStatus: (status: StreamStatus) => void
  onError: (message: string) => void
}

const TARGET_RATE = 16000

/**
 * Approximately 256 ms of audio at 16 kHz.
 */
const CHUNK_SAMPLES = 4096

/**
 * AudioWorklet that forwards microphone frames
 * from the audio thread to the main thread.
 */
const WORKLET_SOURCE = `
class VgPcmForwarder extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]
    const channel = input && input[0]

    if (channel) {
      this.port.postMessage(channel.slice(0))
    }

    return true
  }
}

registerProcessor('vg-pcm-forwarder', VgPcmForwarder)
`

/**
 * Resample audio to 16 kHz using linear interpolation.
 *
 * Browsers commonly provide 44.1 kHz or 48 kHz audio,
 * while VoiceGuard backend expects 16 kHz mono PCM.
 */
export const resampleTo16k = (
  input: Float32Array,
  fromRate: number,
): Float32Array => {
  if (fromRate === TARGET_RATE) {
    return input
  }

  if (input.length === 0) {
    return new Float32Array(0)
  }

  const outLen = Math.floor(
    (input.length * TARGET_RATE) / fromRate,
  )

  const output = new Float32Array(outLen)
  const ratio = fromRate / TARGET_RATE

  for (let i = 0; i < outLen; i++) {
    const position = i * ratio

    const index0 = Math.floor(position)

    const index1 = Math.min(
      index0 + 1,
      input.length - 1,
    )

    const fraction = position - index0

    output[i] =
      input[index0] +
      (input[index1] - input[index0]) * fraction
  }

  return output
}

/**
 * Convert Float32 audio [-1, 1] to signed 16-bit PCM.
 */
export const floatToInt16 = (
  input: Float32Array,
): Int16Array => {
  const output = new Int16Array(input.length)

  for (let i = 0; i < input.length; i++) {
    const value = Math.max(
      -1,
      Math.min(1, input[i]),
    )

    output[i] =
      value < 0
        ? value * 0x8000
        : value * 0x7fff
  }

  return output
}

export class LiveMicStream {
  private ws: WebSocket | null = null

  private ctx: AudioContext | null = null

  private mediaStream: MediaStream | null = null

  private pending: Float32Array[] = []

  private pendingSamples = 0

  private stopped = false

  constructor(
    private handlers: StreamHandlers,
  ) {}

  async start(): Promise<void> {
    this.stopped = false

    this.handlers.onStatus('connecting')

    console.log(
      '[VoiceGuard] Starting live microphone stream',
    )

    const wsUrl = getWsUrl(
      `${API_BASE}/ws/stream`,
    )

    console.log(
      '[VoiceGuard] WebSocket URL:',
      wsUrl,
    )

    const ws = new WebSocket(wsUrl)

    this.ws = ws

    ws.binaryType = 'arraybuffer'

    /**
     * WebSocket opened.
     *
     * Send JWT as first message.
     */
    ws.onopen = () => {
      console.log(
        '[VoiceGuard] WebSocket OPEN',
      )

      this.handlers.onStatus(
        'authenticating',
      )

      const token = getToken()

      console.log(
        '[VoiceGuard] Sending authentication:',
        token ? 'TOKEN PRESENT' : 'NO TOKEN',
      )

      ws.send(
        JSON.stringify({
          token,
        }),
      )
    }

    /**
     * Handle messages from backend.
     */
    ws.onmessage = (message) => {
      console.log(
        '[VoiceGuard] WS MESSAGE:',
        message.data,
      )

      let data: Record<string, unknown>

      try {
        data = JSON.parse(
          message.data as string,
        )
      } catch {
        console.warn(
          '[VoiceGuard] Invalid WS message',
        )

        return
      }

      /**
       * Authentication successful.
       */
      if (data.type === 'auth_ok') {
        console.log(
          '[VoiceGuard] AUTH OK - starting microphone',
        )

        void this.startMicrophone()

        return
      }

      /**
       * Backend authentication/general error.
       */
      if (
        data.type === 'error' ||
        data.type === 'auth_error'
      ) {
        console.error(
          '[VoiceGuard] Server error:',
          data,
        )

        this.handlers.onError(
          String(
            data.message ??
              'Live analysis error',
          ),
        )

        return
      }

      /**
       * Detection event.
       */
      if (
        typeof data.label === 'string' &&
        typeof data.confidence === 'number'
      ) {
        console.log(
          '[VoiceGuard] DETECTION:',
          data,
        )

        this.handlers.onEvent(
          data as unknown as StreamEvent,
        )
      }
    }

    /**
     * WebSocket error.
     */
    ws.onerror = (error) => {
      console.error(
        '[VoiceGuard] WebSocket ERROR:',
        error,
      )

      if (this.stopped) {
        return
      }

      this.handlers.onStatus('error')

      this.handlers.onError(
        'Could not reach the live-analysis stream.',
      )
    }

    /**
     * WebSocket closed.
     */
    ws.onclose = (event) => {
      console.log(
        '[VoiceGuard] WebSocket CLOSED:',
        'code=',
        event.code,
        'reason=',
        event.reason,
        'wasClean=',
        event.wasClean,
      )

      if (this.stopped) {
        return
      }

      /**
       * Server is busy.
       */
      if (event.code === 1013) {
        this.handlers.onStatus('busy')

        this.handlers.onError(
          'All live-analysis slots are busy. Try again in a minute.',
        )
      }

      /**
       * Authentication rejected.
       */
      else if (event.code === 1008) {
        this.handlers.onStatus('error')

        this.handlers.onError(
          'Stream rejected. Log in again and retry.',
        )
      }

      /**
       * Other close.
       */
      else {
        this.handlers.onStatus('closed')
      }

      this.teardownAudio()
    }
  }

  /**
   * Request microphone access and initialize AudioWorklet.
   */
  private async startMicrophone(): Promise<void> {
    try {
      console.log(
        '[VoiceGuard] Requesting microphone permission...',
      )

      this.mediaStream =
        await navigator.mediaDevices.getUserMedia(
          {
            audio: {
              channelCount: 1,
              echoCancellation: true,
              noiseSuppression: false,
              autoGainControl: false,
            },
          },
        )

      console.log(
        '[VoiceGuard] MICROPHONE ACCESS OK',
      )
    } catch (error) {
      console.error(
        '[VoiceGuard] Microphone error:',
        error,
      )

      this.handlers.onError(
        'Microphone access denied. Allow microphone permission and retry.',
      )

      this.stop()

      return
    }

    try {
      /**
       * Create audio context.
       */
      const ctx = new AudioContext()

      this.ctx = ctx

      console.log(
        '[VoiceGuard] AudioContext sample rate:',
        ctx.sampleRate,
      )

      /**
       * Create inline AudioWorklet.
       */
      const workletUrl =
        URL.createObjectURL(
          new Blob(
            [WORKLET_SOURCE],
            {
              type: 'application/javascript',
            },
          ),
        )

      try {
        await ctx.audioWorklet.addModule(
          workletUrl,
        )

        console.log(
          '[VoiceGuard] AudioWorklet loaded',
        )
      } finally {
        URL.revokeObjectURL(
          workletUrl,
        )
      }

      /**
       * Connect microphone to AudioWorklet.
       */
      const source =
        ctx.createMediaStreamSource(
          this.mediaStream!,
        )

      const node =
        new AudioWorkletNode(
          ctx,
          'vg-pcm-forwarder',
        )

      /**
       * Receive audio frames from worklet.
       */
      node.port.onmessage = (
        event: MessageEvent<Float32Array>,
      ) => {
        this.push(
          event.data,
          ctx.sampleRate,
        )
      }

      source.connect(node)

      /**
       * Keep AudioWorklet alive without
       * playing microphone audio to the user.
       */
      const mute =
        ctx.createGain()

      mute.gain.value = 0

      node.connect(mute)

      mute.connect(ctx.destination)

      /**
       * Some browsers create AudioContext
       * in suspended state.
       */
      if (
        ctx.state === 'suspended'
      ) {
        await ctx.resume()
      }

      console.log(
        '[VoiceGuard] LIVE MICROPHONE STREAM READY',
      )

      this.handlers.onStatus('live')
    } catch (error) {
      console.error(
        '[VoiceGuard] Audio setup error:',
        error,
      )

      this.handlers.onError(
        'Could not initialize microphone audio.',
      )

      this.stop()
    }
  }

  /**
   * Receive microphone audio frames.
   *
   * Buffer enough audio, resample to 16 kHz,
   * convert to int16 PCM, and send through WS.
   */
  private push(
    frame: Float32Array,
    sampleRate: number,
  ): void {
    if (
      !this.ws ||
      this.ws.readyState !== WebSocket.OPEN
    ) {
      return
    }

    this.pending.push(frame)

    this.pendingSamples += frame.length

    console.log(
      '[VoiceGuard] AUDIO FRAME:',
      frame.length,
      'samples at',
      sampleRate,
      'Hz',
    )

    /**
     * Calculate how many samples are required
     * at the browser's actual microphone rate.
     */
    const requiredSamples =
      CHUNK_SAMPLES *
      (sampleRate / TARGET_RATE)

    if (
      this.pendingSamples <
      requiredSamples
    ) {
      return
    }

    /**
     * Combine buffered Float32 frames.
     */
    const joined =
      new Float32Array(
        this.pendingSamples,
      )

    let offset = 0

    for (const chunk of this.pending) {
      joined.set(
        chunk,
        offset,
      )

      offset += chunk.length
    }

    this.pending = []

    this.pendingSamples = 0

    console.log(
      '[VoiceGuard] BUFFER READY:',
      joined.length,
      'samples',
    )

    /**
     * Resample to 16 kHz.
     */
    const resampled =
      resampleTo16k(
        joined,
        sampleRate,
      )

    console.log(
      '[VoiceGuard] RESAMPLED:',
      resampled.length,
      'samples',
    )

    /**
     * Convert Float32 [-1, 1]
     * to signed 16-bit PCM.
     */
    const pcm =
      floatToInt16(resampled)

    console.log(
      '[VoiceGuard] SENDING PCM:',
      pcm.byteLength,
      'bytes',
    )

    try {
      /**
       * IMPORTANT:
       *
       * Newer TypeScript versions may type
       * Int16Array.buffer as ArrayBufferLike.
       *
       * WebSocket.send() requires a BufferSource
       * backed by ArrayBuffer.
       *
       * Creating a new Uint8Array gives us a
       * browser-compatible byte view.
       */
      const bytes = new Uint8Array(
        pcm.byteLength,
      )

      bytes.set(
        new Uint8Array(
          pcm.buffer,
          pcm.byteOffset,
          pcm.byteLength,
        ),
      )

      this.ws.send(bytes)

      console.log(
        '[VoiceGuard] PCM SENT SUCCESSFULLY',
      )
    } catch (error) {
      console.error(
        '[VoiceGuard] PCM SEND ERROR:',
        error,
      )
    }
  }

  /**
   * Stop live microphone analysis.
   */
  stop(): void {
    console.log(
      '[VoiceGuard] STOP',
    )

    this.stopped = true

    this.teardownAudio()

    if (
      this.ws &&
      this.ws.readyState <=
        WebSocket.OPEN
    ) {
      this.ws.close(1000)
    }

    this.ws = null

    this.handlers.onStatus('idle')
  }

  /**
   * Release microphone and AudioContext.
   */
  private teardownAudio(): void {
    this.mediaStream
      ?.getTracks()
      .forEach((track) => {
        track.stop()
      })

    this.mediaStream = null

    if (this.ctx) {
      void this.ctx
        .close()
        .catch(() => undefined)
    }

    this.ctx = null

    this.pending = []

    this.pendingSamples = 0
  }
}