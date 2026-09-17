import { useState } from 'react'

import DetectTab from './pages/DetectTab'
import LiveTab from './pages/LiveTab'
import CallTab from './pages/CallTab'
import GenerateTab from './pages/GenerateTab'
import SimulateTab, {
  type SimulateCallData,
} from './pages/SimulateTab'
import VerifyTab from './pages/VerifyTab'
import ResultsTab from './pages/ResultsTab'
import LoginControl from './components/LoginControl'

type Tab =
  | 'detect'
  | 'live'
  | 'generate'
  | 'simulate'
  | 'call'
  | 'verify'
  | 'results'

export type CallSession = {
  mode: 'live' | 'generate'
  callerName: string
  phone: string
  purpose: string
  status: 'ringing' | 'connected' | 'rejected' | 'ended'
  startTime: string
}

export type DetectionModelResult = {
  label?: string
  confidence?: number
  fake_probability?: number
  weight?: number
}

export type CallRecording = {
  url: string
  blob: Blob
  durationSeconds: number
}

export type CallDetection = {
  riskScore: number | null
  verdict: 'real' | 'fake' | 'unknown' | null
  secondsAnalyzed: number | null
  wav2vec2: DetectionModelResult | null
  aasist: DetectionModelResult | null
  wav2vec2_v2: DetectionModelResult | null
  analysisConnected: boolean
}

const EMPTY_DETECTION: CallDetection = {
  riskScore: null,
  verdict: null,
  secondsAnalyzed: null,
  wav2vec2: null,
  aasist: null,
  wav2vec2_v2: null,
  analysisConnected: false,
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('detect')
  const [callSession, setCallSession] = useState<CallSession | null>(null)
  const [callDetection, setCallDetection] =
    useState<CallDetection>(EMPTY_DETECTION)
  const [callRecording, setCallRecording] =
    useState<CallRecording | null>(null)

  const tabs: { id: Tab; label: string }[] = [
    { id: 'detect', label: 'Detect' },
    { id: 'live', label: 'Live' },
    { id: 'generate', label: 'Generate' },
    { id: 'simulate', label: 'Simulate' },
    { id: 'call', label: 'Call' },
    { id: 'verify', label: 'Verify' },
    { id: 'results', label: 'Results' },
  ]

  const handleStartSimulatedCall = (data: SimulateCallData) => {
    setCallDetection(EMPTY_DETECTION)

    if (callRecording) {
      URL.revokeObjectURL(callRecording.url)
    }

    setCallRecording(null)

    const session: CallSession = {
      mode: data.mode,
      callerName: data.callerName,
      phone: data.phone,
      purpose: data.purpose,
      status: 'ringing',
      startTime: new Date().toISOString(),
    }

    setCallSession(session)
    setActiveTab('call')
  }

  const handleAcceptCall = () => {
    if (!callSession) return

    setCallSession((previous) => {
      if (!previous) return previous
      return { ...previous, status: 'connected' }
    })

    if (callSession.mode === 'generate') {
      setActiveTab('generate')
    } else {
      setActiveTab('live')
    }
  }

  const handleRejectCall = () => {
    if (!callSession) return

    setCallSession((previous) => {
      if (!previous) return previous
      return { ...previous, status: 'rejected' }
    })
  }

  const handleEndCall = () => {
    if (!callSession) return

    setCallSession((previous) => {
      if (!previous) return previous
      return { ...previous, status: 'ended' }
    })

    setActiveTab('call')
  }

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">VG</span>
            </div>

            <span className="text-white font-semibold text-lg">
              VoiceGuard
            </span>

            <span className="text-gray-500 text-sm hidden sm:block">
              Real-time Voice Deepfake Detection
            </span>
          </div>

          <div className="flex items-center gap-3">
            <LoginControl />
            <span className="text-xs text-gray-600 font-mono">
              v1.0.0
            </span>
          </div>
        </div>
      </header>

      <nav className="bg-gray-900 border-b border-gray-800 px-6">
        <div className="max-w-5xl mx-auto flex gap-1 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-5 py-3 text-sm font-medium transition-colors border-b-2 whitespace-nowrap ${
                activeTab === tab.id
                  ? 'text-indigo-400 border-indigo-500'
                  : 'text-gray-400 border-transparent hover:text-gray-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="flex-1 px-6 py-8">
        <div className="max-w-5xl mx-auto">
          {activeTab === 'detect' && <DetectTab />}

          {activeTab === 'live' && (
            <LiveTab
              session={callSession}
              onEndCall={handleEndCall}
              onDetectionChange={setCallDetection}
              onCallRecording={setCallRecording}
            />
          )}

          {activeTab === 'generate' && <GenerateTab />}

          {activeTab === 'simulate' && (
            <SimulateTab
              onStartCall={handleStartSimulatedCall}
            />
          )}

          {activeTab === 'call' && (
            <CallTab
              session={callSession}
              onAccept={handleAcceptCall}
              onReject={handleRejectCall}
              onEndCall={handleEndCall}
              detection={callDetection}
              recording={callRecording}
            />
          )}

          {activeTab === 'verify' && <VerifyTab />}

          {activeTab === 'results' && <ResultsTab />}
        </div>
      </main>

      <footer className="bg-gray-900 border-t border-gray-800 px-6 py-3">
        <div className="max-w-5xl mx-auto flex items-center justify-between text-xs text-gray-600">
          <span>
            Canadian University Dubai · GP2 2025/26
          </span>

          <span>
            Apache 2.0 License
          </span>
        </div>
      </footer>
    </div>
  )
}