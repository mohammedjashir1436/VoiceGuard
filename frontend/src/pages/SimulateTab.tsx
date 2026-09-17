
import { useState } from 'react'

export type SimulationMode = 'live' | 'generate'

export type SimulateCallData = {
  mode: SimulationMode
  callerName: string
  phone: string
  purpose: string
}

type SimulateTabProps = {
  onStartCall: (data: SimulateCallData) => void
}

export default function SimulateTab({
  onStartCall,
}: SimulateTabProps) {
  const [mode, setMode] = useState<SimulationMode>('live')
  const [callerName, setCallerName] = useState('')
  const [phone, setPhone] = useState('')
  const [purpose, setPurpose] = useState('General Call')

  const canCall =
    callerName.trim().length > 0 &&
    phone.trim().length > 0

  const handleCall = () => {
    if (!canCall) return

    onStartCall({
      mode,
      callerName: callerName.trim(),
      phone: phone.trim(),
      purpose,
    })
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-white mb-1">
          Simulate Call
        </h2>

        <p className="text-sm text-gray-400">
          Create a simulated incoming call and test VoiceGuard's
          voice-security workflow.
        </p>
      </div>

      {/* Simulation mode */}
      <div className="grid gap-4 md:grid-cols-2 mb-6">
        <button
          type="button"
          onClick={() => setMode('live')}
          className={`rounded-2xl border p-6 text-left transition ${
            mode === 'live'
              ? 'border-indigo-500 bg-indigo-950/40'
              : 'border-gray-800 bg-gray-900 hover:border-gray-700'
          }`}
        >
          <div className="text-3xl mb-3">🎙️</div>

          <h3 className="text-white font-semibold">
            Simulate Live
          </h3>

          <p className="text-sm text-gray-400 mt-2">
            Simulate a real-time caller whose voice will be analyzed
            through the Live detection pipeline.
          </p>

          {mode === 'live' && (
            <div className="mt-4 text-xs text-indigo-400 font-medium">
              SELECTED
            </div>
          )}
        </button>

        <button
          type="button"
          onClick={() => setMode('generate')}
          className={`rounded-2xl border p-6 text-left transition ${
            mode === 'generate'
              ? 'border-indigo-500 bg-indigo-950/40'
              : 'border-gray-800 bg-gray-900 hover:border-gray-700'
          }`}
        >
          <div className="text-3xl mb-3">🤖</div>

          <h3 className="text-white font-semibold">
            Simulate Generate
          </h3>

          <p className="text-sm text-gray-400 mt-2">
            Simulate a call where the caller voice comes from the
            Generate/TTS or cloning workflow.
          </p>

          {mode === 'generate' && (
            <div className="mt-4 text-xs text-indigo-400 font-medium">
              SELECTED
            </div>
          )}
        </button>
      </div>

      {/* Caller details */}
      <div className="rounded-2xl border border-gray-800 bg-gray-900 p-6">
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-white">
            Caller Details
          </h3>

          <p className="text-xs text-gray-500 mt-1">
            These details will appear on the incoming-call screen.
          </p>
        </div>

        <div className="space-y-5">
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Caller Name
            </label>

            <input
              value={callerName}
              onChange={(e) => setCallerName(e.target.value)}
              placeholder="e.g. Ahmed Khan"
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-4 py-3 text-white placeholder-gray-600 outline-none focus:border-indigo-500"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Phone Number
            </label>

            <input
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+91 XXXXX XXXXX"
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-4 py-3 text-white placeholder-gray-600 outline-none focus:border-indigo-500"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Call Purpose
            </label>

            <select
              value={purpose}
              onChange={(e) => setPurpose(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-4 py-3 text-white outline-none focus:border-indigo-500"
            >
              <option>General Call</option>
              <option>Fund Transfer</option>
              <option>Privileged Approval</option>
              <option>Confidential Information</option>
              <option>Account Verification</option>
            </select>
          </div>

          <button
            type="button"
            onClick={handleCall}
            disabled={!canCall}
            className="w-full rounded-lg bg-indigo-600 px-4 py-3 font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-gray-700 disabled:text-gray-500"
          >
            📞 Call
          </button>
        </div>
      </div>

      {/* Current selection */}
      <div className="mt-4 rounded-lg border border-gray-800 bg-gray-950 p-4">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500">
            Simulation mode
          </span>

          <span className="font-medium text-gray-300">
            {mode === 'live' ? 'Live Voice' : 'Generated Voice'}
          </span>
        </div>
      </div>
    </div>
  )
}

