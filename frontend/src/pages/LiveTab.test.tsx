import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import LiveTab from './LiveTab'
import { clearToken } from '../config/apiConfig'

describe('LiveTab', () => {
  beforeEach(() => {
    cleanup()
    clearToken()
    vi.restoreAllMocks()
  })

  it('renders the live voice analysis screen', () => {
    render(
      <LiveTab
        session={null}
        onEndCall={vi.fn()}
      />,
    )

    expect(
      screen.getByText(/live voice analysis/i),
    ).toBeTruthy()
  })

  it('renders caller information for a connected call session', () => {
    render(
      <LiveTab
        session={{
          mode: 'live',
          callerName: 'Test Caller',
          phone: '+91 98765 43210',
          purpose: 'Test Call',
          status: 'connected',
          startTime: new Date().toISOString(),
        }}
        onEndCall={vi.fn()}
      />,
    )

    expect(
      screen.getByText('+91 98765 43210'),
    ).toBeTruthy()

    expect(
      screen.getByText(/test caller/i),
    ).toBeTruthy()
  })
})
