import React, { useState, useEffect, useRef, useCallback } from 'react';

// EEG frequency band colors
const BAND_COLORS = {
  delta: '#d4af37',  // Gold - deep sleep
  theta: '#f59e0b',  // Amber - meditation
  alpha: '#10b981',  // Green - relaxed
  beta: '#f59e0b',   // Amber - active
  gamma: '#ec4899',  // Pink - cognition
};

// EEG channel names for Muse
const EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10'];

const targetStates = [
  { id: 'generativity', name: 'Generativity', description: 'Creative flow & productive emergence', color: '#f59e0b' },
  { id: 'confidence', name: 'Confidence', description: 'Grounded self-assurance', color: '#10b981' },
  { id: 'enlightenment', name: 'Enlightenment', description: 'Clarity & insight', color: '#8b5cf6' },
  { id: 'peak_performance', name: 'Peak Performance', description: 'Optimal cognitive function', color: '#3b82f6' },
  { id: 'loving_awareness', name: 'Loving Awareness', description: 'Open-hearted presence', color: '#ec4899' },
];

// Spark visualization component
const Spark = ({ data, color, width = 120, height = 40 }) => {
  if (!data || data.length === 0) return null;

  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  const points = data.map((value, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((value - min) / range) * (height - 8) - 4;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} className="opacity-80">
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
};

// EEG Waveform display component
const EEGWaveform = ({ data, channel, color = '#10b981', width = 200, height = 40 }) => {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center" style={{ width, height }}>
        <div className="text-xs text-white/90">No signal</div>
      </div>
    );
  }

  // Normalize to display range
  const samples = data.slice(-200); // Last 200 samples
  const max = Math.max(...samples.map(Math.abs), 50);

  const points = samples.map((value, i) => {
    const x = (i / (samples.length - 1)) * width;
    const y = height / 2 - (value / max) * (height / 2 - 2);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="relative">
      <div className="absolute left-0 top-0 text-xs text-white/90 font-mono">{channel}</div>
      <svg width={width} height={height} className="mt-3">
        <line x1="0" y1={height/2} x2={width} y2={height/2} stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
        <polyline
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />
      </svg>
    </div>
  );
};

// Band power bar component
const BandPowerBar = ({ band, power, maxPower = 100 }) => {
  const percentage = Math.min((power / maxPower) * 100, 100);
  return (
    <div className="flex items-center gap-2">
      <div className="w-12 text-xs text-white/90 capitalize">{band}</div>
      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${percentage}%`,
            backgroundColor: BAND_COLORS[band] || '#d4af37'
          }}
        />
      </div>
      <div className="w-10 text-xs text-white/80 text-right tabular-nums">
        {power.toFixed(1)}
      </div>
    </div>
  );
};

// Band power time series chart - shows 5 minutes of history
const BandTimeSeries = ({ history, width = 500, height = 150 }) => {
  if (!history || history.length < 2) {
    return (
      <div className="flex items-center justify-center text-white/50 text-sm" style={{ width, height }}>
        Collecting data...
      </div>
    );
  }

  const bands = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
  const maxPower = Math.max(...history.flatMap(h => bands.map(b => h[b] || 0)), 50);
  const padding = { top: 10, right: 10, bottom: 25, left: 40 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Time range (last 5 minutes)
  const now = Date.now();
  const fiveMinutesAgo = now - 5 * 60 * 1000;
  const visibleHistory = history.filter(h => h.timestamp >= fiveMinutesAgo);

  if (visibleHistory.length < 2) {
    return (
      <div className="flex items-center justify-center text-white/50 text-sm" style={{ width, height }}>
        Collecting data...
      </div>
    );
  }

  const getX = (timestamp) => {
    const progress = (timestamp - fiveMinutesAgo) / (5 * 60 * 1000);
    return padding.left + progress * chartWidth;
  };

  const getY = (power) => {
    return padding.top + chartHeight - (power / maxPower) * chartHeight;
  };

  // Generate paths for each band
  const paths = bands.map(band => {
    const points = visibleHistory.map((h, i) => {
      const x = getX(h.timestamp);
      const y = getY(h[band] || 0);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
    return { band, path: points };
  });

  // Time labels
  const timeLabels = [0, 1, 2, 3, 4, 5].map(min => ({
    x: padding.left + (min / 5) * chartWidth,
    label: min === 0 ? 'now' : `-${5 - min}m`
  }));

  return (
    <svg width={width} height={height} className="overflow-visible">
      {/* Grid lines */}
      {[0, 25, 50, 75, 100].map(pct => (
        <g key={pct}>
          <line
            x1={padding.left}
            y1={padding.top + chartHeight * (1 - pct / 100)}
            x2={width - padding.right}
            y2={padding.top + chartHeight * (1 - pct / 100)}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="1"
          />
          <text
            x={padding.left - 5}
            y={padding.top + chartHeight * (1 - pct / 100) + 4}
            fill="rgba(255,255,255,0.4)"
            fontSize="9"
            textAnchor="end"
          >
            {Math.round(maxPower * pct / 100)}
          </text>
        </g>
      ))}

      {/* Time labels */}
      {timeLabels.map(({ x, label }) => (
        <text
          key={label}
          x={x}
          y={height - 5}
          fill="rgba(255,255,255,0.4)"
          fontSize="9"
          textAnchor="middle"
        >
          {label}
        </text>
      ))}

      {/* Band paths */}
      {paths.map(({ band, path }) => (
        <path
          key={band}
          d={path}
          fill="none"
          stroke={BAND_COLORS[band]}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity="0.8"
        />
      ))}

      {/* Legend */}
      <g transform={`translate(${padding.left}, ${padding.top - 2})`}>
        {bands.map((band, i) => (
          <g key={band} transform={`translate(${i * 70}, 0)`}>
            <rect
              x="0" y="-6" width="8" height="8" rx="2"
              fill={BAND_COLORS[band]}
              opacity="0.8"
            />
            <text x="12" y="0" fill="rgba(255,255,255,0.7)" fontSize="9">
              {band}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
};

// Signal quality indicator
const SignalQualityDot = ({ quality }) => {
  const getColor = (q) => {
    if (q >= 0.8) return '#10b981'; // Good
    if (q >= 0.5) return '#f59e0b'; // Medium
    return '#ef4444'; // Poor
  };

  return (
    <div
      className="w-2 h-2 rounded-full"
      style={{ backgroundColor: getColor(quality) }}
      title={`Quality: ${(quality * 100).toFixed(0)}%`}
    />
  );
};

// Status indicator
const StatusOrb = ({ status }) => {
  const colors = {
    connected: { bg: '#10b981', glow: 'rgba(16, 185, 129, 0.4)' },
    pending: { bg: '#9ca3af', glow: 'rgba(156, 163, 175, 0.3)' },
    error: { bg: '#ef4444', glow: 'rgba(239, 68, 68, 0.4)' },
  };

  const { bg, glow } = colors[status] || colors.pending;

  return (
    <div
      className="w-2 h-2 rounded-full"
      style={{
        backgroundColor: bg,
        boxShadow: `0 0 8px ${glow}, 0 0 16px ${glow}`
      }}
    />
  );
};

function formatTimeAgo(dateString) {
  if (!dateString) return 'Never';
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return `${diffDays}d ago`;
}

function formatDuration(seconds) {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Insight badge component
const InsightBadge = ({ insight }) => {
  const colors = {
    positive: { bg: 'rgba(16, 185, 129, 0.15)', border: 'rgba(16, 185, 129, 0.3)', text: '#10b981' },
    warning: { bg: 'rgba(245, 158, 11, 0.15)', border: 'rgba(245, 158, 11, 0.3)', text: '#f59e0b' },
    info: { bg: 'rgba(99, 102, 241, 0.15)', border: 'rgba(99, 102, 241, 0.3)', text: '#6366f1' }
  };
  const style = colors[insight.type] || colors.info;

  return (
    <div className="p-3 rounded-lg mb-2" style={{
      background: style.bg,
      border: `1px solid ${style.border}`
    }}>
      <div className="flex items-start gap-2">
        <span className="text-xs uppercase tracking-wide" style={{ color: style.text }}>
          {insight.category}
        </span>
      </div>
      <p className="text-sm text-white/80 mt-1">{insight.message}</p>
    </div>
  );
};

// Score ring component
const ScoreRing = ({ score, label, color }) => {
  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <svg width="80" height="80" className="transform -rotate-90">
        <circle
          cx="40" cy="40" r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="6"
        />
        <circle
          cx="40" cy="40" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute mt-5 text-lg font-light tabular-nums" style={{ fontFamily: "'Annex Latin', system-ui, sans-serif" }}>
        {score}
      </div>
      <div className="text-xs text-white/90 mt-1">{label}</div>
    </div>
  );
};

export default function App() {
  const [dataSources, setDataSources] = useState([]);
  const [healthData, setHealthData] = useState([]);
  const [trends, setTrends] = useState(null);
  const [todayContext, setTodayContext] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentHRV, setCurrentHRV] = useState(null);
  const [firstDataDate, setFirstDataDate] = useState(null);

  // Cognitive load per date (from calendar data)
  const [cognitiveLoads, setCognitiveLoads] = useState({});

  // EEG State
  const [eegStatus, setEegStatus] = useState({
    connected: false,
    streaming: false,
    state: 'disconnected',
    deviceAddress: null
  });
  const [eegData, setEegData] = useState({
    waveforms: [[], [], [], []], // 4 channels
    bandPowers: { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 },
    bandHistory: [], // Time series of band powers [{timestamp, delta, theta, alpha, beta, gamma}]
    signalQuality: [0, 0, 0, 0],
    metrics: {
      engagement: 0,
      relaxation: 0,
      focus: 0,
      meditation: 0
    }
  });
  const wsRef = useRef(null);
  const eegBufferRef = useRef([[], [], [], []]);
  const lastBandUpdateRef = useRef(0); // Track last band history update

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSession, setRecordingSession] = useState(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const recordingStartRef = useRef(null);
  const autoRecordTriggeredRef = useRef(false);

  useEffect(() => {
    async function fetchData() {
      try {
        const [statsRes, healthRes, trendsRes, todayRes, eegSessionsRes] = await Promise.all([
          fetch('/api/stats'),
          fetch('/api/health?days=14'),
          fetch('/api/trends?days=7'),
          fetch(`/api/daily/${new Date().toISOString().split('T')[0]}`),
          fetch('/api/eeg/sessions?days=365')
        ]);

        const stats = await statsRes.json();
        const health = await healthRes.json();
        const trendsData = await trendsRes.json();
        const today = await todayRes.json();
        const eegSessions = eegSessionsRes.ok ? await eegSessionsRes.json() : [];

        // Study start date: January 26, 2026 = Day 1
        const STUDY_START_DATE = new Date('2026-01-26');
        let firstDataDate = STUDY_START_DATE;

        setDataSources(stats.sources);
        setHealthData(health);
        setTrends(trendsData);
        setTodayContext(today);

        // Store first data date for day count calculation
        if (firstDataDate) {
          setFirstDataDate(firstDataDate);
        }

        // Set current HRV from latest data
        if (health.length > 0) {
          const latest = health[health.length - 1];
          setCurrentHRV(latest.hrv_rmssd);

          // Fetch cognitive load for each date in health data
          const dates = health.slice(-7).map(h => h.timestamp.split('T')[0]);
          const loadPromises = dates.map(async (date) => {
            try {
              const res = await fetch(`/api/analysis/daily/${date}`);
              if (res.ok) {
                const data = await res.json();
                return { date, load: data.scores?.cognitive_load || 0 };
              }
            } catch (e) {}
            return { date, load: null };
          });
          const loads = await Promise.all(loadPromises);
          const loadMap = {};
          loads.forEach(l => { loadMap[l.date] = l.load; });
          setCognitiveLoads(loadMap);
        }
      } catch (err) {
        console.error('Failed to fetch data:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  // Simulate live HRV variation
  useEffect(() => {
    if (currentHRV === null) return;
    const interval = setInterval(() => {
      setCurrentHRV(prev => prev + (Math.random() - 0.5) * 2);
    }, 3000);
    return () => clearInterval(interval);
  }, [currentHRV !== null]);

  // EEG API base URL (WebSocket server handles EEG since it receives the data)
  const EEG_API = 'http://localhost:5002';

  // Start recording function
  const startRecording = async () => {
    try {
      const res = await fetch(`${EEG_API}/api/eeg/record/start`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setIsRecording(true);
        setRecordingSession(data.session_id);
        recordingStartRef.current = Date.now();
        console.log('Recording started:', data.session_id);
      }
    } catch (err) {
      console.error('Failed to start recording:', err);
    }
  };

  // Stop recording function
  const stopRecording = async () => {
    try {
      const res = await fetch(`${EEG_API}/api/eeg/record/stop`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        console.log('Recording stopped:', data.session);
        setIsRecording(false);
        setRecordingSession(null);
        setRecordingDuration(0);
        recordingStartRef.current = null;
      }
    } catch (err) {
      console.error('Failed to stop recording:', err);
    }
  };

  // Poll EEG status from API
  useEffect(() => {
    const checkEegStatus = async () => {
      try {
        const res = await fetch(`${EEG_API}/api/eeg/status`);
        if (res.ok) {
          const data = await res.json();
          const isConnected = data.state === 'streaming' || data.state === 'recording';
          const wasConnected = eegStatus.connected;

          setEegStatus({
            connected: isConnected,
            streaming: isConnected,
            state: data.state || 'disconnected',
            deviceAddress: data.device_address
          });

          // Check if recording state from server
          if (data.current_session) {
            setIsRecording(true);
            setRecordingSession(data.current_session.session_id);
            if (!recordingStartRef.current) {
              recordingStartRef.current = new Date(data.current_session.start_time).getTime();
            }
          } else {
            // No active recording on server
            // AUTO-RECORD: Start recording when EEG is streaming but not recording
            if (isConnected && !autoRecordTriggeredRef.current) {
              console.log('EEG streaming without recording - auto-starting (research mode)');
              autoRecordTriggeredRef.current = true;
              startRecording();
            }
          }

          // Reset auto-record flag when disconnected
          if (!isConnected) {
            autoRecordTriggeredRef.current = false;
            setIsRecording(false);
            setRecordingSession(null);
            recordingStartRef.current = null;
          }
        }
      } catch (err) {
        // API not available, EEG not connected
        setEegStatus(prev => ({ ...prev, connected: false, state: 'unavailable' }));
      }
    };

    checkEegStatus();
    const interval = setInterval(checkEegStatus, 5000);
    return () => clearInterval(interval);
  }, [eegStatus.connected, isRecording]);

  // Recording duration timer
  useEffect(() => {
    if (!isRecording || !recordingStartRef.current) return;

    const updateDuration = () => {
      setRecordingDuration(Math.floor((Date.now() - recordingStartRef.current) / 1000));
    };

    updateDuration();
    const interval = setInterval(updateDuration, 1000);
    return () => clearInterval(interval);
  }, [isRecording]);

  // WebSocket connection for real-time EEG
  useEffect(() => {
    if (!eegStatus.streaming) {
      // Close existing connection if not streaming
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    // Connect to WebSocket server
    const connectWebSocket = () => {
      try {
        // Use Socket.IO client protocol
        const ws = new WebSocket('ws://localhost:5002/socket.io/?EIO=4&transport=websocket');

        ws.onopen = () => {
          console.log('EEG WebSocket connected');
          // Socket.IO handshake
          ws.send('40');
          // Start stream after connection
          setTimeout(() => {
            ws.send('42["start_stream",{}]');
          }, 100);
        };

        ws.onmessage = (event) => {
          const data = event.data;

          // Handle Socket.IO protocol messages
          if (data.startsWith('42')) {
            try {
              const payload = JSON.parse(data.slice(2));
              const [eventName, eventData] = payload;

              if (eventName === 'eeg_sample') {
                // Update waveform buffers
                const newWaveforms = [...eegBufferRef.current];
                if (eventData.eeg) {
                  eventData.eeg.forEach((value, ch) => {
                    newWaveforms[ch] = [...newWaveforms[ch].slice(-199), value];
                  });
                }
                eegBufferRef.current = newWaveforms;
                setEegData(prev => ({ ...prev, waveforms: newWaveforms }));
              } else if (eventName === 'signal_quality') {
                setEegData(prev => ({
                  ...prev,
                  signalQuality: eventData.channels || [0, 0, 0, 0]
                }));
              } else if (eventName === 'band_powers') {
                const now = Date.now();
                // Only add to history every 5 seconds to avoid too many points
                const shouldAddToHistory = now - lastBandUpdateRef.current >= 5000;

                setEegData(prev => {
                  let newHistory = prev.bandHistory;
                  if (shouldAddToHistory) {
                    lastBandUpdateRef.current = now;
                    // Add new data point with timestamp
                    const newPoint = { timestamp: now, ...eventData };
                    // Keep only last 5 minutes of data
                    const fiveMinutesAgo = now - 5 * 60 * 1000;
                    newHistory = [...prev.bandHistory.filter(h => h.timestamp >= fiveMinutesAgo), newPoint];
                  }
                  return {
                    ...prev,
                    bandPowers: eventData,
                    bandHistory: newHistory
                  };
                });
              } else if (eventName === 'metrics') {
                setEegData(prev => ({
                  ...prev,
                  metrics: eventData
                }));
              }
            } catch (e) {
              // Ignore parse errors
            }
          } else if (data === '2') {
            // Ping - respond with pong
            ws.send('3');
          }
        };

        ws.onerror = (error) => {
          console.error('EEG WebSocket error:', error);
        };

        ws.onclose = () => {
          console.log('EEG WebSocket disconnected');
          wsRef.current = null;
          // Attempt reconnect after delay
          if (eegStatus.streaming) {
            setTimeout(connectWebSocket, 3000);
          }
        };

        wsRef.current = ws;
      } catch (err) {
        console.error('Failed to connect EEG WebSocket:', err);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [eegStatus.streaming]);

  // Calculate day count from first data date
  const dayCount = firstDataDate
    ? Math.max(1, Math.floor((new Date() - firstDataDate) / (1000 * 60 * 60 * 24)) + 1)
    : Math.max(1, healthData.length);

  return (
    <div className="min-h-screen text-white" style={{
      background: '#000000'
    }}>
      {/* Subtle texture overlay */}
      <div className="fixed inset-0 opacity-30 pointer-events-none" style={{
        backgroundImage: `radial-gradient(circle at 50% 50%, rgba(212, 175, 55, 0.06) 0%, transparent 50%)`,
      }} />

      {/* Header */}
      <header className="relative border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs tracking-[0.3em] text-amber-400/70 uppercase mb-2">
                Dynamical Systems Study of Cognitive State Dynamics (N=1)
              </p>
              <h1 className="text-3xl font-light tracking-tight" style={{
                fontFamily: "'Annex Latin', system-ui, sans-serif"
              }}>
                Consciousness Observatory
              </h1>
              <p className="mt-2 text-sm text-white/60 max-w-xl">
                Mapping the high-dimensional dynamics of mind. Discovering attractors for generativity,
                clarity, and loving awareness through multimodal data.
              </p>
            </div>

            <div className="text-right">
              <div className="text-xs text-white/90 mb-1">Data Collection</div>
              <div className="text-2xl font-light tabular-nums" style={{ fontFamily: "'Annex Latin', system-ui, sans-serif" }}>
                Day {dayCount}
              </div>
              <div className="text-xs text-white/90 mt-1">
                {new Date().toLocaleDateString('en-US', {
                  weekday: 'long',
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {loading ? (
          <div className="text-center py-20 text-white/90">Loading data...</div>
        ) : (
          <div className="grid grid-cols-12 gap-6">

            {/* Left Column - Data Sources & Live Metrics */}
            <div className="col-span-3 space-y-6">

              {/* Data Sources Panel */}
              <div className="rounded-xl p-5" style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)'
              }}>
                <h2 className="text-xs tracking-widest text-white/90 uppercase mb-4">Data Streams</h2>
                <div className="space-y-3">
                  {dataSources.map(source => (
                    <div key={source.id} className="flex items-center justify-between py-2">
                      <div className="flex items-center gap-3">
                        <StatusOrb status={source.status} />
                        <div>
                          <div className="text-sm text-white">{source.name}</div>
                          <div className="text-xs text-white/90">
                            {formatTimeAgo(source.lastSync)}
                          </div>
                        </div>
                      </div>
                      {source.records > 0 && (
                        <div className="text-xs text-white/90 tabular-nums">
                          {source.records.toLocaleString()}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Live Metric */}
              <div className="rounded-xl p-5" style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)'
              }}>
                <h2 className="text-xs tracking-widest text-white/90 uppercase mb-4">Latest HRV</h2>
                <div className="text-center">
                  <div className="text-4xl font-light tabular-nums" style={{
                    fontFamily: "'Annex Latin', system-ui, sans-serif",
                    color: currentHRV > 45 ? '#10b981' : currentHRV > 35 ? '#f59e0b' : '#ef4444'
                  }}>
                    {currentHRV ? currentHRV.toFixed(1) : '--'}
                  </div>
                  <div className="text-xs text-white/90 mt-1">ms RMSSD</div>
                </div>
                <div className="mt-4 flex justify-center">
                  <Spark
                    data={trends?.hrv?.values || []}
                    color="#d4af37"
                    width={140}
                    height={50}
                  />
                </div>
              </div>

              {/* Target States */}
              <div className="rounded-xl p-5" style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)'
              }}>
                <h2 className="text-xs tracking-widest text-white/90 uppercase mb-4">Target Attractors</h2>
                <div className="space-y-3">
                  {targetStates.map(state => (
                    <div key={state.id} className="group cursor-pointer">
                      <div className="flex items-center gap-3 py-1">
                        <div
                          className="w-1.5 h-1.5 rounded-full"
                          style={{ backgroundColor: state.color }}
                        />
                        <span className="text-sm text-white/80 group-hover:text-white transition-colors">
                          {state.name}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Center Column - Data Overview */}
            <div className="col-span-6 space-y-6">

              <div className="flex items-center justify-between">
                <h2 className="text-lg font-light" style={{ fontFamily: "'Annex Latin', system-ui, sans-serif" }}>
                  Recent Health Data
                </h2>
              </div>

              {/* Health data table */}
              <div className="rounded-xl overflow-hidden" style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)'
              }}>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/5">
                      <th className="text-left text-xs text-white/90 font-normal px-5 py-3">Date</th>
                      <th className="text-right text-xs text-white/90 font-normal px-5 py-3">Sleep</th>
                      <th className="text-right text-xs text-white/90 font-normal px-5 py-3">HRV</th>
                      <th className="text-right text-xs text-white/90 font-normal px-5 py-3">Stress</th>
                      <th className="text-right text-xs text-white/90 font-normal px-5 py-3">
                        <div className="flex items-center justify-end gap-1">
                          <span>Cog Load</span>
                          <div className="relative group">
                            <div className="w-4 h-4 rounded-full border border-white/40 flex items-center justify-center cursor-help text-[10px] text-white/90 hover:border-white hover:text-white transition-colors">
                              ?
                            </div>
                            <div className="absolute right-0 top-6 w-64 p-3 rounded-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 text-left" style={{
                              background: 'rgba(30, 30, 30, 0.95)',
                              border: '1px solid rgba(255,255,255,0.1)',
                              boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
                            }}>
                              <div className="text-xs text-white font-medium mb-2">Cognitive Load Score (0-100)</div>
                              <div className="text-[11px] text-white/80 space-y-1">
                                <div>Meeting minutes / 10 <span className="text-white/90">(max 50)</span></div>
                                <div>Context switches × 5 <span className="text-white/90">(max 30)</span></div>
                                <div>Event count × 2 <span className="text-white/90">(max 20)</span></div>
                              </div>
                              <div className="text-[10px] text-white/90 mt-2 pt-2 border-t border-white/10">
                                Sourced from Google Calendar data
                              </div>
                            </div>
                          </div>
                        </div>
                      </th>
                      <th className="text-right text-xs text-white/90 font-normal px-5 py-3">Steps</th>
                      <th className="text-right text-xs text-white/90 font-normal px-5 py-3">RHR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {healthData.slice(-7).reverse().map((day, i) => {
                      const dateKey = day.timestamp.split('T')[0];
                      const cogLoad = cognitiveLoads[dateKey];
                      return (
                      <tr key={i} className="border-b border-white/5 last:border-0 hover:bg-white/5">
                        <td className="px-5 py-3 text-sm text-white">
                          {new Date(day.timestamp).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
                        </td>
                        <td className="px-5 py-3 text-sm text-right tabular-nums" style={{
                          color: day.sleep_score >= 80 ? '#10b981' : day.sleep_score >= 60 ? '#f59e0b' : '#ef4444'
                        }}>
                          {day.sleep_score || '--'}
                        </td>
                        <td className="px-5 py-3 text-sm text-right tabular-nums" style={{
                          color: day.hrv_rmssd > 45 ? '#10b981' : day.hrv_rmssd > 35 ? '#f59e0b' : '#ef4444'
                        }}>
                          {day.hrv_rmssd || '--'}
                        </td>
                        <td className="px-5 py-3 text-sm text-right tabular-nums text-white">
                          {day.stress_level || '--'}
                        </td>
                        <td className="px-5 py-3 text-sm text-right tabular-nums" style={{
                          color: cogLoad > 60 ? '#ef4444' : cogLoad > 40 ? '#f59e0b' : '#10b981'
                        }}>
                          {cogLoad != null ? cogLoad : '--'}
                        </td>
                        <td className="px-5 py-3 text-sm text-right tabular-nums text-white">
                          {day.steps?.toLocaleString() || '--'}
                        </td>
                        <td className="px-5 py-3 text-sm text-right tabular-nums text-white">
                          {day.resting_heart_rate || '--'}
                        </td>
                      </tr>
                    );})}
                  </tbody>
                </table>
              </div>

              {/* EEG Visualization Panel */}
              <div className="rounded-xl overflow-hidden" style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)'
              }}>
                {/* EEG Header */}
                <div className="px-5 py-4 border-b border-white/5 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <h3 className="text-sm font-medium text-white">Muse EEG</h3>
                    <div className="flex items-center gap-2">
                      <StatusOrb status={eegStatus.connected ? 'connected' : 'pending'} />
                      <span className={`text-xs capitalize ${isRecording ? 'text-red-400 font-medium' : 'text-white/90'}`}>
                        {isRecording ? 'RECORDING' :
                         eegStatus.state === 'streaming' ? 'Live' :
                         eegStatus.state === 'recording' ? 'Recording' :
                         eegStatus.state === 'connecting' ? 'Connecting...' :
                         eegStatus.state === 'unavailable' ? 'Unavailable' :
                         'Disconnected'}
                      </span>
                    </div>
                    {/* Recording indicator - prominent */}
                    {isRecording && (
                      <div className="flex items-center gap-2 ml-3 px-3 py-1.5 rounded-lg" style={{
                        background: 'rgba(239, 68, 68, 0.25)',
                        border: '1px solid rgba(239, 68, 68, 0.5)',
                        boxShadow: '0 0 12px rgba(239, 68, 68, 0.3)'
                      }}>
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                        <span className="text-sm text-red-400 font-medium font-mono tabular-nums">
                          REC {formatDuration(recordingDuration)}
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-4">
                    {/* Recording controls */}
                    {eegStatus.connected && (
                      <button
                        onClick={isRecording ? stopRecording : startRecording}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
                        style={{
                          background: isRecording ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                          border: `1px solid ${isRecording ? 'rgba(239, 68, 68, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
                          color: isRecording ? '#f87171' : '#34d399'
                        }}
                      >
                        {isRecording ? 'Stop Recording' : 'Start Recording'}
                      </button>
                    )}
                    {/* Signal quality */}
                    {eegStatus.connected && (
                      <div className="flex items-center gap-3">
                        <div className="text-xs text-white/90">Signal</div>
                        <div className="flex gap-1">
                          {EEG_CHANNELS.map((ch, i) => (
                            <div key={ch} className="flex flex-col items-center">
                              <SignalQualityDot quality={eegData.signalQuality[i] || 0} />
                              <span className="text-[10px] text-white/90 mt-0.5">{ch}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {eegStatus.connected ? (
                  <div className="p-5">
                    {/* Band Power Time Series - 5 minute history */}
                    <div className="mb-4">
                      <div className="text-xs text-white/90 mb-3">Frequency Bands (5 min history)</div>
                      <BandTimeSeries
                        history={eegData.bandHistory}
                        width={480}
                        height={160}
                      />
                    </div>

                    {/* Current Band Powers */}
                    <div className="border-t border-white/5 pt-4 mb-4">
                      <div className="text-xs text-white/90 mb-3">Current Levels</div>
                      <div className="space-y-2">
                        {Object.entries(eegData.bandPowers).map(([band, power]) => (
                          <BandPowerBar key={band} band={band} power={power} />
                        ))}
                      </div>
                    </div>

                    {/* State Metrics */}
                    <div className="border-t border-white/5 pt-4">
                      <div className="text-xs text-white/90 mb-3">Mental State Indicators</div>
                      <div className="grid grid-cols-4 gap-4">
                        {[
                          { label: 'Engagement', value: eegData.metrics.engagement, color: '#3b82f6' },
                          { label: 'Relaxation', value: eegData.metrics.relaxation, color: '#10b981' },
                          { label: 'Focus', value: eegData.metrics.focus, color: '#f59e0b' },
                          { label: 'Meditation', value: eegData.metrics.meditation, color: '#8b5cf6' }
                        ].map(metric => (
                          <div key={metric.label} className="text-center">
                            <div className="text-xl font-light tabular-nums" style={{
                              fontFamily: "'Annex Latin', system-ui, sans-serif",
                              color: metric.color
                            }}>
                              {metric.value.toFixed(1)}
                            </div>
                            <div className="text-[10px] text-white/90 mt-1">{metric.label}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Disconnected State */
                  <div className="p-8 text-center">
                    <div className="text-white/90 text-sm">
                      <div className="mb-3">
                        <svg className="w-8 h-8 mx-auto text-white/90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                            d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                      </div>
                      <div className="font-medium text-white/80 mb-1">Muse EEG Not Connected</div>
                      <div className="text-xs text-white/90 max-w-xs mx-auto">
                        Start OpenMuse stream with your Muse S device to enable real-time brainwave visualization
                      </div>
                      <div className="mt-4 text-xs text-white/90 font-mono">
                        <code className="px-2 py-1 bg-white/5 rounded">OpenMuse stream [address]</code>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Quick Stats */}
            <div className="col-span-3 space-y-6">

              {/* Weekly Overview */}
              <div className="rounded-xl p-5" style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)'
              }}>
                <h2 className="text-xs tracking-widest text-white/90 uppercase mb-4">7-Day Trends</h2>

                {trends && [
                  { label: 'Avg HRV', value: trends.hrv.avg, unit: 'ms', trend: trends.hrv.trend, data: trends.hrv.values, color: '#8b5cf6' },
                  { label: 'Avg Stress', value: trends.stress.avg, unit: '', trend: trends.stress.trend, data: trends.stress.values, color: '#f59e0b' },
                  { label: 'Sleep Score', value: trends.sleep.avg, unit: '', trend: trends.sleep.trend, data: trends.sleep.values, color: '#10b981' },
                ].map(metric => (
                  <div key={metric.label} className="py-3 border-b border-white/5 last:border-0">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-white/90">{metric.label}</span>
                      {metric.trend !== 0 && (
                        <span className={`text-xs ${metric.trend > 0 ? 'text-green-400' : 'text-amber-400'}`}>
                          {metric.trend > 0 ? '+' : ''}{metric.trend}%
                        </span>
                      )}
                    </div>
                    <div className="flex items-end justify-between">
                      <div className="text-xl font-light tabular-nums" style={{ fontFamily: "'Annex Latin', system-ui, sans-serif" }}>
                        {metric.value ?? '--'}
                        <span className="text-xs text-white/90 ml-1">{metric.unit}</span>
                      </div>
                      <Spark data={metric.data} color={metric.color} width={60} height={24} />
                    </div>
                  </div>
                ))}
              </div>

              {/* Today's Calendar Context */}
              <div className="rounded-xl p-5" style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)'
              }}>
                <h2 className="text-xs tracking-widest text-white/90 uppercase mb-4">Today's Context</h2>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-white/90">Meetings</span>
                    <span className="text-sm text-white">{todayContext?.meetings ?? '--'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-white/90">Meeting Hours</span>
                    <span className="text-sm text-white">{todayContext?.meetingHours ?? '--'}h</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-white/90">Context Switches</span>
                    <span className={`text-sm ${todayContext?.contextSwitches > 4 ? 'text-amber-400' : 'text-white'}`}>
                      {todayContext?.contextSwitches ?? '--'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Quick State Log */}
              <div className="rounded-xl p-5" style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)'
              }}>
                <h2 className="text-xs tracking-widest text-white/90 uppercase mb-4">Quick State Log</h2>
                <div className="grid grid-cols-5 gap-2">
                  {targetStates.map(state => (
                    <button
                      key={state.id}
                      className="aspect-square rounded-lg flex items-center justify-center transition-all hover:scale-110"
                      style={{
                        background: `${state.color}15`,
                        border: `1px solid ${state.color}30`
                      }}
                      title={state.name}
                    >
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: state.color }}
                      />
                    </button>
                  ))}
                </div>
                <p className="text-xs text-white/90 mt-3 text-center">
                  Tap to log current state
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <p className="text-xs text-white/90">
            Consciousness Observatory • Multimodal Research Platform
          </p>
          <div className="flex gap-6">
            <button className="text-xs text-white/90 hover:text-white transition-colors">
              Sync Data
            </button>
            <button className="text-xs text-white/90 hover:text-white transition-colors">
              Export
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
