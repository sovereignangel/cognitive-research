import React, { useState, useEffect } from 'react';

// Mock data - in production this would come from your API
const mockHealthData = [
  { date: '2026-01-15', hrv: 42, stress: 28, sleepScore: 82, bodyBattery: 75 },
  { date: '2026-01-16', hrv: 38, stress: 35, sleepScore: 71, bodyBattery: 62 },
  { date: '2026-01-17', hrv: 45, stress: 22, sleepScore: 88, bodyBattery: 84 },
  { date: '2026-01-18', hrv: 51, stress: 18, sleepScore: 91, bodyBattery: 89 },
  { date: '2026-01-19', hrv: 48, stress: 25, sleepScore: 85, bodyBattery: 78 },
  { date: '2026-01-20', hrv: 44, stress: 31, sleepScore: 79, bodyBattery: 71 },
  { date: '2026-01-21', hrv: 47, stress: 24, sleepScore: 86, bodyBattery: 82 },
];

const targetStates = [
  { id: 'generativity', name: 'Generativity', description: 'Creative flow & productive emergence', color: '#f59e0b' },
  { id: 'confidence', name: 'Confidence', description: 'Grounded self-assurance', color: '#10b981' },
  { id: 'enlightenment', name: 'Enlightenment', description: 'Clarity & insight', color: '#8b5cf6' },
  { id: 'peak_performance', name: 'Peak Performance', description: 'Optimal cognitive function', color: '#3b82f6' },
  { id: 'loving_awareness', name: 'Loving Awareness', description: 'Open-hearted presence', color: '#ec4899' },
];

const dataSources = [
  { id: 'garmin', name: 'Garmin Health', status: 'connected', lastSync: '2 hours ago', records: 847 },
  { id: 'calendar', name: 'Google Calendar', status: 'connected', lastSync: '1 hour ago', records: 234 },
  { id: 'eeg', name: 'Muse EEG', status: 'pending', lastSync: null, records: 0 },
  { id: 'journal', name: 'Journal Entries', status: 'pending', lastSync: null, records: 0 },
];

// Spark visualization component
const Spark = ({ data, color, width = 120, height = 40 }) => {
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
      <defs>
        <linearGradient id={`grad-${color}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
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

// Radial status indicator
const StatusOrb = ({ status }) => {
  const colors = {
    connected: { bg: '#10b981', glow: 'rgba(16, 185, 129, 0.4)' },
    pending: { bg: '#6b7280', glow: 'rgba(107, 114, 128, 0.2)' },
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

// Main Dashboard Component
export default function ConsciousnessObservatory() {
  const [activeInsight, setActiveInsight] = useState(null);
  const [insights, setInsights] = useState([
    {
      id: 1,
      date: '2026-01-20',
      title: 'HRV-Creativity Correlation',
      content: 'Noticed that mornings with HRV > 45 consistently correlate with higher generativity ratings. The body battery threshold seems to be around 75 for peak creative output.',
      tags: ['hrv', 'generativity', 'pattern'],
      type: 'observation'
    },
    {
      id: 2,
      date: '2026-01-18',
      title: 'Calendar Density Effect',
      content: 'Days with >4 context switches show a 23% drop in self-reported clarity. Blocking 2+ hour focus periods appears protective.',
      tags: ['calendar', 'focus', 'pattern'],
      type: 'finding'
    },
  ]);
  const [newInsight, setNewInsight] = useState({ title: '', content: '', tags: '' });
  const [showInsightForm, setShowInsightForm] = useState(false);

  // Simulate real-time metric
  const [currentHRV, setCurrentHRV] = useState(47);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentHRV(prev => prev + (Math.random() - 0.5) * 2);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const addInsight = () => {
    if (newInsight.title && newInsight.content) {
      setInsights([
        {
          id: Date.now(),
          date: new Date().toISOString().split('T')[0],
          title: newInsight.title,
          content: newInsight.content,
          tags: newInsight.tags.split(',').map(t => t.trim()).filter(Boolean),
          type: 'observation'
        },
        ...insights
      ]);
      setNewInsight({ title: '', content: '', tags: '' });
      setShowInsightForm(false);
    }
  };

  return (
    <div className="min-h-screen text-gray-100" style={{
      background: 'linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%)'
    }}>
      {/* Subtle texture overlay */}
      <div className="fixed inset-0 opacity-30 pointer-events-none" style={{
        backgroundImage: `radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%)`,
      }} />
      
      {/* Header */}
      <header className="relative border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs tracking-[0.3em] text-indigo-400/70 uppercase mb-2">
                Personal Research Project
              </p>
              <h1 className="text-3xl font-light tracking-tight" style={{
                fontFamily: 'Georgia, Cambria, serif'
              }}>
                Consciousness Observatory
              </h1>
              <p className="mt-2 text-sm text-gray-500 max-w-xl">
                Mapping the high-dimensional dynamics of mind. Discovering attractors for generativity, 
                clarity, and loving awareness through multimodal data.
              </p>
            </div>
            
            <div className="text-right">
              <div className="text-xs text-gray-500 mb-1">Current Session</div>
              <div className="text-2xl font-light tabular-nums" style={{ fontFamily: 'Georgia, serif' }}>
                Day 47
              </div>
              <div className="text-xs text-gray-600 mt-1">
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
        <div className="grid grid-cols-12 gap-6">
          
          {/* Left Column - Data Sources & Live Metrics */}
          <div className="col-span-3 space-y-6">
            
            {/* Data Sources Panel */}
            <div className="rounded-xl p-5" style={{
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)'
            }}>
              <h2 className="text-xs tracking-widest text-gray-500 uppercase mb-4">Data Streams</h2>
              <div className="space-y-3">
                {dataSources.map(source => (
                  <div key={source.id} className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-3">
                      <StatusOrb status={source.status} />
                      <div>
                        <div className="text-sm text-gray-300">{source.name}</div>
                        <div className="text-xs text-gray-600">
                          {source.lastSync || 'Not connected'}
                        </div>
                      </div>
                    </div>
                    {source.records > 0 && (
                      <div className="text-xs text-gray-500 tabular-nums">
                        {source.records.toLocaleString()}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              
              <button className="w-full mt-4 py-2 text-xs tracking-wide text-indigo-400 border border-indigo-400/20 rounded-lg hover:bg-indigo-400/10 transition-colors">
                + Add Source
              </button>
            </div>

            {/* Live Metric */}
            <div className="rounded-xl p-5" style={{
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)'
            }}>
              <h2 className="text-xs tracking-widest text-gray-500 uppercase mb-4">Live HRV</h2>
              <div className="text-center">
                <div className="text-4xl font-light tabular-nums" style={{ 
                  fontFamily: 'Georgia, serif',
                  color: currentHRV > 45 ? '#10b981' : currentHRV > 35 ? '#f59e0b' : '#ef4444'
                }}>
                  {currentHRV.toFixed(1)}
                </div>
                <div className="text-xs text-gray-600 mt-1">ms RMSSD</div>
              </div>
              <div className="mt-4 flex justify-center">
                <Spark 
                  data={mockHealthData.map(d => d.hrv)} 
                  color="#8b5cf6" 
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
              <h2 className="text-xs tracking-widest text-gray-500 uppercase mb-4">Target Attractors</h2>
              <div className="space-y-3">
                {targetStates.map(state => (
                  <div key={state.id} className="group cursor-pointer">
                    <div className="flex items-center gap-3 py-1">
                      <div 
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ backgroundColor: state.color }}
                      />
                      <span className="text-sm text-gray-400 group-hover:text-gray-200 transition-colors">
                        {state.name}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Center Column - Insights & Learnings */}
          <div className="col-span-6 space-y-6">
            
            {/* Insights Header */}
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-light" style={{ fontFamily: 'Georgia, serif' }}>
                Research Insights
              </h2>
              <button 
                onClick={() => setShowInsightForm(!showInsightForm)}
                className="text-xs tracking-wide text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                + New Insight
              </button>
            </div>

            {/* New Insight Form */}
            {showInsightForm && (
              <div className="rounded-xl p-5" style={{
                background: 'rgba(99, 102, 241, 0.05)',
                border: '1px solid rgba(99, 102, 241, 0.2)'
              }}>
                <input
                  type="text"
                  placeholder="Insight title..."
                  value={newInsight.title}
                  onChange={e => setNewInsight({...newInsight, title: e.target.value})}
                  className="w-full bg-transparent border-b border-white/10 pb-2 mb-4 text-gray-200 placeholder-gray-600 focus:outline-none focus:border-indigo-400/50"
                />
                <textarea
                  placeholder="What did you discover? What patterns are emerging?"
                  value={newInsight.content}
                  onChange={e => setNewInsight({...newInsight, content: e.target.value})}
                  rows={4}
                  className="w-full bg-transparent border border-white/5 rounded-lg p-3 text-sm text-gray-300 placeholder-gray-600 focus:outline-none focus:border-indigo-400/30 resize-none"
                />
                <div className="flex items-center justify-between mt-4">
                  <input
                    type="text"
                    placeholder="Tags (comma separated)"
                    value={newInsight.tags}
                    onChange={e => setNewInsight({...newInsight, tags: e.target.value})}
                    className="bg-transparent border-b border-white/10 pb-1 text-xs text-gray-400 placeholder-gray-600 focus:outline-none focus:border-indigo-400/50 w-48"
                  />
                  <div className="flex gap-2">
                    <button 
                      onClick={() => setShowInsightForm(false)}
                      className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-300"
                    >
                      Cancel
                    </button>
                    <button 
                      onClick={addInsight}
                      className="px-4 py-1.5 text-xs bg-indigo-500/20 text-indigo-300 rounded-lg hover:bg-indigo-500/30 transition-colors"
                    >
                      Save Insight
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Insights List */}
            <div className="space-y-4">
              {insights.map(insight => (
                <div 
                  key={insight.id}
                  className="rounded-xl p-5 cursor-pointer transition-all duration-300 hover:scale-[1.01]"
                  style={{
                    background: activeInsight === insight.id 
                      ? 'rgba(99, 102, 241, 0.08)' 
                      : 'rgba(255,255,255,0.02)',
                    border: activeInsight === insight.id
                      ? '1px solid rgba(99, 102, 241, 0.3)'
                      : '1px solid rgba(255,255,255,0.05)'
                  }}
                  onClick={() => setActiveInsight(activeInsight === insight.id ? null : insight.id)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-base text-gray-200" style={{ fontFamily: 'Georgia, serif' }}>
                      {insight.title}
                    </h3>
                    <span className="text-xs text-gray-600">{insight.date}</span>
                  </div>
                  <p className="text-sm text-gray-400 leading-relaxed">
                    {insight.content}
                  </p>
                  {insight.tags.length > 0 && (
                    <div className="flex gap-2 mt-4">
                      {insight.tags.map(tag => (
                        <span 
                          key={tag}
                          className="px-2 py-0.5 text-xs rounded-full bg-white/5 text-gray-500"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Placeholder for future visualizations */}
            <div className="rounded-xl p-8 text-center" style={{
              background: 'rgba(255,255,255,0.01)',
              border: '1px dashed rgba(255,255,255,0.1)'
            }}>
              <div className="text-gray-600 text-sm">
                <div className="mb-2">◇</div>
                Manifold visualizations will appear here as data accumulates
              </div>
            </div>
          </div>

          {/* Right Column - Quick Stats & State Tracker */}
          <div className="col-span-3 space-y-6">
            
            {/* Weekly Overview */}
            <div className="rounded-xl p-5" style={{
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)'
            }}>
              <h2 className="text-xs tracking-widest text-gray-500 uppercase mb-4">7-Day Trends</h2>
              
              {[
                { label: 'Avg HRV', value: '45.3', unit: 'ms', trend: '+8%', data: mockHealthData.map(d => d.hrv), color: '#8b5cf6' },
                { label: 'Avg Stress', value: '26', unit: '', trend: '-12%', data: mockHealthData.map(d => d.stress), color: '#f59e0b' },
                { label: 'Sleep Score', value: '83', unit: '', trend: '+5%', data: mockHealthData.map(d => d.sleepScore), color: '#10b981' },
              ].map(metric => (
                <div key={metric.label} className="py-3 border-b border-white/5 last:border-0">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-500">{metric.label}</span>
                    <span className={`text-xs ${metric.trend.startsWith('+') ? 'text-green-400' : 'text-amber-400'}`}>
                      {metric.trend}
                    </span>
                  </div>
                  <div className="flex items-end justify-between">
                    <div className="text-xl font-light tabular-nums" style={{ fontFamily: 'Georgia, serif' }}>
                      {metric.value}
                      <span className="text-xs text-gray-600 ml-1">{metric.unit}</span>
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
              <h2 className="text-xs tracking-widest text-gray-500 uppercase mb-4">Today's Context</h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Meetings</span>
                  <span className="text-sm text-gray-300">4</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Meeting Hours</span>
                  <span className="text-sm text-gray-300">3.5h</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Context Switches</span>
                  <span className="text-sm text-amber-400">6</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Focus Blocks</span>
                  <span className="text-sm text-green-400">2</span>
                </div>
              </div>
            </div>

            {/* Quick State Log */}
            <div className="rounded-xl p-5" style={{
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)'
            }}>
              <h2 className="text-xs tracking-widest text-gray-500 uppercase mb-4">Quick State Log</h2>
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
              <p className="text-xs text-gray-600 mt-3 text-center">
                Tap to log current state
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <p className="text-xs text-gray-600">
            Consciousness Observatory • Multimodal Research Platform
          </p>
          <div className="flex gap-6">
            <button className="text-xs text-gray-600 hover:text-gray-400 transition-colors">
              Export Data
            </button>
            <button className="text-xs text-gray-600 hover:text-gray-400 transition-colors">
              Settings
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
