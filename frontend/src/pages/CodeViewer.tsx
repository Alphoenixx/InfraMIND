import { useState, useEffect } from 'react';
import { Shield, Coins, Zap, CheckCircle, AlertTriangle } from 'lucide-react';

export default function CodeViewer() {
  const [candidates, setCandidates] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    const data = localStorage.getItem('inframind_candidates');
    if (data) {
        try {
            setCandidates(JSON.parse(data));
        } catch(e) {}
    }
  }, []);

  if (candidates.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-400">
        No generated code found. Go to Prompt Dashboard to generate something.
      </div>
    );
  }

  const activeCandidate = candidates[activeTab];
  const scores = activeCandidate.scores;

  return (
    <div className="h-full flex flex-col">
      <div className="flex gap-4 mb-6">
        {candidates.map((cand, idx) => (
          <button
            key={cand.id}
            onClick={() => setActiveTab(idx)}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === idx
                ? 'bg-accent-purple/20 border border-accent-purple text-white shadow-[0_0_15px_rgba(139,92,246,0.3)]'
                : 'glass-panel text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            {cand.provider || `Candidate ${idx + 1}`}
            {activeTab === idx && <span className="ml-2 text-accent-green">★</span>}
          </button>
        ))}
        <div className="ml-auto flex items-center">
            <button className="glass-panel px-4 py-2 text-sm text-white hover:bg-white/10">Download .tf</button>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-6 h-full">
        {/* Code Editor Area */}
        <div className="col-span-3 glass-panel overflow-hidden flex flex-col">
          <div className="bg-bg-secondary px-4 py-2 border-b border-gray-800 flex justify-between items-center text-sm text-gray-400 font-mono">
            <span>main.tf</span>
            <span>{activeCandidate.language}</span>
          </div>
          <div className="p-4 overflow-y-auto flex-1 bg-[#0d0d1a]">
            {/* Simple pre formatting, normally use highlight.js or prism */}
            <pre className="text-[#a5b4fc] font-mono text-sm leading-relaxed">
              <code>{activeCandidate.code}</code>
            </pre>
          </div>
        </div>

        {/* Mini Evaluation Panel */}
        <div className="col-span-1 flex flex-col gap-4">
          <div className="glass-panel p-5">
            <h3 className="text-white font-medium mb-4 flex items-center gap-2">
              <Zap size={18} className="text-accent-gold" />
              Composite Score
            </h3>
            <div className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-accent-green to-accent-cyan">
              {scores.composite_score.toFixed(1)}
            </div>
            <div className="mt-2 text-sm text-gray-400">Out of 100</div>
          </div>

          <div className="glass-panel p-5">
            <h3 className="text-white font-medium mb-4 flex items-center gap-2">
              <CheckCircle size={18} className="text-accent-blue" />
              Syntax & Execution
            </h3>
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-400 text-sm">Valid Syntax</span>
              <span className="text-accent-green">{scores.syntax_score}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Deployment Test</span>
              {scores.execution_result === 'success' ? (
                  <span className="text-accent-green bg-accent-green/20 px-2 py-0.5 rounded text-xs">Success</span>
              ) : (
                  <span className="text-accent-red bg-accent-red/20 px-2 py-0.5 rounded text-xs flex items-center gap-1"><AlertTriangle size={12}/> Failed</span>
              )}
            </div>
          </div>

          <div className="glass-panel p-5">
            <h3 className="text-white font-medium mb-4 flex items-center gap-2">
              <Coins size={18} className="text-accent-orange" />
              Estimated Cost
            </h3>
            <div className="text-2xl font-bold text-white">
              ${scores.cost_estimate_usd.toFixed(2)}
            </div>
            <div className="mt-1 text-sm text-gray-400">per month (projected)</div>
          </div>

          <div className="glass-panel p-5 flex-1">
            <h3 className="text-white font-medium mb-4 flex items-center gap-2">
              <Shield size={18} className="text-accent-purple" />
              Security Score
            </h3>
            <div className="flex items-end gap-2">
                <div className="text-3xl font-bold text-white">{scores.security_score}</div>
                <div className="text-sm text-gray-400 mb-1">/ 100</div>
            </div>
            
            <div className="mt-4 space-y-2">
                {scores.security_score < 100 && (
                    <div className="text-xs text-accent-red flex items-start gap-1">
                        <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                        Weak configs detected in generation
                    </div>
                )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
