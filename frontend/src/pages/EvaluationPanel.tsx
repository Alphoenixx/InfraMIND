import { useState, useEffect } from 'react';
import { Shield, Coins, Zap, CheckCircle, AlertTriangle, Cpu, TrendingUp, Settings2, BarChart2, Check, X, Code2, Play } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';
import { useNavigate } from 'react-router-dom';
import { API_BASE_URL, WEIGHT_SLIDERS, DEFAULT_WEIGHTS, computeComposite } from '../constants';

export default function EvaluationPanel() {
  const [candidates, setCandidates] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState(0);
  const [baseWeights, setBaseWeights] = useState(DEFAULT_WEIGHTS);
  const [userWeights, setUserWeights] = useState(DEFAULT_WEIGHTS);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simResults, setSimResults] = useState<any>(null);
  const [lastWeightChange, setLastWeightChange] = useState<{key: string, delta: number, compositeDelta: number} | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const data = localStorage.getItem('inframind_candidates');
    if (data) {
        try {
            const parsed = JSON.parse(data);
            setCandidates(parsed);
            
            // Auto-select best candidate
            if (parsed.length > 0) {
                let bestIdx = 0;
                let maxScore = -1;
                parsed.forEach((c: any, i: number) => {
                    if (c.scores.composite_score > maxScore) {
                        maxScore = c.scores.composite_score;
                        bestIdx = i;
                    }
                });
                setActiveTab(bestIdx);
            }
        } catch(e) {}
    }
  }, []);

  if (candidates.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[80vh] text-gray-400 gap-4">
        <BarChart2 size={48} className="text-gray-600 mb-2" />
        <h2 className="text-xl text-white">No Evaluation Data</h2>
        <p>Go to Prompt Dashboard to generate infrastructure candidates first.</p>
        <button onClick={() => navigate('/')} className="mt-4 px-6 py-2 bg-accent-purple/20 text-accent-purple border border-accent-purple rounded-lg hover:bg-accent-purple/30">
            Go to Prompt Dashboard
        </button>
      </div>
    );
  }

  const activeCand = candidates[activeTab];
  const scores = activeCand.scores;
  const findings = scores.findings || [];

  // Recalculate composite based on user weights
  const uComp = computeComposite(scores, userWeights);
  const bComp = computeComposite(scores, baseWeights);
  const displayComp = Math.round(uComp * 10) / 10;

  // Prepare Radar Data for ALL candidates
  const radarData = [
      { subject: 'Syntax', A: candidates[0]?.scores.syntax_score, B: candidates[1]?.scores.syntax_score, C: candidates[2]?.scores.syntax_score },
      { subject: 'Execution', A: candidates[0]?.scores.execution_result === 'success' ? 100 : 0, B: candidates[1]?.scores.execution_result === 'success' ? 100 : 0, C: candidates[2]?.scores.execution_result === 'success' ? 100 : 0 },
      { subject: 'Cost', A: Math.max(0, 100 - candidates[0]?.scores.cost_estimate_usd), B: Math.max(0, 100 - candidates[1]?.scores.cost_estimate_usd), C: Math.max(0, 100 - candidates[2]?.scores.cost_estimate_usd) },
      { subject: 'Security', A: candidates[0]?.scores.security_score, B: candidates[1]?.scores.security_score, C: candidates[2]?.scores.security_score },
      { subject: 'Correctness', A: candidates[0]?.scores.correctness_score, B: candidates[1]?.scores.correctness_score, C: candidates[2]?.scores.correctness_score },
  ];

  const handleWeightChange = (key: string, val: number) => {
      const oldWeight = (userWeights as any)[key];
      const newUserWeights = { ...userWeights, [key]: val };
      setUserWeights(newUserWeights);
      
      const newUComp = computeComposite(scores, newUserWeights);

      setLastWeightChange({
          key,
          delta: val - oldWeight,
          compositeDelta: newUComp - uComp
      });
  };

  const getReadinessColor = () => {
      if (scores.deploy_readiness === 'green') return 'bg-accent-green/20 border-accent-green text-accent-green';
      if (scores.deploy_readiness === 'yellow') return 'bg-accent-gold/20 border-accent-gold text-accent-gold';
      return 'bg-accent-red/20 border-accent-red text-accent-red';
  };

  const getGaugeColor = () => {
      if (displayComp >= 90) return '#10b981'; // green
      if (displayComp >= 75) return '#f59e0b'; // gold
      return '#ef4444'; // red
  };
  
  const gaugeCircumference = 2 * Math.PI * 54; // r=54
  const gaugeOffset = gaugeCircumference - (displayComp / 100) * gaugeCircumference;

  return (
    <div className="h-full flex flex-col pb-8 overflow-y-auto pr-4">
      
      {/* Header Tabs */}
      <div className="flex gap-4 mb-2">
        {candidates.map((cand, idx) => {
            const isBest = cand.scores.composite_score === Math.max(...candidates.map(c => c.scores.composite_score));
            return (
                <button
                    key={cand.id}
                    onClick={() => { setActiveTab(idx); setLastWeightChange(null); }}
                    className={`px-6 py-3 rounded-lg font-medium transition-all flex items-center gap-2 ${
                    activeTab === idx
                        ? 'bg-accent-purple/20 border border-accent-purple text-white shadow-[0_0_15px_rgba(139,92,246,0.3)]'
                        : 'glass-panel text-gray-400 hover:text-white hover:bg-white/5'
                    }`}
                >
                    Candidate {idx + 1}
                    {isBest && <span className="flex items-center gap-1 text-xs bg-accent-gold/20 text-accent-gold px-2 py-0.5 rounded-full border border-accent-gold/50 ml-2 shadow-[0_0_8px_rgba(252,211,77,0.4)]">★ Best</span>}
                </button>
            )
        })}
      </div>
      <div className="text-gray-400 text-sm mb-6 px-1">
          {activeCand.scores.composite_score === Math.max(...candidates.map(c => c.scores.composite_score)) 
            ? <><span className="text-accent-gold">Why Best?</span> Highest composite ({activeCand.scores.composite_score.toFixed(1)}) — balanced across all axes.</>
            : "Alternative candidate topology."}
      </div>

      {/* Deploy Readiness Banner */}
      <div className={`border rounded-lg p-3 mb-6 flex justify-between items-center ${getReadinessColor()}`}>
          <div className="flex items-center gap-3 font-semibold text-lg tracking-wide shrink-0">
             {scores.deploy_readiness === 'green' && <CheckCircle className="animate-pulse" />}
             {scores.deploy_readiness === 'yellow' && <AlertTriangle />}
             {scores.deploy_readiness === 'red' && <X />}
             {scores.deploy_readiness === 'green' ? 'Ready to Deploy' : (scores.deploy_readiness === 'yellow' ? 'Risky Configuration' : 'Not Safe for Production')}
          </div>
          <div className="flex gap-6 text-sm opacity-90 mx-auto justify-center w-full pr-12">
              <span className="flex items-center gap-2"><Shield size={16} /> Security: {scores.security_score}/100</span>
              <span className="flex items-center gap-2"><Coins size={16} /> Cost: ${scores.cost_estimate_usd.toFixed(2)}/mo</span>
              <span className="flex items-center gap-2"><AlertTriangle size={16} /> Findings: {findings.length}</span>
          </div>
      </div>

      {/* Row 1: Gauge & Radar */}
      <div className="grid grid-cols-3 gap-6 mb-6 h-72">
          {/* Composite Gauge */}
          <div className="col-span-1 glass-panel flex flex-col items-center justify-center p-6 relative">
              <h3 className="absolute top-4 left-4 text-xs font-medium text-gray-400 tracking-widest uppercase">Composite</h3>
              
              <div className="relative w-40 h-40 flex items-center justify-center">
                  <svg className="transform -rotate-90 w-full h-full">
                      <circle cx="80" cy="80" r="54" stroke="rgba(255,255,255,0.05)" strokeWidth="12" fill="none" />
                      <circle cx="80" cy="80" r="54" stroke={getGaugeColor()} strokeWidth="12" fill="none" 
                              strokeDasharray={gaugeCircumference} strokeDashoffset={gaugeOffset} 
                              className="transition-all duration-500 ease-out" />
                  </svg>
                  <div className="absolute flex flex-col items-center justify-center">
                      <span className="text-4xl font-bold text-white">{displayComp}</span>
                      <span className="text-xs text-gray-500">/ 100</span>
                  </div>
              </div>

              <div className="mt-4 text-center">
                  <div className={`text-sm flex items-center justify-center gap-1 font-medium ${scores.baseline_delta >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      {scores.baseline_delta >= 0 ? <TrendingUp size={14}/> : <TrendingUp size={14} className="transform rotate-180"/>}
                      {scores.baseline_delta >= 0 ? '+' : ''}{scores.baseline_delta} vs baseline
                  </div>
                  <div className="text-xs text-gray-500 mt-1 bg-black/40 px-2 py-1 rounded-full inline-block border border-gray-800">
                      Confidence: <span className={scores.confidence === 'high' ? 'text-accent-green' : 'text-accent-gold'}>{scores.confidence.toUpperCase()}</span>
                  </div>
              </div>
          </div>

          {/* Radar Chart */}
          <div className="col-span-2 glass-panel p-4 relative flex flex-col">
              <h3 className="absolute top-4 left-4 text-xs font-medium text-gray-400 tracking-widest uppercase z-10">Axis Comparison</h3>
              <div className="absolute top-4 right-4 flex gap-3 text-xs z-10">
                  <span className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-[#8b5cf6]"></div> Cand 1</span>
                  <span className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-[#06b6d4]"></div> Cand 2</span>
                  <span className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-[#f97316]"></div> Cand 3</span>
              </div>
              <div className="flex-1 -mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                      <RadarChart cx="50%" cy="55%" outerRadius="75%" data={radarData}>
                          <PolarGrid stroke="#333" />
                          <PolarAngleAxis dataKey="subject" tick={{ fill: '#888', fontSize: 12 }} />
                          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                          <RechartsTooltip contentStyle={{backgroundColor: '#0a0a1f', borderColor: '#333'}} />
                          <Radar name="Cand 1" dataKey="A" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={activeTab===0 ? 0.3 : 0.05} strokeWidth={activeTab===0 ? 3 : 1} />
                          {candidates.length > 1 && <Radar name="Cand 2" dataKey="B" stroke="#06b6d4" fill="#06b6d4" fillOpacity={activeTab===1 ? 0.3 : 0.05} strokeWidth={activeTab===1 ? 3 : 1} />}
                          {candidates.length > 2 && <Radar name="Cand 3" dataKey="C" stroke="#f97316" fill="#f97316" fillOpacity={activeTab===2 ? 0.3 : 0.05} strokeWidth={activeTab===2 ? 3 : 1} />}
                      </RadarChart>
                  </ResponsiveContainer>
              </div>
          </div>
      </div>

      {/* Row 2: Score Cards */}
      <div className="grid grid-cols-5 gap-4 mb-6">
          <ScoreCard title="Syntax" icon={<Code2/>} val={`${scores.syntax_score}%`} explain="All HCL blocks formatted" fix={scores.syntax_score < 100 ? "Fix syntax errors in terraform" : null} max={100} current={scores.syntax_score} color="bg-accent-green" />
          <ScoreCard title="Execution" icon={<Cpu/>} val={scores.execution_result} explain="tf plan would succeed" fix={scores.execution_result !== 'success' ? "Fix broken references" : null} max={1} current={scores.execution_result==='success'?1:0} color="bg-accent-blue" />
          <ScoreCard title="Cost" icon={<Coins/>} val={`$${scores.cost_estimate_usd}`} explain="Monthly proj." fix={scores.cost_estimate_usd > 10 ? "Reduce replica count to save cost" : null} max={50} current={50 - scores.cost_estimate_usd} color="bg-accent-orange" />
          <ScoreCard title="Security" icon={<Shield/>} val={`${scores.security_score}`} explain="Rules & Encryption" fix={scores.security_score < 100 ? "Open ports / No enc flag" : null} max={100} current={scores.security_score} color="bg-accent-purple" />
          <ScoreCard title="Corrects" icon={<CheckCircle/>} val={`${scores.correctness_score}`} explain="VPC/Subnet links" fix={scores.correctness_score < 100 ? "Link floating subnets to VPC" : null} max={100} current={scores.correctness_score} color="bg-accent-cyan" />
      </div>

      {/* Row 3: What-if Analysis & Findings Log */}
      <div className="grid grid-cols-3 gap-6">
          
          {/* Weight Sliders */}
          <div className="col-span-1 glass-panel p-5 flex flex-col">
              <h3 className="text-white font-medium mb-1 flex items-center gap-2"><Settings2 size={18} /> What-If Analysis</h3>
              <p className="text-xs text-gray-500 mb-6 border-b border-gray-800 pb-3">Adjust weights to see dynamic composite recalculations.</p>
              
              <div className="flex flex-col gap-5 flex-1 relative">
                  {WEIGHT_SLIDERS.map(w => (
                      <div key={w.key}>
                          <div className="flex justify-between text-xs mb-1">
                              <span className="text-gray-300">{w.label}</span>
                              <span className="text-white font-mono">{(userWeights as any)[w.key].toFixed(2)}</span>
                          </div>
                          <input 
                              type="range" min="0" max="1" step="0.05"
                              value={(userWeights as any)[w.key]}
                              onChange={(e) => handleWeightChange(w.key, parseFloat(e.target.value))}
                              className={`w-full h-1 bg-${w.color} rounded-lg appearance-none cursor-pointer`}
                          />
                      </div>
                  ))}

                  {/* Why Score Changed Inline Panel */}
                  {lastWeightChange && Math.abs(lastWeightChange.compositeDelta) > 0.05 && (
                      <div className="absolute -right-4 -bottom-4 translate-x-full animate-in slide-in-from-left-4 fade-in z-20">
                          <div className="bg-[#111122] border border-gray-700 shadow-2xl rounded-lg p-3 w-56">
                              <div className="text-xs font-bold text-accent-gold mb-1 uppercase tracking-wider">Why Score Changed</div>
                              <div className="text-sm text-gray-300">
                                  {lastWeightChange.delta > 0 ? '+' : ''}{(lastWeightChange.delta * 10).toFixed(1)} {lastWeightChange.key} weight
                              </div>
                              <div className={`text-sm font-semibold mt-1 flex items-center gap-1 ${lastWeightChange.compositeDelta > 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                                  {lastWeightChange.compositeDelta > 0 ? <TrendingUp size={16}/> : <TrendingUp size={16} className="transform rotate-180"/>}
                                  Score {lastWeightChange.compositeDelta > 0 ? 'jumped' : 'dropped'} by {Math.abs(lastWeightChange.compositeDelta).toFixed(1)}
                              </div>
                          </div>
                      </div>
                  )}
              </div>

              <div className="mt-6 pt-4 border-t border-gray-800">
                  <label className="flex items-center gap-2 cursor-pointer group">
                      <input type="checkbox" className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-accent-purple focus:ring-accent-purple focus:ring-offset-gray-900" />
                      <span className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors">Apply weights to RL training system</span>
                  </label>
              </div>
          </div>

          {/* Findings Log & Simulate */}
          <div className="col-span-2 flex flex-col gap-6">
              
              <div className="glass-panel p-5 flex-1 flex flex-col min-h-0">
                  <h3 className="text-white font-medium mb-4 flex items-center gap-2">🔍 Evaluation Findings</h3>
                  <div className="flex-1 overflow-y-auto space-y-3 pr-2">
                      {findings.map((finding: any, i: number) => {
                          let fColor = 'border-accent-green text-accent-green';
                          let bg = 'bg-accent-green/5';
                          let icon = <CheckCircle size={18} className="mt-0.5 shrink-0" />;
                          if (finding.type === 'warn') {
                              fColor = 'border-accent-gold text-accent-gold';
                              bg = 'bg-accent-gold/5';
                              icon = <AlertTriangle size={18} className="mt-0.5 shrink-0" />;
                          } else if (finding.type === 'fail') {
                              fColor = 'border-accent-red text-accent-red';
                              bg = 'bg-accent-red/5';
                              icon = <X size={18} className="mt-0.5 shrink-0 bg-accent-red/20 rounded-full" />;
                          }

                          return (
                              <div key={i} className={`border-l-4 ${fColor} ${bg} p-3 rounded-r-lg`}>
                                  <div className="flex gap-3">
                                      {icon}
                                      <div className="flex-1">
                                          <div className="text-white font-medium flex justify-between items-center">
                                              {finding.msg}
                                              <span className={`text-[10px] uppercase px-2 py-0.5 rounded border border-gray-700 font-mono text-gray-400 bg-black/30`}>{finding.category}</span>
                                          </div>
                                          <div className="text-sm text-gray-400 mt-1"><span className="text-gray-500">Reason:</span> {finding.reason}</div>
                                          <div className="text-sm mt-1"><span className="text-gray-500">Impact:</span> <span className={finding.type==='fail'?'text-accent-red/80':'text-gray-400'}>{finding.impact}</span></div>
                                          {finding.fix && (
                                              <div className="text-sm mt-2 text-accent-cyan/90 bg-accent-cyan/10 p-2 rounded-md border border-accent-cyan/20">
                                                  <span className="font-bold">💡 Fix:</span> {finding.fix}
                                              </div>
                                          )}
                                      </div>
                                  </div>
                              </div>
                          )
                      })}
                  </div>
              </div>

              <div className="glass-panel p-4 flex justify-between items-center bg-gradient-to-r from-[#0a0a1f] to-[#12123a] border border-accent-blue/30">
              <div className="flex items-center gap-6">
                      <button 
                          onClick={async () => {
                              setIsSimulating(true);
                              setSimResults(null);
                              try {
                                  const config = {
                                      "api_gateway": { "replicas": 2, "cpu_millicores": 500, "memory_mb": 512 },
                                      "auth": { "replicas": 2, "cpu_millicores": 1000, "memory_mb": 1024 },
                                      "catalog": { "replicas": 3, "cpu_millicores": 1000, "memory_mb": 2048 },
                                  };
                                  const res = await fetch(`${API_BASE_URL}/api/simulate`, {
                                      method: 'POST',
                                      headers: { 'Content-Type': 'application/json' },
                                      body: JSON.stringify({ config, workload_type: 'bursty', duration_s: 300 })
                                  });
                                  const data = await res.json();
                                  setSimResults(data);
                              } catch (e) {
                                  console.error(e);
                              } finally {
                                  setIsSimulating(false);
                              }
                          }}
                          disabled={isSimulating}
                          className="bg-accent-blue/20 hover:bg-accent-blue/30 text-accent-blue border border-accent-blue transition-colors px-6 py-2 rounded-lg font-semibold flex items-center gap-2 disabled:opacity-50"
                      >
                          <Play size={18} /> {isSimulating ? 'Simulating...' : 'Simulate Impact Before Deploy'}
                      </button>
                      
                      {simResults && (
                          <div className="flex gap-4 items-center animate-in fade-in duration-500">
                              <span className="text-sm text-gray-400 border-r border-gray-700 pr-4">Simulated:</span>
                              <span className="text-sm text-white font-mono flex flex-col leading-tight">Cost 
                                <span className="text-accent-orange font-bold">${simResults.cost_estimate.toFixed(2)}/mo</span>
                              </span>
                              <span className="text-sm text-white font-mono flex flex-col leading-tight">Latency 
                                <span className="text-accent-gold font-bold">{Math.round(simResults.p99_latency_ms)}ms P99</span>
                              </span>
                              <span className="text-sm text-white font-mono flex flex-col leading-tight">Security 
                                  <span className={scores.deploy_readiness==='green'?'text-accent-green font-bold':'text-accent-red font-bold'}>
                                      {findings.filter((f: any) => f.type === 'fail').length} Risks
                                  </span>
                              </span>
                          </div>
                      )}
                  </div>
              </div>

          </div>
      </div>
    </div>
  );
}

function ScoreCard({ title, icon, val, explain, fix, max, current, color }: any) {
    const pct = Math.max(0, Math.min(100, (current / max) * 100));
    const isWarning = fix != null;

    return (
        <div className="col-span-1 glass-panel p-4 flex flex-col relative overflow-hidden group">
            <div className="flex items-center gap-2 mb-3 text-gray-400">
                <div className={`w-6 h-6 rounded flex justify-center items-center ${color}/20 ${color.replace('bg-', 'text-')}`}>{icon}</div>
                <span className="text-xs uppercase tracking-wider font-semibold">{title}</span>
            </div>
            
            <div className="text-2xl font-bold text-white mb-1 font-mono tracking-tight">{val}</div>
            <div className="text-xs text-gray-500 leading-tight mb-4 h-8">{explain}</div>

            <div className="w-full h-1 bg-gray-800 rounded-full mt-auto overflow-hidden">
                <div className={`h-full ${color} transition-all duration-1000 ease-out`} style={{width: `${pct}%`}}></div>
            </div>

            {/* Hover Explainer Panel */}
            {isWarning && (
                <div className="absolute inset-x-0 bottom-0 top-16 bg-black/95 backdrop-blur-md p-3 translate-y-full group-hover:translate-y-0 transition-transform duration-300 flex flex-col justify-center border-t border-gray-800">
                    <div className="text-xs text-accent-gold mb-1 font-bold">⚠️ Issue Detected</div>
                    <div className="text-xs text-gray-300 leading-snug">{fix}</div>
                </div>
            )}
        </div>
    )
}
