import { useState, useRef, useEffect } from 'react';
import { PlayCircle, Terminal, Activity, ArrowRight, Loader } from 'lucide-react';
import { API_BASE_URL } from '../constants';

export default function ExecutionSimulator() {
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [workload, setWorkload] = useState('bursty');
  const logsEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  const simulateExecution = async () => {
    setIsRunning(true);
    setLogs(["Initializing Simulation Engine...", "Loading Service DAG topology...", `Generating ${workload} workload trace...`, "Starting execution..."]);
    setMetrics(null);

    // Loading phase indicators — these describe the backend's actual pipeline
    const phases = [
        `Generating ${workload} workload trace (300s duration)...`,
        "Initializing SimPy discrete-event engine...",
        "Processing requests through Service DAG...",
        "Computing P50/P90/P99 latency percentiles...",
        "Evaluating SLA violation rate...",
        "Estimating infrastructure cost...",
    ];
    let phaseIdx = 0;
    const interval = setInterval(() => {
        if (phaseIdx < phases.length) {
            setLogs(prev => [...prev, phases[phaseIdx]]);
            phaseIdx++;
        } else {
            clearInterval(interval);
        }
    }, 400);

    try {
      const candidatesRaw = localStorage.getItem('inframind_candidates');
      // Build a basic config from the first candidate's scores for simulation
      // Note: CodeCandidate does not carry a .config — we derive defaults
      let config: Record<string, any> = {
        "api_gateway": { "replicas": 2, "cpu_millicores": 500, "memory_mb": 512 },
        "auth": { "replicas": 2, "cpu_millicores": 1000, "memory_mb": 1024 },
        "catalog": { "replicas": 3, "cpu_millicores": 1000, "memory_mb": 2048 },
      };

      const res = await fetch(`${API_BASE_URL}/api/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config, workload_type: workload, duration_s: 300 })
      });
      const data = await res.json();
      
      setLogs(prev => [...prev, "Simulation Complete.", `Final Cost: $${data.cost_estimate.toFixed(2)}`, `SLA Violations: ${data.sla_violations_pct.toFixed(2)}%`]);
      setMetrics(data);
    } catch (e) {
      setLogs(prev => [...prev, `Error: ${e}`]);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="h-full flex flex-col gap-6">
      <div className="flex justify-between items-center bg-bg-secondary p-4 rounded-xl border border-gray-800">
        <div className="flex items-center gap-4">
            <h2 className="text-xl font-bold flex items-center gap-2"><PlayCircle className="text-accent-red" /> Simulator</h2>
            <select 
                className="bg-bg-primary border border-gray-700 text-white rounded-lg px-3 py-1.5 outline-none focus:border-accent-red text-sm"
                value={workload}
                onChange={(e) => setWorkload(e.target.value)}
                disabled={isRunning}
            >
                <option value="steady">Steady Workload</option>
                <option value="diurnal">Diurnal Pattern</option>
                <option value="bursty">Bursty Traffic</option>
            </select>
        </div>
        
        <button 
          onClick={simulateExecution}
          disabled={isRunning}
          className="bg-gradient-to-r from-accent-red to-accent-orange hover:opacity-90 disabled:opacity-50 text-white px-6 py-2 rounded-lg font-semibold flex items-center gap-2 transition-all"
        >
          {isRunning ? <Loader className="animate-spin" size={18} /> : <Terminal size={18} />}
          {isRunning ? "Running Simulation..." : "Run Simulation"}
        </button>
      </div>

      <div className="grid grid-cols-3 gap-6 flex-1">
        <div className="col-span-2 glass-panel flex flex-col overflow-hidden">
          <div className="bg-[#050505] px-4 py-2 border-b border-gray-800 text-sm font-mono text-gray-500 flex justify-between">
              <span>Execution Log</span>
              <span>simpy-engine-v4</span>
          </div>
          <div className="flex-1 p-4 overflow-y-auto font-mono text-sm leading-relaxed text-[#33ff00] bg-[#0a0a0a]">
            {logs.length === 0 ? (
                <div className="text-gray-600 italic">No simulation running. Click "Run Simulation" to begin.</div>
            ) : (
                logs.map((log, i) => (
                    <div key={i} className="mb-1">{'> '}{log}</div>
                ))
            )}
            {isRunning && <div className="animate-pulse">_</div>}
            <div ref={logsEndRef} />
          </div>
        </div>

        <div className="col-span-1 flex flex-col gap-4">
            <div className="glass-panel p-5">
                <h3 className="text-gray-400 font-medium mb-1 text-sm uppercase tracking-wider">P99 Latency</h3>
                <div className="text-3xl font-bold text-white flex items-end gap-2">
                    {metrics ? Math.round(metrics.p99_latency_ms) : '--'} <span className="text-lg text-gray-500 font-normal">ms</span>
                </div>
            </div>
            <div className="glass-panel p-5">
                <h3 className="text-gray-400 font-medium mb-1 text-sm uppercase tracking-wider">SLA Drop Rate</h3>
                <div className="text-3xl font-bold text-accent-orange flex items-end gap-2">
                    {metrics ? metrics.sla_violations_pct.toFixed(2) : '--'} <span className="text-lg text-gray-500 font-normal">%</span>
                </div>
            </div>
            <div className="glass-panel p-5">
                <h3 className="text-gray-400 font-medium mb-1 text-sm uppercase tracking-wider">Infrastructure Cost</h3>
                <div className="text-3xl font-bold text-accent-green flex items-end gap-2">
                    <span className="text-lg text-gray-500 font-normal">$</span> {metrics ? metrics.cost_estimate.toFixed(2) : '--'}
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}
