import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, Code2, Activity, PlayCircle, BarChart3, ArrowLeftRight, Network, Trash2 } from 'lucide-react';

import PromptDashboard from './pages/PromptDashboard';
import CodeViewer from './pages/CodeViewer';
import ExecutionSimulator from './pages/ExecutionSimulator';
import TrainingDashboard from './pages/TrainingDashboard';
import ArchitectureView from './pages/ArchitectureView';
import EvaluationPanel from './pages/EvaluationPanel';
import ComparisonPanel from './pages/ComparisonPanel';

const Sidebar = () => {
  const navItems = [
    { path: '/', label: '1. Prompt Dashboard', icon: <LayoutDashboard size={20} /> },
    { path: '/architecture', label: '2. Architecture Viz', icon: <Network size={20} /> },
    { path: '/simulator', label: '3. Execution Simulator', icon: <PlayCircle size={20} /> },
    { path: '/evaluation', label: '4. Evaluation Panel', icon: <Activity size={20} /> },
    { path: '/training', label: '5. RL Training', icon: <BarChart3 size={20} /> },
    { path: '/compare', label: '6. Comparison & Deploy', icon: <ArrowLeftRight size={20} /> },
    { path: '/code', label: 'Advanced Code View', icon: <Code2 size={20} /> },
  ];

  return (
    <div className="w-64 glass-panel border-l-0 border-t-0 border-b-0 rounded-none h-screen fixed left-0 top-0 flex flex-col p-4 z-10">
      <div className="flex items-center gap-2 mb-8 mt-2 px-2">
        <span className="text-2xl">🧠</span>
        <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-accent-purple to-accent-cyan">
          InfraMIND <span className="text-sm font-normal text-gray-400">v4</span>
        </h1>
      </div>
      <nav className="flex flex-col gap-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive 
                  ? 'bg-white/10 text-white' 
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
              }`
            }
          >
            {item.icon}
            <span className="font-medium">{item.label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto border-t border-gray-800 pt-4">
        <button
          onClick={() => {
            localStorage.clear();
            window.location.href = '/';
          }}
          className="flex w-full items-center gap-3 px-4 py-3 rounded-lg transition-colors text-accent-red hover:bg-accent-red/10"
        >
          <Trash2 size={20} />
          <span className="font-medium">Reset Session</span>
        </button>
      </div>
    </div>
  );
};

const Layout = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className="flex min-h-screen relative">
      <div className="bg-glow bg-glow-1"></div>
      <div className="bg-glow bg-glow-2"></div>
      <Sidebar />
      <main className="flex-1 ml-64 p-8 relative z-0">
        {children}
      </main>
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<PromptDashboard />} />
          <Route path="/code" element={<CodeViewer />} />
          <Route path="/evaluation" element={<EvaluationPanel />} />
          <Route path="/simulator" element={<ExecutionSimulator />} />
          <Route path="/training" element={<TrainingDashboard />} />
          <Route path="/compare" element={<ComparisonPanel />} />
          <Route path="/architecture" element={<ArchitectureView />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
