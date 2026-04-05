import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Cpu, Database, Network as NetworkIcon, Sparkles } from 'lucide-react';
import { API_BASE_URL } from '../constants';

export default function PromptDashboard() {
  const [prompt, setPrompt] = useState("");
  const [target, setTarget] = useState("terraform");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, target })
      });
      const data = await res.json();
      // Store in localStorage or state management in real app
      localStorage.setItem('inframind_candidates', JSON.stringify(data.candidates));
      navigate('/code');
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const templates = [
    { icon: <Cpu />, label: "Simple EC2", text: "Create an EC2 instance with autoscaling and a basic VPC" },
    { icon: <Database />, label: "VPC + RDS", text: "Create a VPC with public/private subnets and an RDS postgres instance" },
    { icon: <NetworkIcon />, label: "Full Microservices", text: "Create API Gateway, Auth service, and Database in a secure ECS cluster" },
  ];

  return (
    <div className="max-w-4xl mx-auto mt-8">
      <div className="mb-12 text-center">
        <h1 className="text-4xl font-bold text-white mb-4">
          Infrastructure <span className="bg-clip-text text-transparent bg-gradient-to-r from-accent-purple to-accent-cyan">Intelligence</span>
        </h1>
        <p className="text-gray-400 text-lg">
          Describe your requirements. We'll generate, evaluate, and optimize the code.
        </p>
      </div>

      <div className="glass-panel p-6 mb-8">
        <div className="flex gap-4 mb-4">
          <select 
            className="bg-bg-secondary border border-gray-700 text-white rounded-lg px-4 py-2 outline-none focus:border-accent-purple transition-colors"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="terraform">Terraform</option>
            <option value="k8s">Kubernetes</option>
            <option value="docker">Docker Compose</option>
          </select>
        </div>

        <textarea
          className="w-full bg-bg-secondary/50 border border-gray-700 text-white rounded-lg p-4 h-48 outline-none focus:border-accent-purple transition-colors resize-none placeholder:text-gray-600 mb-4 font-mono text-sm leading-relaxed"
          placeholder="Describe your infrastructure... e.g., Create a highly available web server configuration with an application load balancer."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />

        <div className="flex justify-end">
          <button 
            onClick={handleGenerate}
            disabled={isLoading || !prompt.trim()}
            className="bg-gradient-to-r from-accent-purple to-accent-cyan hover:opacity-90 disabled:opacity-50 text-white px-8 py-3 rounded-lg font-semibold flex items-center gap-2 transition-all"
          >
            {isLoading ? (
              <span className="animate-spin text-xl">◌</span>
            ) : (
              <Sparkles size={20} />
            )}
            {isLoading ? "Reasoning & Generating..." : "Generate Infrastructure"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {templates.map((t, i) => (
          <div 
            key={i}
            onClick={() => setPrompt(t.text)}
            className="glass-panel p-4 cursor-pointer hover:bg-white/5 transition-colors group"
          >
            <div className="text-accent-purple mb-2 opacity-70 group-hover:opacity-100 transition-opacity">
              {t.icon}
            </div>
            <h3 className="text-white font-medium mb-1">{t.label}</h3>
            <p className="text-gray-500 text-sm">{t.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
