import { useState, useCallback, useEffect } from 'react';
import ReactFlow, { 
    Background, 
    Controls, 
    useNodesState, 
    useEdgesState,
    MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Network, Server, Cpu } from 'lucide-react';
import { API_BASE_URL } from '../constants';

const defaultNodeOptions = {
    style: {
        background: 'rgba(10, 10, 31, 0.8)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(139, 92, 246, 0.3)',
        borderRadius: '8px',
        color: '#fff',
        padding: '15px',
        boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
        width: 180,
    }
};

const defaultEdgeOptions = {
    style: { stroke: '#8b5cf6', strokeWidth: 2 },
    animated: true,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#8b5cf6' }
};

export default function ArchitectureView() {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [selectedNode, setSelectedNode] = useState<any>(null);

    useEffect(() => {
        const fetchDag = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/api/dag`);
                const data = await res.json();
                
                const formattedNodes = data.nodes.map((n: any) => ({
                    ...n,
                    ...defaultNodeOptions,
                    data: {
                        label: (
                            <div className="flex flex-col items-center gap-2">
                                <div className="text-accent-cyan bg-accent-cyan/10 p-2 rounded-full"><Server size={20}/></div>
                                <div className="font-semibold text-sm">{n.data.label}</div>
                                <div className="text-xs text-gray-400 bg-black/40 px-2 py-1 rounded w-full text-center">
                                    {n.data.base_replicas} Replicas
                                </div>
                            </div>
                        ),
                        raw: n.data
                    }
                }));

                const formattedEdges = data.edges.map((e: any) => ({
                    ...e,
                    ...defaultEdgeOptions
                }));

                setNodes(formattedNodes);
                setEdges(formattedEdges);
            } catch(e) {
                console.error("Failed to fetch DAG", e);
            }
        };
        fetchDag();
    }, []);

    const onNodeClick = useCallback((_: any, node: any) => {
        setSelectedNode(node);
    }, []);

    return (
        <div className="h-full flex flex-col gap-6">
            <div className="flex justify-between items-center bg-bg-secondary p-4 rounded-xl border border-gray-800 shrink-0">
                <h2 className="text-xl font-bold flex items-center gap-2">
                    <Network className="text-accent-cyan" /> Architecture Visualizer
                </h2>
                <div className="flex gap-2">
                    <span className="bg-accent-purple/20 text-accent-purple text-xs font-mono px-3 py-1 rounded-full border border-accent-purple/30">
                        Graph Context: Active
                    </span>
                    <span className="bg-accent-green/20 text-accent-green text-xs font-mono px-3 py-1 rounded-full border border-accent-green/30">
                        {nodes.length} Nodes
                    </span>
                </div>
            </div>

            <div className="flex gap-6 flex-1 h-[600px]">
                {/* Flow Graph */}
                <div className="flex-1 glass-panel overflow-hidden border-gray-800">
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        onNodeClick={onNodeClick}
                        fitView
                        attributionPosition="bottom-right"
                    >
                        <Background color="#333" gap={16} />
                        <Controls className="!bg-bg-secondary !border-gray-800 !fill-white" />
                    </ReactFlow>
                </div>

                {/* Properties Inspector */}
                <div className="w-80 glass-panel p-5 flex flex-col shrink-0">
                    <h3 className="text-white font-medium mb-4 flex items-center gap-2 border-b border-gray-800 pb-3">
                        <Cpu size={18} className="text-gray-400" />
                        Service Inspector
                    </h3>
                    
                    {selectedNode ? (
                        <div className="animate-in fade-in slide-in-from-right-4 duration-300">
                            <div className="bg-bg-primary border border-gray-800 p-4 rounded-lg mb-4 text-center">
                                <div className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-accent-cyan to-accent-blue mb-1">
                                    {selectedNode.data.raw.label}
                                </div>
                                <div className="text-xs text-gray-500 font-mono">ID: {selectedNode.id}</div>
                            </div>

                            <div className="space-y-4">
                                <div>
                                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Base Component Time</div>
                                    <div className="text-white font-mono text-lg">{selectedNode.data.raw.base_service_time} ms</div>
                                </div>
                                
                                <div>
                                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Initial Replicas</div>
                                    <div className="text-white font-mono text-lg">{selectedNode.data.raw.base_replicas}</div>
                                </div>
                                
                                <div className="pt-4 border-t border-gray-800/50 mt-4">
                                    <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Connected Downstream</div>
                                    <div className="flex flex-wrap gap-2">
                                        {edges.filter(e => e.source === selectedNode.id).map(e => (
                                            <span key={e.id} className="text-xs font-mono bg-white/5 border border-white/10 px-2 py-1 rounded text-gray-300">
                                                {e.target}
                                            </span>
                                        ))}
                                        {edges.filter(e => e.source === selectedNode.id).length === 0 && (
                                            <span className="text-sm text-gray-500 italic">None (Leaf Node)</span>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center flex-1 text-gray-500 text-sm text-center px-4">
                            <Network size={32} className="mb-3 opacity-20" />
                            Click any node in the graph to view its real-time telemetry and optimization configuration.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
