from fastapi import APIRouter
from config.settings import get_default_config
from simulator.dag import ServiceDAG

router = APIRouter(prefix="/dag", tags=["dag"])
settings = get_default_config()
dag = ServiceDAG(settings)

@router.get("")
async def get_dag():
    nodes = []
    edges = []
    
    # Build a dynamic layout based on graph depth (BFS from root nodes)
    # instead of hardcoded positions per service name
    service_list = list(dag.services.values())
    
    # Find root nodes (no one points to them)
    all_downstream = set()
    for svc in service_list:
        all_downstream.update(svc.downstream)
    roots = [svc.name for svc in service_list if svc.name not in all_downstream]
    if not roots:
        roots = [service_list[0].name] if service_list else []
    
    # BFS to assign depth levels
    depth = {}
    queue = [(r, 0) for r in roots]
    visited = set()
    while queue:
        name, d = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        depth[name] = d
        svc = dag.services.get(name)
        if svc:
            for ds in svc.downstream:
                queue.append((ds, d + 1))
    
    # Assign positions: y by depth level, x spread evenly per level
    levels: dict[int, list[str]] = {}
    for name, d in depth.items():
        levels.setdefault(d, []).append(name)
    
    for level, names in levels.items():
        for i, name in enumerate(names):
            x = 100 + i * 200 + (100 if len(names) == 1 else 0)
            y = level * 150
            svc = dag.services[name]
            nodes.append({
                "id": svc.name,
                "position": {"x": x, "y": y},
                "data": {
                    "label": svc.name,
                    "base_service_time": svc.base_service_time,
                    "base_replicas": svc.base_replicas
                }
            })
    
    for svc in service_list:
        for ds in svc.downstream:
            edges.append({
                "id": f"{svc.name}-{ds}",
                "source": svc.name,
                "target": ds,
                "animated": True
            })
            
    return {"nodes": nodes, "edges": edges}
