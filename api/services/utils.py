"""Shared utility functions for InfraMIND API services."""

import numpy as np
from typing import Dict, Any


def clean_config_for_json(config: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal keys and convert numpy types for JSON serialization.
    
    Used by both training_ws.py and comparison.py to normalize
    optimizer output before sending to the frontend.
    """
    clean = {}
    for svc, params in config.items():
        if svc.startswith("_"):
            continue  # Skip _global metadata
        clean[svc] = {
            k: int(v) if isinstance(v, (np.integer, int)) else round(float(v), 2)
            for k, v in params.items()
        }
    return clean
