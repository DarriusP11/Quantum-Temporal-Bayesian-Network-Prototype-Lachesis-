# foresight.py
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import numpy as np

@dataclass
class SweepSpec:
    shots: int
    seeds: List[int]
    pdep: List[float]
    pamp: List[float]
    pph:  List[float]
    pcnot: List[float]

def aggregate_counts(runs: List[Dict[str,int]], keys: List[str]) -> Dict[str, float]:
    tot = {k:0 for k in keys}
    for c in runs:
        for k in keys:
            tot[k] += c.get(k,0)
    N = sum(tot.values()) or 1
    return {k: tot[k]/N for k in keys}

def kl_div(p: Dict[str,float], q: Dict[str,float], eps=1e-12) -> float:
    k = set(p) | set(q)
    s=0.0
    for key in k:
        a = max(p.get(key,0.0), eps)
        b = max(q.get(key,0.0), eps)
        s += a*np.log(a/b)
    return float(s)

def blend(a: Dict[str,float], b: Dict[str,float], w: float) -> Dict[str,float]:
    keys = set(a)|set(b)
    return {k: (1-w)*a.get(k,0.0)+w*b.get(k,0.0) for k in keys}
