"""Patch sides and subsides (side pieces shared with one neighbour, the quantization unknowns), as computeChartData."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from .field import opposite_halfedges


@dataclass
class Subside:
    vertices: List[int]
    steps: List[float]                # length of each segment over the quad size there
    charts: List[int]                 # [this chart, neighbour chart or -1]
    chart_sides: List[int]            # side index in each incident chart

    @property
    def length(self) -> float:
        return sum(self.steps)


@dataclass
class Side:
    subsides: List[int] = field(default_factory=list)
    reversed: List[bool] = field(default_factory=list)


@dataclass
class Chart:
    faces: np.ndarray
    sides: List[Side] = field(default_factory=list)
    subsides: List[int] = field(default_factory=list)


def _chain_config(start: int, nxt: dict) -> dict:
    """Pick, for vertices with several outgoing border edges, the choice that closes the loop back at start."""
    config = {}

    def rec(v, depth):
        if depth > 0 and v == start:
            return True
        if depth > len(nxt) + 1:
            return False
        for i, w in enumerate(nxt[v]):
            config[v] = i
            if rec(w, depth + 1):
                return True
        return False

    if any(len(n) > 1 for n in nxt.values()):
        rec(start, 0)
    return config


def compute_chart_data(verts: np.ndarray, faces: np.ndarray, face_label: np.ndarray, corners: List[List[int]],
                       size: np.ndarray):
    """Charts and subsides from per-face patch labels and per-patch corner vertices."""
    opp = opposite_halfedges(torch.as_tensor(faces)).numpy()
    opp_label = np.where(opp >= 0, face_label[np.maximum(opp, 0) // 3], -1)
    a, b = faces.reshape(-1), np.roll(faces, -1, 1).reshape(-1)
    order = np.argsort(face_label, kind="stable")
    bounds = np.searchsorted(face_label[order], np.arange(face_label.max() + 2))
    charts = [Chart(order[bounds[c]:bounds[c + 1]]) for c in range(len(bounds) - 1)]
    ek = lambda u, v: (min(u, v), max(u, v))
    seg_len = lambda u, v: float(np.linalg.norm(verts[v] - verts[u]) * 2 / (size[u] + size[v]))

    subsides: List[Subside] = []
    edge_subside = {}
    for c, chart in enumerate(charts):
        h = (3 * chart.faces[:, None] + np.arange(3)).reshape(-1)
        h = h[opp_label[h] != c]
        edge_label, nxt = {}, {}
        for u, v, label in zip(a[h].tolist(), b[h].tolist(), opp_label[h].tolist()):
            edge_label[ek(u, v)] = label
            nxt.setdefault(u, []).append(v)
        remaining = set(nxt) | {w for n in nxt.values() for w in n}
        while remaining:
            cur = min(remaining)
            config = _chain_config(cur, nxt)
            step = lambda v: nxt[v][config.get(v, 0)]
            start = cur
            for _ in range(len(nxt) + 1):
                cur = step(cur)
                if cur in corners[c] or cur == start:
                    break
            start = cur
            side, side_id = Side(), 0
            while True:
                sid, sub_start = len(subsides), cur
                adj = edge_label[ek(cur, step(cur))]
                adj_corners = corners[adj] if adj >= 0 else ()
                sub_verts, steps, is_corner = [], [], False
                while True:
                    nxt_v = step(cur)
                    key = ek(cur, nxt_v)
                    if key not in edge_subside:
                        sub_verts.append(cur)
                        steps.append(seg_len(cur, nxt_v))
                        edge_subside[key] = sid
                    elif cur == sub_start:
                        sid = edge_subside[key]
                    remaining.discard(cur)
                    cur = nxt_v
                    if cur == sub_start:
                        break
                    is_corner = cur in corners[c]
                    if is_corner or cur in adj_corners or edge_label[ek(cur, step(cur))] != adj:
                        break
                if sub_verts or adj >= 0:
                    chart.subsides.append(sid)
                if sub_verts:
                    subsides.append(Subside(sub_verts + [cur], steps, [c, adj], [side_id, -1]))
                elif adj >= 0:
                    subsides[sid].chart_sides[1] = side_id
                side.subsides.append(sid)
                side.reversed.append(not sub_verts)
                if is_corner:
                    chart.sides.append(side)
                    side, side_id = Side(), side_id + 1
                if cur == start:
                    break
    return charts, subsides
