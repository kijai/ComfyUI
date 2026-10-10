"""Quad patterns for N-sided patches (N = 3..6) with given boundary subdivisions, a port of Takayama et al.'s patchgen
("Pattern-Based Quadrangulation for N-Sided Patches", SGP 2014) as used by QuadRetopology. Each pattern is a small base
topology plus integer counts (padding p, inner padding q, edge-flow counts x, y, z, w) tied to the side subdivisions by a
linear system; the counts become quad strips (padding) and edge loops (strip splits). Patterns are tried in the original
order over all side rotations and flips; the first feasible one is used."""
from __future__ import annotations

import functools
import math
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import Bounds, LinearConstraint, milp

# Pattern table: variables (matrix columns), constraint matrix, rhs offsets (A v = l - c), whether x is held to the middle
# of its feasible range, extra "a <= b" constraints, objective variables to maximise, interior vertex count, base faces and
# edge-loop insertions (parameter, vertex whose outgoing boundary halfedge starts the strip).
_P = {
    (3, 0): dict(v="p0 p1 p2", A=[[0, 1, 1], [1, 0, 1], [1, 1, 0]], c=[2, 1, 1], nv=1, faces=["C0 V0 C1 C2"], ins=[]),
    (3, 1): dict(v="p0 p1 p2 q1 q2 x", A=[[0, 1, 1, 1, 1, 2], [1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0]], c=[4, 1, 1], mid_x=True,
                 le=[("p1", "q1"), ("p2", "q2")], obj="p0 p1 p2",
                 nv=4, faces=["C0 V0 V3 C2", "V0 V1 V2 V3", "C2 V3 V2 C1"], ins=[("x", "V0"), ("q1", "V2"), ("q2", "V1")]),
    (3, 2): dict(v="p0 p1 p2 q2 x", A=[[0, 1, 1, 1, 2], [1, 0, 1, 1, 0], [1, 1, 0, 0, 0]], c=[3, 2, 1], mid_x=True,
                 le=[("p2", "q2")], obj="p0 p1 p2",
                 nv=5, faces=["C0 V0 V3 C2", "V0 V1 V4 V3", "C1 V2 V4 V1", "C2 V3 V4 V2"], ins=[("x", "V0"), ("q2", "V1")]),
    (4, 0): dict(v="p0 p1", A=[[0, 1], [1, 0], [0, 1], [1, 0]], c=[1, 1, 1, 1], nv=0, faces=["C0 C1 C2 C3"], ins=[]),
    (4, 1): dict(v="p0 p1 p2 p3 x", A=[[0, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]], c=[2, 2, 1, 1],
                 le=[("p0", "p2"), ("p1", "p3")], obj="p0 p1",
                 nv=3, faces=["C0 V0 V2 C3", "V0 C1 V1 V2", "V1 C2 C3 V2"], ins=[("x", "V0")]),
    (4, 2): dict(v="p0 p1 p2 p3 x y", A=[[0, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 1]],
                 c=[3, 1, 1, 1], le=[("p0", "p2"), ("p1", "p3")], obj="p0 p1",
                 nv=2, faces=["C0 V0 C2 C3", "V0 V1 C1 C2"], ins=[("x", "V1"), ("y", "C1")]),
    (4, 3): dict(v="p0 p1 p2 p3 q1 x", A=[[0, 1, 0, 1, 1, 2], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 0]],
                 c=[3, 1, 1, 1], le=[("p0", "p2"), ("p1", "p3"), ("p3", "q1")], obj="p0 p1",
                 nv=4, faces=["C0 V0 V3 C3", "V0 V1 V2 V3", "V1 C1 C2 V2", "C2 C3 V3 V2"], ins=[("x", "V0"), ("q1", "V1")]),
    (4, 4): dict(v="p0 p1 p2 p3 q1 x y", A=[[0, 1, 0, 1, 1, 2, 1], [1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 1, 0, 0],
                                             [1, 0, 1, 0, 0, 0, 0]],
                 c=[4, 2, 1, 1], le=[("p0", "p2"), ("p1", "p3"), ("p3", "q1")], obj="p0 p1",
                 nv=7, faces=["C0 V0 V6 C3", "V0 V1 V5 V6", "V1 V2 V4 V5", "V2 C1 V3 V4", "V3 C2 V5 V4", "C2 C3 V6 V5"],
                 ins=[("x", "V0"), ("y", "V2"), ("q1", "V1")]),
    (5, 0): dict(v="p0 p1 p2 p3 p4", A=[[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
                 c=[2, 1, 1, 1, 1], nv=1, faces=["V0 C3 C4 C0", "V0 C1 C2 C3"], ins=[]),
    (5, 1): dict(v="p0 p1 p2 p3 p4 q4 x", A=[[0, 1, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0],
                                              [0, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 0, 0]],
                 c=[2, 1, 1, 1, 1], mid_x=True, le=[("p4", "q4")], obj="p0 p1 p2 p3 p4",
                 nv=1, faces=["C0 V0 C1 C2", "C0 C2 C3 C4"], ins=[("x", "V0"), ("q4", "C1")]),
    (5, 2): dict(v="p0 p1 p2 p3 p4 q0 q1 q4 x", A=[[0, 1, 0, 0, 1, 0, 1, 1, 2], [1, 0, 1, 0, 0, 1, 0, 0, 0],
                                                    [0, 1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 1, 0],
                                                    [1, 0, 0, 1, 0, 1, 0, 0, 0]],
                 c=[4, 1, 1, 1, 1], mid_x=True, le=[("p0", "q0"), ("p1", "q1"), ("p4", "q4")], obj="p0 p1 p2 p3 p4",
                 nv=5, faces=["C0 V0 V3 V4", "C1 V4 V3 V2", "V0 V1 V2 V3", "V4 C3 C4 C0", "V4 C1 C2 C3"],
                 ins=[("x", "V0"), ("q0", "C0"), ("q1", "V1"), ("q4", "V2")]),
    (5, 3): dict(v="p0 p1 p2 p3 p4 q1 q4 x y", A=[[0, 1, 0, 0, 1, 1, 1, 2, 1], [1, 0, 1, 0, 0, 0, 0, 0, 1],
                                                   [0, 1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0, 0],
                                                   [1, 0, 0, 1, 0, 0, 0, 0, 0]],
                 c=[5, 2, 1, 1, 1], mid_x=True, le=[("p1", "q1"), ("p4", "q4")], obj="p0 p1 p2 p3 p4",
                 nv=9, faces=["C0 V0 V5 C4", "V0 V1 V6 V5", "V1 V2 V7 V6", "V2 V3 V8 V7", "V3 C1 V4 V8", "V4 C2 V7 V8",
                              "C2 C3 V6 V7", "C3 C4 V5 V6"], ins=[("x", "V0"), ("y", "V3"), ("q1", "V2"), ("q4", "V1")]),
    (6, 0): dict(v="p0 p1 p2 p3 p4 p5 x", A=[[0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0],
                                              [0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0]],
                 c=[1, 1, 1, 1, 1, 1], mid_x=True, obj="p0 p1 p2 p3 p4 p5",
                 nv=0, faces=["C0 C1 C2 C5", "C2 C3 C4 C5"], ins=[("x", "C1")]),
    (6, 1): dict(v="p0 p1 p2 p3 p4 p5 x y z w", A=[[0, 1, 0, 0, 0, 1, 1, 1, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                                                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                                                    [0, 0, 0, 1, 0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
                 c=[2, 2, 1, 1, 1, 1], mid_x=True, obj="p0 p1 p2 p3 p4 p5",
                 nv=4, faces=["C0 V0 V2 V3", "C1 V1 V2 V0", "V1 C2 V3 V2", "C2 C3 C4 V3", "C4 C5 C0 V3"],
                 ins=[("x", "V0"), ("y", "C1"), ("z", "V1"), ("w", "C3")]),
    (6, 2): dict(v="p0 p1 p2 p3 p4 p5 q0 q3 x y", A=[[0, 1, 0, 0, 0, 1, 0, 0, 2, 1], [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                                      [0, 1, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 1, 0, 0, 0]],
                 c=[3, 1, 1, 1, 1, 1], mid_x=True, le=[("p0", "q0"), ("p3", "q3")], obj="p0 p1 p2 p3 p4 p5",
                 nv=6, faces=["C0 V0 V2 V4", "V0 V1 V3 V2", "V1 C1 V5 V3", "C1 C2 C3 V5", "C3 C4 V4 V5", "C4 C5 C0 V4",
                              "V2 V3 V5 V4"], ins=[("x", "V0"), ("y", "V1"), ("q0", "C0"), ("q3", "C3")]),
    (6, 3): dict(v="p0 p1 p2 p3 p4 p5 q3 x y z", A=[[0, 1, 0, 0, 0, 1, 0, 2, 1, 1], [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                                                     [0, 1, 0, 1, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                                                     [0, 0, 0, 1, 0, 1, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
                 c=[4, 2, 1, 1, 1, 1], mid_x=True, obj="p0 p1 p2 p3 p4 p5",
                 nv=9, faces=["C0 V0 V6 C5", "V0 V1 V5 V6", "V1 V2 V4 V5", "V2 C1 V3 V4", "V3 C2 V5 V4", "C2 C3 V8 V5",
                              "C3 C4 V7 V8", "C4 C5 V6 V7", "V5 V8 V7 V6"], ins=[("x", "V0"), ("y", "V2"), ("z", "V1"), ("q3", "C3")]),
}


def _solve(a_eq, b_eq, le_rows, le_rhs, obj):
    """Nonnegative integer v with a_eq v = b_eq and le_rows v <= le_rhs minimising obj, or None (patchgen's lp_solve ILP)."""
    cons = [LinearConstraint(a_eq, b_eq, b_eq)]
    if le_rows:
        cons.append(LinearConstraint(np.array(le_rows), -np.inf, np.array(le_rhs, dtype=float)))
    res = milp(obj, constraints=cons, integrality=np.ones(a_eq.shape[1]), bounds=Bounds(0, np.inf))
    return np.rint(res.x).astype(np.int64) if res.status == 0 else None


def _parameters(spec, l_perm):
    names = spec["v"].split()
    eye = dict(zip(names, np.eye(len(names))))
    a = np.array(spec["A"], dtype=float)
    b = np.array(l_perm, dtype=float) - np.array(spec["c"], dtype=float)
    le_rows = [eye[lhs] - eye[rhs] for lhs, rhs in spec.get("le", [])]
    le_rhs = [0.0] * len(le_rows)
    if spec.get("mid_x"):
        # x within one of the middle of its feasible range
        lo = _solve(a, b, [], [], eye["x"])
        hi = _solve(a, b, [], [], -eye["x"]) if lo is not None else None
        if hi is None:
            return None
        mid = (lo + hi)[names.index("x")] // 2
        le_rows += [eye["x"], -eye["x"]]
        le_rhs += [mid + 1, -(mid - 1)]
    v = _solve(a, b, le_rows, le_rhs, -sum(eye[k] for k in spec.get("obj", spec["v"]).split()))
    return None if v is None else {k: int(x) for k, x in zip(names, v)}


class _QuadPatch:
    """Minimal quad mesh with corner tags; faces counter-clockwise."""

    def __init__(self, n_corners, n_inner, faces):
        self.corner = list(range(n_corners)) + [-1] * n_inner
        self.tag = {f"C{i}": i for i in range(n_corners)} | {f"V{i}": n_corners + i for i in range(n_inner)}
        self.faces = [[self.tag[t] for t in f.split()] for f in faces]

    def new_vertex(self):
        self.corner.append(-1)
        return len(self.corner) - 1

    def _face_edges(self):
        return {(f[i], f[(i + 1) % 4]): fi for fi, f in enumerate(self.faces) for i in range(4)}

    def boundary_next(self):
        """Boundary halfedges (a -> b, opposite to the face orientation) as a successor map a -> b."""
        fe = self._face_edges()
        return {b: a for (a, b) in fe if (b, a) not in fe}

    def insert_edgeloop(self, v):
        """Split the quad strip that starts at v's outgoing boundary halfedge."""
        w = self.boundary_next()[v]
        fe = self._face_edges()
        crossed, strip = [], []
        a, b = w, v                                   # face edge opposite to the boundary halfedge v -> w
        while (a, b) in fe:
            fi = fe[(a, b)]
            if strip and fi == strip[0]:
                break
            f = self.faces[fi]
            i = f.index(a)
            crossed.append((a, b))
            strip.append(fi)
            a, b = f[(i + 3) % 4], f[(i + 2) % 4]     # the neighbour across the opposite edge has it this way round
        crossed.append((b, a))
        mid = {k: self.new_vertex() for k in dict.fromkeys(tuple(sorted(e)) for e in crossed)}
        for fi in strip:
            f = self.faces[fi]
            for i in range(4):
                e1, e2 = tuple(sorted((f[i], f[(i + 1) % 4]))), tuple(sorted((f[(i + 2) % 4], f[(i + 3) % 4])))
                if e1 in mid and e2 in mid:
                    self.faces[fi] = [f[i], mid[e1], mid[e2], f[(i + 3) % 4]]
                    self.faces.append([mid[e1], f[(i + 1) % 4], f[(i + 2) % 4], mid[e2]])
                    break

    def add_padding(self, p):
        """p[i] quad strips along side i, corners moved outward (patchgen add_padding)."""
        n = len(p)
        for i in range(n - 1, -1, -1):
            for _ in range(p[i]):
                nxt = self.boundary_next()
                chain = [self.corner.index((i + 1) % n)]
                while self.corner[chain[-1]] != i or len(chain) == 1:
                    chain.append(nxt[chain[-1]])
                outer = [self.new_vertex() for _ in chain]
                self.faces += [[chain[j], chain[j + 1], outer[j + 1], outer[j]] for j in range(len(chain) - 1)]
                self.corner[chain[0]] = self.corner[chain[-1]] = -1
                self.corner[outer[0]] = (i + 1) % n
                self.corner[outer[-1]] = i


@functools.lru_cache(maxsize=None)
def pattern(l: Tuple[int, ...]):
    """Quad patch for side subdivisions l: 2D vertices (V, 2), quads (F, 4), and per side the vertex chain from corner i to
    corner i+1 (counter-clockwise). None if no pattern fits."""
    n = len(l)
    if not 3 <= n <= 6 or sum(l) % 2 or sum(l) < 4:
        return None
    for spec in [spec for key, spec in _P.items() if key[0] == n]:
        for pid in range(2 * n):
            # patchgen Permutation: rotation by pid for pid < n, flipped rotation otherwise
            perm = [((i + pid) % n) if pid < n else (n - 1 - (i + pid - n) % n) for i in range(n)]
            prm = _parameters(spec, [l[j] for j in perm])
            if prm is None:
                continue
            patch = _QuadPatch(n, spec["nv"], spec["faces"])
            for name, start in spec["ins"]:
                for _ in range(prm[name]):
                    patch.insert_edgeloop(patch.tag[start])
            patch.add_padding([prm.get(f"p{i}", 0) for i in range(n)])
            patch.corner = [-1 if c < 0 else (perm[c] + (pid >= n)) % n for c in patch.corner]
            if pid >= n:
                patch.faces = [f[::-1] for f in patch.faces]
            return _geometry(patch, l)
    return None


def _geometry(patch: _QuadPatch, l):
    """Vertex positions (boundary on the regular n-gon, interior by the uniform Laplacian), quads and side chains."""
    n = len(l)
    nv = len(patch.corner)
    faces = np.array(patch.faces, dtype=np.int64)
    succ = {a: b for b, a in patch.boundary_next().items()}          # counter-clockwise boundary successor (face order)
    angle = 2 * math.pi / n
    corners = np.array([(math.cos(angle * i - 0.5 * math.pi - 0.5 * angle), math.sin(angle * i - 0.5 * math.pi - 0.5 * angle)) for i in range(n)])
    pos = np.zeros((nv, 2))
    fixed = np.zeros(nv, dtype=bool)
    v = patch.corner.index(0)
    sides = []
    for i in range(n):
        side = [v]
        for j in range(l[i]):
            s = (i + j / l[i]) % 1                                    # the perimeter position i + j / l[i] within side i
            pos[v] = (1 - s) * corners[i] + s * corners[(i + 1) % n]
            fixed[v] = True
            v = succ[v]
            side.append(v)
        sides.append(side)
    free = np.nonzero(~fixed)[0]
    if free.size:
        e = np.concatenate([faces[:, [k, (k + 1) % 4]] for k in range(4)])
        e = np.concatenate([e, e[:, ::-1]])
        adj = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
        adj.data[:] = 1.0
        lap = sp.diags(np.asarray(adj.sum(1)).ravel()) - adj
        rhs = -lap[free][:, np.nonzero(fixed)[0]] @ pos[fixed]
        pos[free] = spla.spsolve(lap[free][:, free].tocsc(), rhs).reshape(-1, 2)
    return pos, faces, sides
