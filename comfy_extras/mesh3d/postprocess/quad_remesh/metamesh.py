"""QuadWild's trace removal (xfield_tracer MetaMesh::MergeLoop): a trace is dissolved by merging the patches on its
two sides when every side it bounds matches one side of one other patch and the merged patches score no worse."""
from __future__ import annotations

import numpy as np

MIN_SIDES, MAX_SIDES, MIN_ADMITTIBLE, MAX_ADMITTIBLE = 3, 5, 3, 6
_LISTS = ("V", "L", "adj", "pid", "real", "vdir")


def catmull_clarkable(L, thr):
    """IsCatmullClarkable (catmull_clarkability.h) within slack thr, with 4-sided patches skipped."""
    n = len(L)
    if n == 4:
        return True
    m = min(L, default=0.0)
    if n == 3:
        cc = min(L[i] + L[(i + 1) % 3] - L[(i + 2) % 3] + m for i in range(3))
    elif n == 5:
        cc = min(L[(i + 2) % 5] + L[(i + 3) % 5] + L[(i + 4) % 5] - L[i] - L[(i + 1) % 5] + m for i in range(5))
    elif n == 6:
        cc = min(a + b - c + min(a, b, c) for a, b, c in (L[0::2], L[1::2]))
    else:
        cc = 0.0
    return cc + thr > 0


class _Face:
    __slots__ = _LISTS + ("exp_val", "nsing")

    def __init__(self, V, L, adj, pid, real, vdir, exp_val=4, nsing=0):
        self.V, self.L, self.adj, self.pid, self.real, self.vdir = V, L, adj, pid, real, vdir
        self.exp_val, self.nsing = exp_val, nsing

    def copy(self):
        return _Face(*(list(getattr(self, a)) for a in _LISTS), self.exp_val, self.nsing)

    def sides(self):
        """Meta edge ranges from each corner to the next (a face starts at a corner whenever it has one)."""
        starts = [k for k, r in enumerate(self.real) if r]
        return [range(a, b) for a, b in zip(starts, starts[1:] + [len(self.V)])]


class MetaMesh:
    def __init__(self, tr, corners, fixed, avoid):
        """Meta faces of the tracer's current patches (corners per patch, ascending) skipping the avoided ones."""
        self.pos_id = np.unique(tr.P, axis=0, return_inverse=True)[1].reshape(-1).tolist()   # equal positions share an id
        self.fixed = set(fixed)
        meta = {v for c in corners for v in c} | self.fixed
        bnd, nxt = tr.boundary_next()
        hb = np.nonzero(bnd)[0]
        lead = np.array([c[0] if c else -1 for c in corners])                # each patch's first corner
        hb = hb[tr.head[hb] == lead[tr.label[hb // 3]]]
        part, at = np.unique(tr.label[hb // 3], return_index=True)
        all_h = np.arange(3 * tr.nf)
        head, nxt, elen, eid = tr.head.tolist(), nxt.tolist(), tr.elen.tolist(), tr.eid.tolist()
        dhead, dtail = tr.dir_at(all_h, tr.head).tolist(), tr.dir_at(all_h, tr.tail).tolist()
        self.faces = [None] * tr.npart
        self.edge_map = {}
        for p, h0 in zip(part.tolist(), hb[at].tolist()):                     # its first boundary half-edge ending there
            if p in avoid:
                continue
            V, L, vdir = [], [], []
            h = h0
            while True:
                if head[h] in meta:
                    V.append(head[h])
                    L.append(0.0)
                    vdir.append([dhead[h], None])
                L[-1] += elen[h]
                vdir[-1][1] = dtail[h]
                self.edge_map.setdefault(eid[h], []).append((p, len(V) - 1))
                h = nxt[h]
                if h == h0 or h < 0:
                    break
            self.faces[p] = _Face(V, L, [None] * len(V), [-1] * len(V), [v in corners[p] for v in V], vdir)
        for e in sorted(self.edge_map):
            if len(self.edge_map[e]) < 2:
                continue
            (f0, e0), (f1, e1) = self.edge_map[e][:2]
            a0, a1 = self.faces[f0].adj[e0], self.faces[f1].adj[e1]
            bad = a0 != (f1, e1) and (a0 is not None or a1 is not None)
            self.faces[f0].adj[e0] = None if bad else (f1, e1)
            self.faces[f1].adj[e1] = None if bad else (f0, e0)
        for i, f in enumerate(self.faces):
            if f is not None:
                self._check_adj(i)
        # InitExpectedValence: exp_val = the only singularity's valence (-1: several, 4: none); nsing = ones off corners
        pv = np.unique(np.repeat(tr.label, 3) * tr.nv + tr.F.reshape(-1))
        per = {}
        for key in pv[tr.valence[pv % tr.nv] != 4].tolist():
            per.setdefault(key // tr.nv, []).append(key % tr.nv)
        for p, vs in per.items():
            f = self.faces[p]
            if f is not None:
                f.nsing = sum(v not in corners[p] for v in vs)
                f.exp_val = -1 if len(vs) > 1 else sum(f.real) if vs[0] in corners[p] else int(tr.valence[vs[0]])

    def set_path_ids(self, tr, paths):
        """UpdatePathId: label every meta edge with the (last) trace running along it."""
        for p, (verts, loop) in enumerate(paths):
            a = np.asarray(verts)
            n = len(a) - (not loop)
            h = tr.he_of(a[:n], np.roll(a, -1)[:n])
            for e in tr.eid[h[h >= 0]].tolist():
                for F, E in self.edge_map.get(e, ()):
                    self.faces[F].pid[E] = p
        self.max_path = len(paths)
        self.path_faces = {}
        for i, f in enumerate(self.faces):
            for q in set(f.pid) if f is not None else ():
                self.path_faces.setdefault(q, set()).add(i)

    def _check_adj(self, i):
        f = self.faces[i]
        for k, a in enumerate(f.adj):
            if a is not None and (not self.faces[a[0]].V or self.faces[a[0]].adj[a[1]] != (i, k)):
                f.adj[k] = None

    def which_side(self, i, k):
        return sum(self.faces[i].real[1:k + 1])

    def side_edges(self, i, s):
        sides = self.faces[i].sides()
        return sides[s] if sides else ()

    def info(self, i):
        f = self.faces[i]
        sides = f.sides()
        return dict(corners=len(sides), genus=len({self.pos_id[v] for v in f.V}) == len(f.V), L=[sum(f.L[k] for k in E) for E in sides],
                    exp=f.exp_val, nsing=f.nsing)

    def removable_side(self, i, s):
        f = self.faces[i]
        E = self.side_edges(i, s)
        if not E or any(f.adj[k] is None or f.V[k] in self.fixed or f.V[(k + 1) % len(f.V)] in self.fixed for k in E):
            return False
        opp = {(f.adj[k][0], self.which_side(*f.adj[k])) for k in E}
        if len(opp) > 1:
            return False
        (g, sg), = opp
        if g == i or len({self.which_side(i, k) for k, a in enumerate(f.adj) if a is not None and a[0] == g}) > 1:
            return False
        return len(self.side_edges(g, sg)) == len(E)

    def path_side(self, i, p):
        f = self.faces[i]
        return self.which_side(i, f.pid.index(p)) if p in f.pid else None

    # ------------------------------------------------------------------ merging
    def merge_side(self, i0, s0):
        f0 = self.faces[i0]
        E0 = self.side_edges(i0, s0)
        i1, e = f0.adj[E0[0]]
        f1 = self.faces[i1]
        E1 = self.side_edges(i1, self.which_side(i1, e))
        new = _Face([], [], [], [], [], [], f1.exp_val if f0.exp_val == 4 else f0.exp_val if f1.exp_val == 4 else -1, f0.nsing + f1.nsing)
        for f, E in ((f0, E0), (f1, E1)):
            for a in _LISTS:
                x = getattr(f, a)
                getattr(new, a).extend(x[E.stop:] + x[:E.start])                 # the rest of the face, cyclically
        for i in (i0, i1):
            self._backup.setdefault(i, self.faces[i].copy())
        m = len(new.V)
        for k in (0, (len(f0.V) - len(E0)) % m) if m else ():
            d0, d1 = new.vdir[k][0], new.vdir[(k - 1) % m][1]
            new.real[k] = d0 % 2 != d1 % 2 or d0 == d1
        if any(new.real):
            r = new.real.index(True)
            for a in _LISTS:
                setattr(new, a, getattr(new, a)[r:] + getattr(new, a)[:r])
        self.faces[i0] = new
        self.faces[i1] = _Face([], [], [], [], [], [])
        for k, a in enumerate(new.adj):
            if a is not None:
                self._backup.setdefault(a[0], self.faces[a[0]].copy())
                self.faces[a[0]].adj[a[1]] = (i0, k)
        for a in new.adj:
            if a is not None:
                self._check_adj(a[0])
        self._check_adj(i0)

    def _merge_path(self, faces, p):
        """MergePathIndex: absorb across path p from each of the faces that border it, in order (lower index absorbs)."""
        for i in faces:
            s = self.path_side(i, p)
            while s is not None:
                if not self.removable_side(i, s):
                    return False
                self.merge_side(i, s)
                s = self.path_side(i, p)
        return True

    def _score(self, infos):
        """BetterConfiguration (MinSides 3, MaxSides 5, slack avgEdge) as a tuple, lower is better."""
        return (sum(not x["genus"] for x in infos),
                sum(not MIN_SIDES <= x["corners"] <= MAX_SIDES for x in infos),
                max([1] + [x["nsing"] for x in infos]),
                -sum(x["exp"] != 4 and x["corners"] == x["exp"] for x in infos),
                sum(not catmull_clarkable(x["L"], self.thr) for x in infos))

    def remove_if_possible(self, p):
        faces = sorted(self.path_faces.get(p, ()))
        if not faces or not all(self.removable_side(i, self.path_side(i, p)) for i in faces):
            return False
        info0 = [self.info(i) for i in faces]
        self._backup = {}
        ok = self._merge_path(faces, p)
        if ok:
            info1 = [self.info(i) for i in faces if self.faces[i].V]
            ok = all(MIN_ADMITTIBLE <= x["corners"] <= MAX_ADMITTIBLE for x in info1) and self._score(info1) <= self._score(info0)
        if not ok:
            for i, f in self._backup.items():
                self.faces[i] = f
            return False
        for i, old in self._backup.items():
            for q in set(old.pid):
                self.path_faces[q].discard(i)
            for q in set(self.faces[i].pid):
                self.path_faces.setdefault(q, set()).add(i)
        return True

    def _changed_paths(self):
        """Paths along a face the last merge changed or next to one: only their removal can turn out differently."""
        out = set()
        for i, old in self._backup.items():
            for f in (old, self.faces[i]):
                out.update(f.pid)
                for a in f.adj:
                    if a is not None:
                        out.update(self.faces[a[0]].pid)
        return out - {-1}

    def merge_loop(self, thr, keep):
        self.thr = thr
        todo = set(range(self.max_path)) - keep
        while todo:
            for p in range(self.max_path - 1, -1, -1):
                if p in todo:
                    todo.discard(p)
                    if self.remove_if_possible(p):
                        todo |= self._changed_paths() - keep

    def remaining_paths(self):
        return {q for f in self.faces if f is not None for q in f.pid if q >= 0}
