"""QuadWild's patch tracer (xfield_tracer PatchTracer) on a FieldGraph: field-aligned traces are added greedily until
every patch is a disk with 3-5 corners, Catmull-Clark-able sides and a corner count matching its singularity."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import comfy_kitchen as ck
import torch

from .metamesh import MetaMesh, catmull_clarkable
from .field import RowSum, opposite_halfedges, surface_tips
from .trace_graph import DRIFT, FieldGraph, Queries, build_graph, link_graph

INTERNAL, FLAT, CONVEX, CONCAVE = 0, 1, 2, 3
NONE = -1
DIRECT, DIJKSTRA, LOOP = 0, 1, 2
OK, NON_DISK, HAS_EMITTER, LOW_CORNERS, HIGH_CORNERS, NOT_CC, NO_MATCH = range(7)
SAMPLE_RATIO, MIN_SAMPLES, MAX_SAMPLES = 0.01, 50, 1000
MAX_NARROW_CONST = 0.05
TRACE_STEPS = 7                                                        # progress steps of trace_patches
TIP_REACH = 4                                                          # tip ring search radius, in average edges


@dataclass
class Trace:
    nodes: list
    loop: bool
    unremovable: bool = False


class Tracer:
    def __init__(self, g: FieldGraph, pd1f: torch.Tensor, comp_v: np.ndarray):
        self.g, self.Q, self.pd1f = g, Queries(g), pd1f
        self.dev = g.dst.device
        self.P, self.F = g.P.cpu().numpy(), g.F.cpu().numpy()
        self.nv, self.nf = len(self.P), len(self.F)
        self.n = 4 * self.nv
        self.comp_v, self.comp_f = comp_v, comp_v[self.F[:, 0]]
        self.ncomp = int(comp_v.max()) + 1
        self.dirs, self.vn, self.valence = g.dirs.cpu().numpy(), g.vn.cpu().numpy(), g.valence.cpu().numpy()
        self.border_v, self.singular = g.border.cpu().numpy(), g.singular.cpu().numpy()

        F, nv = self.F, self.nv
        self.tail, self.head = F.reshape(-1), np.roll(F, -1, 1).reshape(-1)
        self._opp_t = opposite_halfedges(g.F)
        self.opp = self._opp_t.cpu().numpy()
        dkey = self.tail * nv + self.head
        self._dorder = np.argsort(dkey)
        self._dkey = dkey[self._dorder]
        uk, self.eid = np.unique(np.minimum(self.tail, self.head) * nv + np.maximum(self.tail, self.head), return_inverse=True)
        self.ne = len(uk)                                                   # eid: undirected edge id per half-edge
        self.wedge = self._wedge_angles(self.P)
        self.avg_edge = float(np.linalg.norm(self.P[self.head] - self.P[self.tail], axis=1).mean())
        area = 0.5 * np.linalg.norm(np.cross(self.P[F[:, 1]] - self.P[F[:, 0]], self.P[F[:, 2]] - self.P[F[:, 0]]), axis=1)
        self.max_weight = np.sqrt(np.bincount(self.comp_f, weights=area, minlength=self.ncomp)) * MAX_NARROW_CONST * DRIFT   # MaxNarrowWeight per component

        self._eid = self._tensor(self.eid)
        self.full_traced = np.zeros(self.nf, dtype=bool)                   # sub-tracing found nothing in these faces
        # never reset, on purpose: like QuadWild's NodeDist, unreached nodes keep values from earlier searches
        self.node_dist = torch.zeros(self.n, dtype=torch.float64, device=self.dev)
        self.active, self.all_receivers = np.ones(self.n, dtype=bool), False
        self._spoke_y = np.cross(self.vn, self.dirs[:, 0])                  # tangent frame (dirs[:, 0], this)
        self.set_paths([])

    # ------------------------------------------------------------------ mesh helpers
    def he_of(self, a, b):
        """Half-edge a -> b, or -1 if there is none."""
        key = np.asarray(a, dtype=np.int64) * self.nv + np.asarray(b)
        i = np.searchsorted(self._dkey, key).clip(max=len(self._dkey) - 1)
        return np.where(self._dkey[i] == key, self._dorder[i], -1)

    def closest_dir(self, v, d):
        """Index of the vertex direction closest to d, ties to the last (GetClosestDirTo)."""
        return 3 - np.argmax(np.einsum("nkc,nc->nk", self.dirs[v], d)[:, ::-1], 1)

    def dir_at(self, h, v):
        return np.where(self.tail[h] == v, self.tdir[h], self.hdir[h])

    def _tensor(self, x):
        return torch.as_tensor(x, device=self.dev)

    @staticmethod
    def _links(lens, loop):
        """Trace index, previous and next position of each node (-1 past an open trace's ends, loops wrap around)."""
        seg = np.repeat(np.arange(len(lens)), lens)
        first = np.cumsum(lens) - lens
        prv, nxt = np.arange(-1, len(seg) - 1), np.arange(1, len(seg) + 1)
        prv[first] = np.where(loop, first + lens - 1, -1)
        nxt[first + lens - 1] = np.where(loop, first, -1)
        return seg, prv, nxt

    @staticmethod
    def _ranges(start, count):
        """Concatenated index ranges start[i] .. start[i] + count[i] - 1."""
        return np.repeat(start - np.cumsum(count) + count, count) + np.arange(count.sum())

    # ------------------------------------------------------------------ init
    def border_table(self):
        """Set border half-edge field directions and lengths (EdgeDirectionTable::AddBorder, InitEdgeL)."""
        bh = np.nonzero(self.opp < 0)[0]
        a, b = self.tail[bh], self.head[bh]
        u = self.P[b] - self.P[a]
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
        self.tdir[bh], self.hdir[bh] = self.closest_dir(a, u), self.closest_dir(b, -u)
        self.elen[bh] = np.linalg.norm(self.P[b] - self.P[a], axis=1)
        return bh, a, b, u

    def border_frames(self):
        """Per border edge end, sorted by vertex: the vertex, the edge's inward normal rotated into the vertex frame and
        the border direction continued past it (GetOrthoFlatDirections). Also the inward normals summed per vertex."""
        bh, a, b, u = self.border_table()
        fn = np.cross(self.P[self.F[:, 1]] - self.P[self.F[:, 0]], self.P[self.F[:, 2]] - self.P[self.F[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
        f = np.repeat(fn[bh // 3], 2, 0)
        o = np.cross(f, np.repeat(u, 2, 0))
        verts = np.stack([a, b], 1).reshape(-1)
        c = (f * self.vn[verts]).sum(1, keepdims=True)
        k = np.cross(f, self.vn[verts])
        k /= np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-300)
        phi = np.arccos(np.clip(c, -1, 1))
        r = np.where(c > 1 - 1e-9, o, o * np.cos(phi) + np.cross(k, o) * np.sin(phi) + k * (k * o).sum(1, keepdims=True) * (1 - np.cos(phi)))
        ortho = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-300)
        order = np.argsort(verts, kind="stable")
        osum = np.zeros((self.nv, 3))
        np.add.at(osum, verts[order], ortho[order])
        return verts[order], ortho[order], np.stack([-u, u], 1).reshape(-1, 3)[order], osum

    def init_types(self):
        """Set vertex classes (vertex_classifier.h), emitter and receiver nodes (vertex_emitter.h) and trace needs."""
        bv, ortho, cont, osum = self.border_frames()
        b = self.opp < 0
        cnt = np.zeros((self.nv, 4), dtype=np.int64)
        np.add.at(cnt, (np.concatenate([self.tail[b], self.head[b]]), np.concatenate([self.tdir[b], self.hdir[b]])), 1)
        possible = ((cnt[:, 0::2].sum(1) > 0) & (cnt[:, 1::2].sum(1) > 0)) | (cnt > 1).any(1)
        vert_a = np.bincount(self.F.reshape(-1), weights=self.wedge.reshape(-1), minlength=self.nv)
        self.vtype = np.where(self.border_v, FLAT, INTERNAL)
        self.vtype[possible & (vert_a < np.pi)] = CONVEX
        self.vtype[possible & (vert_a > np.pi)] = CONCAVE
        self.emitter, self.receiver = np.full(self.n, NONE, dtype=np.int64), np.full(self.n, NONE, dtype=np.int64)
        fv = np.nonzero(self.vtype == FLAT)[0]
        e = self.closest_dir(fv, osum[fv] / np.maximum(np.linalg.norm(osum[fv], axis=1, keepdims=True), 1e-300))
        self.emitter[4 * fv + e], self.receiver[4 * fv + (e + 2) % 4] = FLAT, FLAT
        cv = np.nonzero(self.vtype == CONCAVE)[0]
        i = np.searchsorted(bv, cv)
        e0, e1 = self.closest_dir(cv, ortho[i]), self.closest_dir(cv, ortho[i + 1])
        # with opposite inward normals, also emit along the border continued past the vertex
        c = cont[i] + cont[i + 1]
        e2 = np.where(e1 == (e0 + 2) % 4, self.closest_dir(cv, c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-300)), e0)
        for e in (e0, e1, e2):
            self.emitter[4 * cv + e], self.receiver[4 * cv + (e + 2) % 4] = CONCAVE, CONCAVE
        self.needs = np.where(self.vtype == CONCAVE, np.maximum((self.emitter.reshape(-1, 4) != NONE).sum(1) - 1, 1), 0)

    def _wedge_angles(self, P):
        T = P[self.F]
        u, w = np.roll(T, -1, 1) - T, np.roll(T, -2, 1) - T
        cos = (u * w).sum(2) / np.maximum(np.linalg.norm(u, axis=2) * np.linalg.norm(w, axis=2), 1e-300)
        return np.arccos(np.clip(cos, -1, 1))

    def sample_loop_emitters(self):
        """Mark PD1 and PD2 loop emitters at internal farthest-point samples of each component (SampleLoopEmitters)."""
        dev = self.dev
        P, comp = self.g.P, self._tensor(self.comp_v)
        size = np.bincount(self.comp_v, minlength=self.ncomp)
        want = self._tensor(np.minimum(np.clip((size * SAMPLE_RATIO).astype(np.int64), MIN_SAMPLES, MAX_SAMPLES), size))
        ids = torch.arange(self.nv, device=dev)
        far = torch.full((self.nv,), torch.inf, dtype=torch.float64, device=dev)
        picked = torch.zeros(self.nv, dtype=torch.bool, device=dev)
        cur = torch.full((self.ncomp,), self.nv, dtype=torch.long, device=dev).scatter_reduce(0, comp, ids, "amin")
        for step in range(int(want.max())):
            take = want > step
            mine = take[comp]
            picked[cur[take]] = True
            far = torch.where(mine, torch.minimum(far, (P - P[cur[comp]]).norm(dim=1)), far)
            key = torch.where(picked | ~mine, -1.0, far)
            best = torch.full((self.ncomp,), -1.0, dtype=torch.float64, device=dev).scatter_reduce(0, comp, key, "amax")
            pick = torch.where((key == best[comp]) & (key >= 0), ids, -1)
            nxt = torch.full((self.ncomp,), -1, dtype=torch.long, device=dev).scatter_reduce(0, comp, pick, "amax")
            cur = torch.where(nxt >= 0, nxt, cur)
        v = np.nonzero(picked.cpu().numpy() & (self.vtype == INTERNAL))[0]
        self.emitter[4 * v], self.emitter[4 * v + 1] = INTERNAL, INTERNAL

    # ------------------------------------------------------------------ partitions
    def update_partitions(self):
        """Recompute patches touched by new traces or needs, with their corners and types (LazyUpdatePartitions)."""
        if self._full:
            fi, aff = np.arange(self.nf), []
            self.label, self.part_faces, self.npart = np.zeros(self.nf, dtype=np.int64), {}, 0
            self.ptype = self.corner_key = np.zeros(0, dtype=np.int64)
        else:
            faces = np.concatenate([np.zeros(0, dtype=np.int64)] + self._dirty)
            if not len(faces):
                return
            aff = np.unique(self.label[faces])
            fi = np.sort(np.concatenate([self.part_faces.pop(int(p)) for p in aff]))
        self._full, self._dirty = False, []
        ncomp, comp, order, hs, pv, pe = self._split(fi)
        new = self.npart + np.arange(ncomp)
        self.label[fi] = self.npart + comp
        self.part_faces.update(zip(new.tolist(), [fi[o] for o in np.split(order, np.cumsum(np.bincount(comp, minlength=ncomp))[:-1])]))
        self.npart += ncomp
        # new patches have the largest ids, so appending their keys keeps corner_key sorted
        self.corner_key = np.concatenate([self.corner_key[~np.isin(self.corner_key // self.nv, aff)], self._find_corners(fi, hs)])
        self.ncorners = np.bincount(self.corner_key // self.nv, minlength=self.npart)
        self.ptype = np.concatenate([self.ptype, self._patch_types(hs, new, comp, pv, pe)])

    def _split(self, fi):
        """Components of faces fi (ascending) across untraced interior half-edges: count, per-face component, fi's order
        by component, fi's traced or border half-edges, and component * nv + vertex and component * ne + edge keys."""
        dev = self.dev
        fs = self._tensor(fi)
        hl = (3 * fs[:, None] + torch.arange(3, device=dev)).reshape(-1)
        sel = self._sel_t[hl]
        inner = hl[~sel & (self._opp_t[hl] >= 0)]
        local = torch.empty(self.nf, dtype=torch.long, device=dev)
        local[fs] = torch.arange(len(fi), device=dev)
        root = ck.connected_components(torch.stack([local[inner // 3], local[self._opp_t[inner] // 3]], 1), len(fi))
        first, comp = torch.unique(root, return_inverse=True)
        c3 = comp.repeat_interleave(3)
        parts = [comp, torch.argsort(comp, stable=True), hl[sel], torch.unique(c3 * self.nv + self.g.F[fs].reshape(-1)),
                 torch.unique(c3 * self.ne + self._eid.view(-1, 3)[fs].reshape(-1))]
        return (len(first),) + tuple(np.split(torch.cat(parts).cpu().numpy(), np.cumsum([len(x) for x in parts])[:-1]))

    def canonical(self):
        """Update the partitions and renumber the patches by smallest face, as UpdatePartitionsFromChoosen does."""
        self.update_partitions()
        live = np.array(sorted(self.part_faces, key=lambda p: self.part_faces[p][0]), dtype=np.int64)
        id_map = np.full(self.npart, -1, dtype=np.int64)
        id_map[live] = np.arange(len(live))
        self.label = id_map[self.label]
        self.part_faces = {int(id_map[p]): f for p, f in self.part_faces.items()}
        self.ptype, self.ncorners = self.ptype[live], self.ncorners[live]
        self.corner_key = np.sort(id_map[self.corner_key // self.nv] * self.nv + self.corner_key % self.nv)
        self.npart = len(live)

    def _find_corners(self, fi, hs):
        """Sorted corner keys (patch * nv + vertex) of the patches of faces fi: convex vertices, and vertices where two
        border edges leave along orthogonal or equal field directions (concave ones only at patch angles below pi)."""
        lab, nv = self.label, self.nv
        p = lab[hs // 3]
        _, first = np.unique(p * self.ne + self.eid[hs], return_index=True)
        hs, p = hs[first], p[first]
        a, b = self.tail[hs], self.head[hs]
        pv = np.concatenate([p * nv + a, p * nv + b])
        keys, inv = np.unique(pv, return_inverse=True)
        cnt = np.zeros((len(keys), 4), dtype=np.int64)
        np.add.at(cnt, (inv, np.concatenate([self.dir_at(hs, a), self.dir_at(hs, b)])), 1)
        corner = ((cnt[:, 0::2].sum(1) > 0) & (cnt[:, 1::2].sum(1) > 0)) | (cnt > 1).any(1)
        ck = np.unique(np.concatenate([keys[corner], pv[self.vtype[pv % nv] == CONVEX]]))
        concave = self.vtype[ck % nv] == CONCAVE
        if concave.any():
            keys_a, inv_a = np.unique(np.repeat(lab[fi], 3) * nv + self.F[fi].reshape(-1), return_inverse=True)
            sums = np.bincount(inv_a, weights=self.wedge[fi].reshape(-1))
            pos = np.searchsorted(keys_a, ck).clip(max=len(keys_a) - 1)
            ck = ck[~(concave & (sums[pos] > np.pi))]
        return ck

    def _patch_types(self, hs, parts, comp, pv, pe):
        """Validity type of each new patch in parts (PatchInfo + UpdatePartitionType)."""
        nv, base, m = self.nv, parts[0], len(parts)
        pvp, pvv = pv // nv, pv % nv
        need = np.bincount(pvp, weights=self.needs[pvv], minlength=m)
        euler = np.bincount(pvp, minlength=m) + np.bincount(comp, minlength=m) - np.bincount(pe // self.ne, minlength=m)
        nc = self.ncorners[parts]
        sk = pv[self.valence[pvv] != 4]
        skp = sk // nv
        nsing = np.bincount(skp, minlength=m)
        expected = np.where(nsing == 0, 4, -1)
        expected[skp] = np.where(nsing[skp] == 1, np.where(self._is_corner(sk + base * nv), nc[skp], self.valence[sk % nv]), -1)
        cc_ok = (nc >= 3) & (nc <= 6) & (euler == 1) & (need == 0)
        test = np.nonzero(cc_ok & ((nc == 3) | (nc == 5) | (nc == 6)))[0]
        for i, L in zip(test, self.side_lengths(parts[test], hs)):
            cc_ok[i] = catmull_clarkable(L, self.avg_edge)
        return np.select([euler != 1, need > 0, nc < 3, nc > 5, ~cc_ok, expected != nc],
                         [NON_DISK, HAS_EMITTER, LOW_CORNERS, HIGH_CORNERS, NOT_CC, NO_MATCH], OK)

    def _boundary_steps(self, hl):
        """Patch boundary half-edges among hl and the next one of each, patch on the right (vcg Pos::NextNotFaux)."""
        lab, opp = self.label, self.opp
        crossing = lambda x: (opp[x] < 0) | (lab[opp[x] // 3] != lab[x // 3])
        h = hl[crossing(hl)]
        e = 3 * (h // 3) + (h % 3 + 2) % 3
        todo = np.ones(len(h), dtype=bool)
        for _ in range(64):
            todo &= ~crossing(e)
            if not todo.any():
                break
            o = opp[e[todo]]
            e[todo] = 3 * (o // 3) + (o % 3 + 2) % 3
        return h, e

    def boundary_next(self):
        """Boundary mask and next boundary half-edge (-1 off the boundary) for every half-edge."""
        h, e = self._boundary_steps(np.arange(3 * self.nf))
        nxt = np.full(3 * self.nf, -1, dtype=np.int64)
        nxt[h] = e
        return nxt >= 0, nxt

    def _is_corner(self, key):
        """Mask of the patch * nv + vertex keys that are corners (corner_key is kept sorted)."""
        return np.append(self.corner_key, -1)[np.searchsorted(self.corner_key, key)] == key

    def side_lengths(self, parts, hl):
        """Yield each listed patch's side lengths (summed EdgeL) between its corners, or [] if it has none."""
        h, e = self._boundary_steps(hl)
        at_corner = self._is_corner(self.label[h // 3] * self.nv + self.head[h])
        part, first = np.unique(self.label[h[at_corner] // 3], return_index=True)
        start = dict(zip(part.tolist(), np.nonzero(at_corner)[0][first].tolist()))
        succ = np.searchsorted(h, e).clip(max=len(h) - 1)
        succ = np.where(h[succ] == e, succ, -1).tolist()
        el, ac = self.elen[h].tolist(), at_corner.tolist()
        for p in parts.tolist():
            i0 = i = start.get(p)
            if i0 is None:
                yield []
                continue
            L, cur = [], 0.0
            for _ in range(len(h)):                     # a non-manifold vertex can trap the walk in a cycle
                cur += el[i]
                i = succ[i]
                if i == i0 or i < 0:
                    break
                if ac[i]:
                    L.append(cur)
                    cur = 0.0
            yield L + [cur]

    # ------------------------------------------------------------------ traces
    def add_traces(self, traces):
        """Add traces in order (AddChoosen), updating needs, edge tables, selection and collision sets."""
        if not traces:
            return
        self.paths += traces
        nodes = np.concatenate([np.asarray(t.nodes) for t in traces])
        v = nodes // 4
        uv, cnt = np.unique(v, return_counts=True)
        dec = np.minimum(self.needs[uv], cnt)
        self.needs[uv] -= dec
        if dec.any():
            self._dirty.append(np.nonzero(np.isin(self.F, uv[dec > 0]).any(1))[0])
        _, prv, nxt = self._links(np.array([len(t.nodes) for t in traces]), np.array([t.loop for t in traces]))
        a = np.nonzero(nxt >= 0)[0]
        a, b = nodes[a], nodes[nxt[a]]
        va, vb = a // 4, b // 4
        dsum = self.dirs[va, a % 4] + self.dirs[vb, b % 4]
        dsum /= np.maximum(np.linalg.norm(dsum, axis=1, keepdims=True), 1e-300)
        fl = ((self.P[vb] - self.P[va]) * dsum).sum(1)
        fwd = self.he_of(va, vb)
        for h, x, y in ((fwd, a % 4, (b % 4 + 2) % 4), (self.he_of(vb, va), (b % 4 + 2) % 4, a % 4)):
            m = h >= 0
            self.tdir[h[m]], self.hdir[h[m]], self.elen[h[m]] = x[m], y[m], fl[m]
            self._sel_t[self._tensor(h[m])] = True
            self._dirty.append(h[m] // 3)
        self._chosen_node[nodes] = True
        self._chosen_edge[self.eid[fwd[fwd >= 0]]] = True
        inner = (prv >= 0) & (nxt >= 0)
        self._chosen_tang[nodes[inner] ^ 2] = True                         # the reverse direction at inner nodes
        # prepend the (previous, next) vertex of inner nodes off the border to their vertex's turn list
        at = np.nonzero(inner & ~self.border_v[v])[0]
        n0, k = self._nturn, len(at)
        if n0 + k > len(self._turn_link):
            self._turn_link = np.resize(self._turn_link, (max(n0 + k, 2 * len(self._turn_link)), 3))
        o = np.argsort(v[at], kind="stable")
        xs, ids = v[at][o], n0 + o
        last = np.diff(xs, append=-1) != 0
        self._turn_link[ids, 0] = np.where(last, self._turn_head[xs], np.roll(ids, -1))
        self._turn_link[n0:n0 + k, 1:] = np.stack([v[prv[at]], v[nxt[at]]], 1)
        lead = np.diff(xs, prepend=-1) != 0
        self._turn_head[xs[lead]] = ids[lead]
        self._nturn += k

    def _drop_self_intersecting(self, paths, loop):
        """Paths with None for those using a (vertex, axis) pair or a mesh edge twice (SelfIntersect)."""
        idx = [i for i, p in enumerate(paths) if p is not None]
        if not idx:
            return list(paths)
        allv = np.concatenate([paths[i] for i in idx])
        seg, _, nxt = self._links(np.array([len(paths[i]) for i in idx]), loop)
        v = allv // 4
        dup = np.zeros(len(idx), dtype=bool)
        key = np.sort(seg * (2 * self.nv) + 2 * v + allv % 2)
        dup[key[1:][key[1:] == key[:-1]] // (2 * self.nv)] = True
        a, b = v[nxt >= 0], v[nxt[nxt >= 0]]
        key = np.sort(seg[nxt >= 0] * (self.nv * self.nv) + np.minimum(a, b) * self.nv + np.maximum(a, b))
        dup[key[1:][key[1:] == key[:-1]] // (self.nv * self.nv)] = True
        dropped = set(np.array(idx)[dup].tolist())
        return [None if i in dropped else p for i, p in enumerate(paths)]

    def collisions(self, nodes, lens, loop):
        """Mask of the traces colliding with a chosen trace (CollideTraces): sharing a mesh edge, running along one
        through a shared vertex, or touching one without crossing it."""
        seg, prv, nxt = self._links(lens, loop)
        out = np.zeros(len(lens), dtype=bool)
        out[seg[(((nxt >= 0) | (prv < 0)) & self._chosen_node[nodes]) | (((prv >= 0) | (nxt < 0)) & self._chosen_node[nodes ^ 2])]] = True
        v = nodes // 4
        a = np.nonzero(nxt >= 0)[0]
        h = self.he_of(v[a], v[nxt[a]])
        out[seg[a][(h >= 0) & self._chosen_edge[self.eid[h]]]] = True
        t = np.nonzero((prv >= 0) & (nxt >= 0) & (self._turn_head[v] >= 0) & ~out[seg])[0]
        at, turn = [t], [self._turn_head[v[t]]]
        while len(turn[-1]):                                                # every chosen turn at those vertices
            cur = self._turn_link[turn[-1], 0]
            at.append(at[-1][cur >= 0])
            turn.append(cur[cur >= 0])
        t, turn = np.concatenate(at), np.concatenate(turn)
        s = v[t]
        d = self.P[np.stack([v[prv[t]], v[nxt[t]], self._turn_link[turn, 1], self._turn_link[turn, 2]], 1)] - self.P[s][:, None]
        ang = np.arctan2(d @ self._spoke_y[s][..., None], d @ self.dirs[s, 0][..., None])[..., 0]
        lab = np.argsort(ang, axis=1) >= 2
        out[seg[t[(lab[:, 1:] == lab[:, :-1]).any(1)]]] = True
        return out

    def collides(self, nodes, loop):
        return bool(self.collisions(np.asarray(nodes), np.array([len(nodes)]), loop)[0])

    def splits_open_patches(self, nodes, lens, loop):
        """Mask of the traces that run through the inside of a patch not valid yet (SplitSomeNonOKPartition)."""
        seg, _, nxt = self._links(lens, loop)
        a = np.nonzero(nxt >= 0)[0]
        h = self.he_of(nodes[a] // 4, nodes[nxt[a]] // 4)
        o, f0 = self.opp[h], self.label[h // 3]
        out = np.zeros(len(lens), dtype=bool)
        out[seg[a][(h >= 0) & (o >= 0) & (f0 == self.label[o // 3]) & (self.ptype[f0] != OK)]] = True
        return out

    # ------------------------------------------------------------------ tracing configurations
    def configuration(self, frm, to):
        """Node masks (can emit, can receive, disabled) for tracing from frm to to (GetTracingConfiguration)."""
        vt = np.repeat(self.vtype, 4)
        tangent = (vt == CONVEX) | ((vt == FLAT) & (self.emitter == NONE) & (self.receiver == NONE))
        removed = (vt == FLAT) & np.repeat(self.singular, 4)
        off = self._chosen_node | self._chosen_tang
        if not (self.all_receivers and to == FLAT):
            off |= tangent
        if frm != CONCAVE:
            off |= vt == CONCAVE
            return (self.emitter == frm) & ~off, (self.receiver == FLAT) & ~off, off | removed
        unsat = np.repeat(self.needs > 0, 4)
        emit = (self.emitter == CONCAVE) & unsat & ~off
        if to == CONCAVE:
            return emit, (self.receiver == CONCAVE) & unsat & ~off, off | removed
        recv = (self.receiver == FLAT) & ~off
        if self.all_receivers:
            recv |= tangent
        return emit, recv, off | (self.receiver == CONCAVE) & ~emit | removed

    # ------------------------------------------------------------------ candidates
    def _trace(self, starts, method, recv, loop_reach=None):
        """Expanded candidate paths from the start nodes, None where the search fails or the path self-intersects."""
        act, s = self._tensor(self.active), self._tensor(starts)
        if method == LOOP:
            paths = self.Q.loops(s, act, loop_reach)
        elif method == DIJKSTRA:
            paths = self.Q.first_hit(s, act, self._tensor(recv), max_weight=self._tensor(self.max_weight[self.comp_v[starts // 4]]))
        else:
            paths = self.Q.walk(s, act, self._tensor(recv))
        return self._expand(self._drop_self_intersecting(paths, method == LOOP), method == LOOP, act)

    def _expand(self, paths, loop, active):
        """Expand every link into the shortest mesh-edge path between its nodes (ExpandPath), updating node_dist."""
        live = np.array([i for i, p in enumerate(paths) if p is not None], dtype=np.int64)
        if not len(live):
            return list(paths)
        lens = np.array([len(paths[i]) for i in live])
        nodes = np.concatenate([paths[i] for i in live])
        seg, _, nxt = self._links(lens, loop)
        a = np.nonzero(nxt >= 0)[0]
        a, b, seg = nodes[a], nodes[nxt[a]], seg[a]
        _, first, hop = np.unique(a * self.n + b, return_index=True, return_inverse=True)   # one search per distinct link
        (flat, ptr), (wq, wv, wd) = self.Q.between(self._tensor(a[first]), self._tensor(b[first]), active)
        size = np.diff(ptr)
        # every path runs its links in order up to its first failed one
        pos = np.arange(len(a)) - np.searchsorted(seg, seg)
        stop = np.full(len(lens), len(a))
        failed = size[hop] == 0
        np.minimum.at(stop, seg[failed], pos[failed])
        ran = pos <= stop[seg]
        last = np.full(len(first), -1, dtype=np.int64)
        np.maximum.at(last, hop[ran], np.cumsum(ran)[ran] - 1)
        ok = stop == len(a)
        use = ok[seg]
        n = size[hop[use]] - 1                                              # each link's path without its end node
        tail = np.nonzero(ok & (not loop))[0]
        body = np.concatenate([flat[self._ranges(ptr[hop[use]], n)], nodes[(np.cumsum(lens) - 1)[tail]]])
        owner = np.concatenate([np.repeat(seg[use], n), tail])
        body = body[np.argsort(owner, kind="stable")]
        out = [None] * len(paths)
        for i, piece in zip(live[ok], np.split(body, np.cumsum(np.bincount(owner, minlength=len(lens))[ok])[:-1])):
            out[i] = piece.tolist()
        # last writer wins, as in QuadWild's shared NodeDist
        when = self._tensor(last)[wq]
        at = torch.full((self.n,), -1, dtype=torch.long, device=self.dev).scatter_reduce_(0, wv, when, "amax")
        win = (when >= 0) & (when == at[wv])
        self.node_dist[wv[win]] = wd[win]
        return self._drop_self_intersecting(out, loop)

    def join(self, frm, to, method):
        """Trace from all allowed emitters and add the chosen traces (JoinConnection). Returns whether any was added."""
        emit, recv, off = self.configuration(frm, to)
        self.active = ~off
        starts = np.nonzero(emit)[0]
        if not len(starts):
            return False
        cands = [p for p in self._trace(starts, method, recv) if p is not None]
        n0 = len(self.paths)
        if frm == CONCAVE:
            length = lambda a: np.abs(((self.P[a[:-1] // 4] - self.P[a[1:] // 4]) * self.dirs[a[:-1] // 4, a[:-1] % 4]).sum(1)).sum()
            for i in np.argsort(np.array([length(np.asarray(p)) for p in cands], dtype=np.float32), kind="stable"):
                p = cands[i]
                if (self.needs[p[0] // 4] or self.needs[p[-1] // 4]) and not self.collides(p, False):
                    self.add_traces([Trace(p, False)])
        else:
            self._choose_by_distance(cands, method == LOOP)
        return len(self.paths) > n0

    def trace_tip_rings(self):
        """Add two short loops winding once around each spike point (surface_tips), kept through removal. Sampled
        loops alone often just pass by a point, leaving sliver strips or a cap of large quads on the tip."""
        tips = surface_tips(self.g.P, self.g.F).cpu().numpy()
        tips = tips[np.linalg.norm(self.vn[tips], axis=1) > 0]            # a point on zero-area faces has no normal
        if not len(tips):
            return
        self.active = ~self.configuration(INTERNAL, INTERNAL)[2]
        reach = TIP_REACH * self.avg_edge
        near = [np.nonzero((np.linalg.norm(self.P - self.P[t], axis=1) < reach) & (np.arange(self.nv) != t))[0] for t in tips.tolist()]
        owner = np.repeat(tips, [2 * len(x) for x in near]).tolist()
        # a ring within 2 * reach of the tip is at most 4 pi reach long, times 4 for the drift penalty
        paths = self._trace(np.concatenate([np.concatenate([4 * x, 4 * x + 1]) for x in near]), LOOP, None, 16 * np.pi * reach)
        cands = {}
        for t, nodes in zip(owner, paths):
            if nodes is None:
                continue
            vs = np.asarray(nodes) // 4
            rel = self.P[vs] - self.P[t]
            if t in vs or np.linalg.norm(rel, axis=1).max() > 2 * reach:
                continue
            n = self.vn[t] / np.linalg.norm(self.vn[t])
            x = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
            ang = np.arctan2(rel @ np.cross(n, x), rel @ x)
            turn = np.angle(np.exp(1j * (np.roll(ang, -1) - ang))).sum() / (2 * np.pi)
            if abs(round(turn)) == 1:
                cands.setdefault(t, []).append((np.linalg.norm(rel - np.roll(rel, -1, 0), axis=1).sum(), nodes))
        for t in sorted(cands):
            added = 0
            for _, nodes in sorted(cands[t], key=lambda c: c[0]):
                if added < 2 and not self.collides(nodes, True):
                    self.add_traces([Trace(nodes, True, True)])
                    added += 1

    def _choose_by_distance(self, paths, loop):
        """Greedily add the farthest candidates (mean node_dist) splitting an invalid patch, one per component per round
        (ChooseGreedyByDistance). Collisions need a shared vertex: only candidates touching new traces are rechecked."""
        if not paths:
            return
        self.update_partitions()
        # InitNodeDistances from disabled nodes, chosen traces and singular vertices (unreached nodes keep their value)
        src = ~self.active | np.repeat(self.singular, 4) | self._chosen_node | self._chosen_node[np.arange(self.n) ^ 2]
        fresh = self.Q.distances(self._tensor(np.nonzero(src)[0]), torch.ones(self.n, dtype=torch.bool, device=self.dev))
        self.node_dist = torch.where(torch.isfinite(fresh), fresh, self.node_dist)
        lens = np.array([len(p) for p in paths])
        nodes = np.concatenate(paths)
        seg = np.repeat(np.arange(len(paths)), lens)
        first = np.cumsum(lens) - lens

        def gather(idx):
            return nodes[self._ranges(first[idx], lens[idx])], lens[idx], loop

        comp = self.comp_v[nodes[first] // 4]
        vc = np.unique(nodes // 4 * len(paths) + seg)                       # (vertex, candidate) pairs, by vertex
        vptr = np.searchsorted(vc // len(paths), np.arange(self.nv + 1))
        queued = ~self.collisions(nodes, lens, loop)
        pos = np.arange(len(paths))
        by_cand = RowSum(self._tensor(seg), len(paths))
        flat, lens_t, act = self._tensor(nodes), self._tensor(lens).double(), self._tensor(self.active)
        while queued.any():
            pri = (by_cand(self.node_dist[flat]) / lens_t).float().cpu().numpy()
            idx = np.nonzero(queued)[0]
            order = idx[np.lexsort((pos[idx], -pri[idx], comp[idx]))]
            pos[order] = np.arange(len(order))
            at, picked = np.nonzero(np.diff(comp[order], prepend=-1))[0], []
            last = np.diff(comp[order], append=-1) != 0
            while len(at):                                                  # next candidate of each searching component
                c = order[at]
                ok = self.splits_open_patches(*gather(c))
                queued[c[~ok]] = False
                picked.append(c[ok])
                at = at[~ok & ~last[at]] + 1
            picked = np.concatenate(picked)
            if not len(picked):
                break
            queued[picked] = False
            self.add_traces([Trace(paths[i], loop) for i in picked])     # one per component, so no shared vertices
            new = gather(picked)[0]
            new_src = np.unique(np.concatenate([new, new ^ 2]))
            self.node_dist = self.Q.distances(self._tensor(new_src[self.active[new_src]]), act, self.node_dist)
            self.update_partitions()
            u = np.unique(new // 4)
            near = np.unique(vc[self._ranges(vptr[u], vptr[u + 1] - vptr[u])] % len(paths))
            near = near[queued[near]]
            queued[near[self.collisions(*gather(near))]] = False

    # ------------------------------------------------------------------ driver
    def join_concave(self):
        while self.join(CONCAVE, CONCAVE, DIJKSTRA) | self.join(CONCAVE, FLAT, DIRECT) | self.join(CONCAVE, FLAT, DIJKSTRA):
            pass

    def batch_add_loops(self, force_receivers=False, force_always=False):
        """Add concave joins, loops, then border-to-border traces (BatchAddLoops with AddOnlyNeeded)."""
        if force_receivers:
            self.all_receivers = True
            self.max_weight /= 100
            self.join_concave()
            if not force_always:
                self.max_weight *= 100
                self.all_receivers = False
        self.join_concave()
        self.join(INTERNAL, INTERNAL, LOOP)
        while self.join(FLAT, FLAT, DIRECT) | self.join(FLAT, FLAT, DIJKSTRA):
            pass

    # ------------------------------------------------------------------ sub-patches
    def set_paths(self, paths):
        """Replace the chosen traces and rebuild the tables derived from them (SetChoosenFromVertDir)."""
        self.paths, self._full, self._dirty, self._nturn = [], True, [], 0
        # field direction along each half-edge at its tail, and back along it at its head
        self.tdir, self.hdir = np.full(3 * self.nf, -1, dtype=np.int64), np.full(3 * self.nf, -1, dtype=np.int64)
        self.elen = np.zeros(3 * self.nf)
        self._sel_t = self._tensor(self.opp < 0)                            # half-edges on the border or a trace
        self._chosen_node, self._chosen_tang = np.zeros(self.n, dtype=bool), np.zeros(self.n, dtype=bool)
        self._chosen_edge = np.zeros(self.ne, dtype=bool)
        # per-vertex linked lists of turns: (next turn, previous vertex, next vertex) of each chosen trace through it
        self._turn_head = np.full(self.nv, -1, dtype=np.int64)
        self._turn_link = np.zeros((0, 3), dtype=np.int64)
        self.border_table()
        self.add_traces(list(paths))

    def sub_tracer(self, parts, resample):
        """Sub-tracer over the listed patches (TraceSubPatch for all at once), and how many traces were copied in."""
        rank = np.full(self.npart, -1, dtype=np.int64)
        rank[parts] = np.arange(len(parts))
        fc = rank[self.label]
        fsel = np.argsort(fc, kind="stable")[np.count_nonzero(fc < 0):]     # the faces by patch, ascending
        fc = fc[fsel]
        uk, inv = np.unique(np.repeat(fc, 3) * self.nv + self.F[fsel].reshape(-1), return_inverse=True)
        vmap = uk % self.nv
        local = lambda c, v: np.searchsorted(uk, c * self.nv + v)
        vm, pd1f = self._tensor(vmap), self.pd1f[self._tensor(fsel)]
        sub = Tracer(link_graph(self.g.P[vm], self._tensor(inv.reshape(-1, 3)), self.g.vn[vm], self.g.dirs[vm], pd1f, self.g.valence[vm]), pd1f, uk // self.nv)
        sub.vmap, sub.avg_edge = vmap, self.avg_edge
        sub.max_weight = self.max_weight[self.comp_f[[self.part_faces[p][0] for p in parts.tolist()]]]
        sub.Q.hop_reach = math.inf                    # a search that misses its end only floods its own patch
        sub.vtype, sub.needs = self.vtype[vmap], self.needs[vmap]
        sub.vtype[np.where(sub.vtype == CONCAVE, sub.needs == 0, sub.border_v)] = FLAT
        cp = rank[self.corner_key // self.nv]
        sub.vtype[local(cp[cp >= 0], self.corner_key[cp >= 0] % self.nv)] = CONVEX

        osum = sub.border_frames()[3]
        norm = np.linalg.norm(osum, axis=1, keepdims=True)
        dots = np.einsum("kdc,kc->kd", sub.dirs, osum / np.maximum(norm, 1e-300))
        # traced along d: emit along d + 1 or d + 3, whichever is nearer the inward normal, receive along the other
        g = dots > np.roll(dots, 2, 1)
        off, g2 = norm == 0, np.roll(g, 2, 1)
        traced = self._chosen_node.reshape(-1, 4)[vmap]
        prv, nxt = np.roll(traced, 1, 1), np.roll(traced, -1, 1)
        emit = ((self.emitter == FLAT) & self.active).reshape(-1, 4)[vmap] | prv & (off | g) | nxt & (off | ~g2)
        recv = ((self.receiver == FLAT) & self.active).reshape(-1, 4)[vmap] | nxt & (off | g2) | prv & (off | ~g)
        flat, cv = (sub.vtype == FLAT)[:, None], (sub.vtype == CONCAVE)[:, None]
        sub.receiver = np.where(cv, self.receiver.reshape(-1, 4)[vmap], np.where(flat & recv, FLAT, NONE)).reshape(-1)
        sub.emitter = np.where(cv, self.emitter.reshape(-1, 4)[vmap], np.where(flat & emit & ~recv, FLAT, NONE)).reshape(-1)

        # CopyPathsFrom: trace links inside a listed patch, as runs of local nodes (first link's start, then link ends)
        nodes = np.array([n for t in self.paths for n in t.nodes], dtype=np.int64)
        seg, _, nxt = self._links(np.array([len(t.nodes) for t in self.paths], dtype=np.int64), np.array([t.loop for t in self.paths], dtype=bool))
        a = np.nonzero(nxt >= 0)[0]
        a, b, seg = nodes[a], nodes[nxt[a]], seg[a]
        h = self.he_of(a // 4, b // 4)
        o = self.opp[h]
        c = np.where((h >= 0) & (o >= 0) & (self.label[h // 3] == self.label[o // 3]), rank[self.label[h // 3]], -1)
        k = np.nonzero(c >= 0)[0]
        new = (np.diff(k, prepend=-2) != 1) | (np.diff(seg[k], prepend=-1) != 0) | (np.diff(c[k], prepend=-2) != 0)
        x, y = 4 * local(c[k], a[k] // 4) + a[k] % 4, 4 * local(c[k], b[k] // 4) + b[k] % 4
        body = np.stack([np.where(new, x, -1), y], 1).reshape(-1)
        pieces = np.split(body[body >= 0], (np.arange(len(k)) + np.cumsum(new) - 1)[new])[1:]
        sub.add_traces([Trace(run.tolist(), False) for run in pieces])
        if resample:
            sub.sample_loop_emitters()
        return sub, len(pieces)

    def solve_sub_patches(self, only_non_disk, force_always):
        """Trace each invalid (or each non-disk) patch on its own until nothing changes (SolveSubPatches)."""
        while True:
            self.canonical()
            todo = np.nonzero(self.ptype == NON_DISK if only_non_disk else self.ptype != OK)[0]
            if not only_non_disk:
                todo = todo[np.isin(todo, self.label[~self.full_traced])]
            if not len(todo):
                return
            sub, copied = self.sub_tracer(todo, only_non_disk)
            sub.batch_add_loops(force_receivers=True, force_always=force_always)
            new = sorted(sub.paths[copied:], key=lambda t: sub.comp_v[t.nodes[0] // 4])
            self.full_traced[np.isin(self.label, np.delete(todo, [sub.comp_v[t.nodes[0] // 4] for t in new]))] = True
            if not new:
                return
            self.set_paths(self.paths + [Trace([4 * int(sub.vmap[n // 4]) + n % 4 for n in t.nodes], t.loop) for t in new])

    # ------------------------------------------------------------------ removal
    def fixed_corners(self):
        """Corner vertices per patch (FixAmmittableValences). Over 6 corners: drop the widest, non-feature ones first,
        down to 5. Under 3: add corners spaced by boundary length, since QuadWild's flattest vertices can leave one side
        longer than the other two together, which no quad pattern fills without stretching."""
        out = [c.tolist() for c in np.split(self.corner_key % self.nv, np.cumsum(self.ncorners)[:-1])]
        bnd, nxt = self.boundary_next()
        for p in [p for p, c in enumerate(out) if len(c) < 3 or len(c) > 6]:
            faces = self.part_faces[p]
            hs = (3 * faces[:, None] + np.arange(3)).reshape(-1)
            hs = hs[bnd[hs]]
            cs = set(out[p])
            if len(cs) < 3 and len(hs):                                       # a closed component has no boundary
                loop = [int(hs[0])]
                while nxt[loop[-1]] >= 0 and nxt[loop[-1]] != loop[0] and len(loop) <= len(hs):
                    loop.append(int(nxt[loop[-1]]))
                at = self.head[loop]                                         # boundary vertices in walking order
                arc = np.cumsum(self.elen[loop])
                total = arc[-1]
                k = [i for i, v in enumerate(at.tolist()) if v in cs]
                if len(k) == 2:
                    inner = (arc[k[1]] - arc[k[0]]) % total
                    start, span = (k[0], inner) if inner >= total - inner else (k[1], total - inner)
                    targets = [arc[start] + span / 2]
                else:
                    start = k[0] if k else len(at) - 1
                    targets = [arc[start] + total * j / 3 for j in range(len(k), 3)]           # from its corner, if any
                for t in targets:
                    d = np.abs((arc - t + total / 2) % total - total / 2)
                    d[np.isin(at, list(cs))] = np.inf
                    cs.add(int(at[np.argmin(d)]))
            elif len(cs) > 6:
                ang = np.bincount(self.F[faces].reshape(-1), weights=self.wedge[faces].reshape(-1), minlength=self.nv)
                bv = np.unique(np.concatenate([self.tail[hs], self.head[hs]]))
                for strict in (True, False):
                    for v in bv[np.lexsort((bv, ang[bv]))][::-1].tolist():
                        if len(cs) <= 5:
                            break
                        if v in cs and not (strict and self.vtype[v] in (CONVEX, CONCAVE)):
                            cs.discard(v)
            out[p] = sorted(cs)
        return out

    def split_into_subpaths(self):
        """Cut every trace at vertices where traces meet or that lie on the border (SplitIntoSubPaths)."""
        q = np.bincount(np.array([n // 4 for t in self.paths for n in t.nodes], dtype=np.int64), minlength=self.nv)
        cut = (q >= 2) | (self.border_v & (q >= 1))
        new = []
        for t in self.paths:
            nodes = t.nodes
            if t.loop:                                                      # a loop starts and ends at its first cut
                hits = [i for i, x in enumerate(nodes) if cut[x // 4]]
                nodes = nodes[hits[0]:] + nodes[:hits[0] + 1] if hits else []
            at = [i for i in range(1, len(nodes) - 1) if cut[nodes[i] // 4]]
            if t.loop and not at:
                new.append(t)
            else:
                new += [Trace(nodes[a:b + 1], False, t.unremovable) for a, b in zip([0] + at, at + [len(nodes) - 1]) if b > a]
        self.set_paths(new)

    @staticmethod
    def merge_contiguous(paths):
        """Join open traces that continue each other, also through reversed nodes (MergeContiguousPaths)."""
        paths = [Trace(list(t.nodes), t.loop, t.unremovable) for t in paths]
        while True:
            starts, ends = {}, {}
            for k, t in enumerate(paths):
                if not t.loop:
                    starts.setdefault(t.nodes[0], []).append(k)
                    ends.setdefault(t.nodes[-1], []).append(k)
            merged = False
            for i, A in enumerate(paths):
                js = i + 1
                while A.nodes and not A.loop:
                    a = A.nodes
                    # traces after A are unchanged or emptied, so the first candidate continues A in one of these ways
                    cand = [j for j in starts.get(a[-1], []) + ends.get(a[0], []) + ends.get(a[-1] ^ 2, []) + starts.get(a[0] ^ 2, [])
                            if j >= js and paths[j].nodes]
                    if not cand:
                        break
                    j = min(cand)
                    b = paths[j].nodes
                    if a[-1] == b[0]:
                        new = a[:-1] + b
                    elif b[-1] == a[0]:
                        new = b[:-1] + a
                    elif a[-1] == b[-1] ^ 2:
                        new = a[:-1] + [x ^ 2 for x in reversed(b)]
                    else:
                        new = [x ^ 2 for x in reversed(a)][:-1] + b
                    paths[j].nodes = []
                    A.loop = new[0] == new[-1]
                    A.nodes = new[:-1] if A.loop else new
                    js = j + 1
                    merged = True
            paths = [t for t in paths if t.nodes]
            if not merged:
                return paths

    def remove_traces(self):
        """Dissolve unneeded traces, then cut the rest at crossings and dissolve again (BatchRemovalMetaMesh)."""
        for step in range(2):
            if step:
                self.split_into_subpaths()
            self.canonical()
            mm = MetaMesh(self, self.fixed_corners(), np.nonzero(np.isin(self.vtype, (CONCAVE, CONVEX)))[0].tolist(),
                          set(np.nonzero(self.ptype == NON_DISK)[0].tolist()))
            mm.set_path_ids(self, [(np.asarray(t.nodes) // 4, t.loop) for t in self.paths])
            mm.merge_loop(self.avg_edge, {i for i, t in enumerate(self.paths) if t.unremovable})
            keep = mm.remaining_paths()
            self.set_paths(self.merge_contiguous([t for i, t in enumerate(self.paths) if i in keep]))
        self.update_partitions()

    # ------------------------------------------------------------------ smoothing and output
    def smooth_patches(self, steps):
        """Smooth self.P (SmoothMeshPatchesFromEdgeSel): traced vertices slide along their trace, others get Laplacian
        smoothing, all reprojected onto the unsmoothed surface. Vertices that fold or thin out a face move back."""
        dev = self.dev
        P = self._tensor(self.P)
        F = self.g.F
        surf = ck.mesh_bvh(P[F].float())
        project = lambda X: ck.closest_point_on_mesh(surf, X.float())[1].double()
        sel, border = self._sel_t, self.g.border
        tail, head = F.reshape(-1), F.roll(-1, 1).reshape(-1)
        once = sel & ((self._opp_t < 0) | (tail < head))
        ea, eb = tail[once], head[once]
        sel_sum = RowSum(torch.cat([ea, eb]), self.nv)
        all_sum = RowSum(torch.cat([tail, head]), self.nv)
        on_trace, slide = sel_sum.lengths > 0, (sel_sum.lengths == 2) & ~border
        deg = all_sum.lengths.double()
        both = sel.view(-1, 3) & sel.view(-1, 3).roll(-1, 1)
        guard_v = F[torch.arange(self.nf, device=dev), 2 - both.flip(1).int().argmax(1)]
        guard = both.any(1) & ~border[guard_v]

        def normals(P):
            n = torch.linalg.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
            return n / n.norm(dim=1, keepdim=True).clamp_min(1e-300)

        def quality(P):                                                     # QualityRadii: 2 * inradius / circumradius
            a, b, c = ((P[F[:, (k + 1) % 3]] - P[F[:, k]]).norm(dim=1) for k in range(3))
            s = (a + b + c) / 2
            a2 = s * (s - a) * (s - b) * (s - c)
            return torch.where(a2 > 0, 8 * a2 / (a * b * c * s).clamp_min(1e-300), 0.0)

        for _ in range(steps):
            old, old_n = P, normals(P)
            P = project(torch.where(slide[:, None], 0.5 * P + 0.5 * sel_sum(torch.cat([P[eb], P[ea]])) / 2, P))
            P = project(torch.where(on_trace[:, None], P, 0.5 * P + 0.5 * all_sum(torch.cat([P[head], P[tail]])) / deg.clamp_min(1)[:, None]))
            for _ in range(8):
                bad = guard & (((normals(P) * old_n).sum(1) < 0) | (quality(P) < 0.2))
                back = torch.zeros(self.nv, dtype=torch.bool, device=dev)
                back[guard_v[bad]] = True
                back &= (P != old).any(1)
                if not bool(back.any()):
                    break
                P = torch.where(back[:, None], old, P)
        rest_n = normals(self.g.P)
        move = torch.zeros(self.nv, dtype=torch.bool, device=dev)
        move[F[(quality(P) < 0.2) | ((normals(P) * rest_n).sum(1) < 0)].reshape(-1)] = True
        move &= ~border
        for _ in range(10):
            P = torch.where(move[:, None], (P + all_sum(torch.cat([P[head], P[tail]]))) / (deg + 1)[:, None], P)
        self.P = P.cpu().numpy()
        self.wedge = self._wedge_angles(self.P)

    def patches(self):
        """Layout as (vertices, faces, patch per face, corners per patch), dropping non-disk patches (SaveAllData)."""
        self.canonical()
        corners = self.fixed_corners()
        keep_part = self.ptype != NON_DISK
        keep = keep_part[self.label]
        return self.P, self.F[keep], (np.cumsum(keep_part) - 1)[self.label[keep]], [c for c, k in zip(corners, keep_part) if k]


def trace_patches(verts, faces, face_dir, progress_callback):
    """Patch layout of a triangle mesh along its cross field (QuadWild's trace()). Calls progress_callback per step."""
    g, pd1f = build_graph(verts, faces, face_dir)
    tr = Tracer(g, pd1f, np.zeros(g.P.shape[0], dtype=np.int64))
    tr.init_types()
    tr.sample_loop_emitters()
    tr.trace_tip_rings()
    progress_callback()
    tr.batch_add_loops()
    progress_callback()
    for only_non_disk, force_always in ((False, False), (True, False), (True, True)):
        tr.solve_sub_patches(only_non_disk, force_always)
        progress_callback()
    tr.remove_traces()
    progress_callback()
    tr.smooth_patches(10)
    tr.smooth_patches(3)
    progress_callback()
    return tr.patches()
