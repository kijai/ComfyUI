"""The field-line graph QuadWild traces patch borders on (xfield_tracer PreProcessMesh + VertexFieldGraph), for meshes
without sharp features: every edge around a singular vertex is split at its midpoint, the face cross field is averaged
onto the vertices, and node 4v+d (direction d of [PD1, PD2, -PD1, -PD2] at vertex v) links to the nodes of its 1- and
2-ring neighbours that lie within 90 degrees of d, with the direction label carried across by the minimal rotation
between the vertex normals. Internal singular vertices get no links, so traces pass around them."""
from __future__ import annotations

from dataclasses import dataclass

import comfy_kitchen as ck
import numpy as np
import torch

from .field import _normals, opposite_halfedges, row_sum

# RefineE split table (vcg refine.h): child triangles over [V0, V1, V2, mid01, mid12, mid20] per split-edge mask, and the
# diagonals compared to pick the shorter one when a face splits into three
_SPLIT = [
    ([(0, 1, 2)], None),
    ([(0, 3, 2), (3, 1, 2)], None),
    ([(0, 1, 4), (0, 4, 2)], None),
    ([(3, 1, 4), (0, 3, 2), (4, 2, 3)], ((0, 4), (3, 2))),
    ([(0, 1, 5), (5, 1, 2)], None),
    ([(0, 3, 5), (3, 1, 5), (2, 5, 1)], ((3, 2), (5, 1))),
    ([(2, 5, 4), (0, 1, 5), (4, 5, 1)], ((0, 4), (5, 1))),
    ([(3, 4, 5), (0, 3, 5), (3, 1, 4), (5, 4, 2)], None),
]
DRIFT, MAX_ANGLE = 100.0, 45.0
# mean links a path expansion search may cover before its hop counts as one that cannot be expanded
HOP_REACH = 32
# bytes of GPU memory the searches' distance rows and hash tables are sized for
BUDGET = 2 ** 29


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-300)


def _ranges(start, count):
    """(range id, index) of every entry of the index ranges start[r] .. start[r] + count[r] - 1."""
    rep = torch.repeat_interleave(torch.arange(len(count), device=count.device), count)
    return rep, start[rep] + torch.arange(len(rep), device=count.device) - (torch.cumsum(count, 0) - count)[rep]


def _rotate(v, a, b):
    """Apply the minimal rotation taking unit a to unit b to v (vcg RotationMatrix)."""
    c = (a * b).sum(-1, keepdim=True)
    # antiparallel normals: 180 degrees about an axis perpendicular to a (only on degenerate folds)
    perp = torch.where(a[..., :1].abs() < 0.9, a.new_tensor([1.0, 0, 0]), a.new_tensor([0, 1.0, 0]))
    k = torch.where(c < -1 + 1e-9, _unit(torch.linalg.cross(a, perp)), _unit(torch.linalg.cross(a, b)))
    phi = torch.acos(c.clamp(-1, 1))
    cs, sn = phi.cos(), phi.sin()
    r = v * cs + torch.linalg.cross(k, v) * sn + k * (k * v).sum(-1, keepdim=True) * (1 - cs)
    return torch.where(c > 1 - 1e-9, v, r)


def _cross_axes(pd1, n):
    pd2 = _unit(torch.linalg.cross(n, pd1))
    return torch.stack([pd1, pd2, -pd1, -pd2], -2)


def _mismatch(axes, fn, f, g):
    """Period jump from face f to face g (FollowDirection of direction 0; ties to the first index)."""
    r = torch.where(((fn[f] * fn[g]).sum(-1, keepdim=True) < -0.99999), -axes[f, 0], _rotate(axes[f, 0], fn[f], fn[g]))
    return (axes[g] * r[:, None]).sum(-1).argmax(-1)


def singular_index(F, opp, pd1f, fn, nv):
    """Field index per vertex (IsSingularByCross with border vertices): period jumps summed clockwise around the fan, plus
    a jump from the last to the first fan face at border vertices. Returns 0 (regular) .. 3."""
    axes = _cross_axes(pd1f, fn)
    v = F.reshape(-1)
    f = torch.arange(len(v), device=F.device) // 3
    inner = opp >= 0
    total = torch.zeros(nv, dtype=torch.long, device=F.device)
    total.index_add_(0, v[inner], _mismatch(axes, fn, f[inner], opp[inner] // 3))
    # border fan: from the face whose edge leaving v is a border to the face whose edge entering v is a border
    last, first = torch.full((2, nv), -1, dtype=torch.long, device=F.device)
    last[v[~inner]] = f[~inner]
    entering = ~inner.view(-1, 3).roll(1, 1).reshape(-1)          # edge (k+2)%3 ends at F[f, k]
    first[v[entering]] = f[entering]
    wrap = (last >= 0) & (first >= 0)
    total[wrap] += _mismatch(axes, fn, last[wrap], first[wrap])
    return total % 4


def _refine(P, F, split):
    """Split the marked edges at their midpoints with vcg RefineE's patterns and numbering (new vertices in order of first
    occurrence over faces and their edges, first child in place of its face, other children appended in face order).
    Returns the vertices, the faces and the parent face of each."""
    key = np.minimum(F, np.roll(F, -1, 1)) * len(P) + np.maximum(F, np.roll(F, -1, 1))
    marked = split[F] | split[np.roll(F, -1, 1)]
    uniq, first = np.unique(key[marked], return_index=True)
    ends = uniq[np.argsort(first)]
    mid_of = dict(zip(ends.tolist(), range(len(P), len(P) + len(ends))))
    newP = np.concatenate([P, 0.5 * (P[ends // len(P)] + P[ends % len(P)])])
    mask = marked @ np.array([1, 2, 4])
    faces, extra, parent = F.copy(), [], []
    for fi in np.nonzero(mask)[0]:
        vv = list(F[fi]) + [mid_of.get(k, -1) for k in key[fi].tolist()]
        tris, swap = _SPLIT[mask[fi]]
        t = [[vv[i] for i in tri] for tri in tris]
        if swap is not None:
            d = lambda e: ((newP[vv[e[0]]] - newP[vv[e[1]]]) ** 2).sum()
            if d(swap[0]) < d(swap[1]):
                t[2][1] = t[1][0]
                t[1][1] = t[2][0]
        faces[fi] = t[0]
        extra += t[1:]
        parent += [fi] * (len(t) - 1)
    newF = np.concatenate([faces, np.array(extra, dtype=np.int64).reshape(-1, 3)])
    return newP, newF, np.concatenate([np.arange(len(F)), np.array(parent, dtype=np.int64)])


@dataclass
class FieldGraph:
    P: torch.Tensor            # (V, 3) float64 vertex positions
    F: torch.Tensor            # (F, 3) faces
    vn: torch.Tensor           # (V, 3) area-weighted vertex normals
    dirs: torch.Tensor         # (V, 4, 3) vertex cross field [PD1, PD2, -PD1, -PD2]
    valence: torch.Tensor      # (V,) singularity valence (4 = regular)
    border: torch.Tensor       # (V,) border vertex
    singular: torch.Tensor     # (V,) singular vertex (border ones included)
    indptr: torch.Tensor       # (4V+1,) links of node n are indptr[n]:indptr[n+1], ordered by angle
    dst: torch.Tensor          # (L,) target node
    angle: torch.Tensor        # (L,) degrees between the node direction and the link
    dist: torch.Tensor         # (L,) link length
    direct: torch.Tensor       # (L,) link is a mesh edge (else a 2-ring jump)


def _vertex_field(F, fn, z, pd1f):
    """Per-vertex 4-RoSy average of the incident face fields in a fixed frame around the vertex normal z (vcg
    InterpolateNRosy3D)."""
    x = _unit(torch.linalg.cross(z, torch.where(z[:, :1].abs() > 0.999, z.new_tensor([0, 1.0, 0]), z.new_tensor([1.0, 0, 0]))))
    y = _unit(torch.linalg.cross(x, z))
    fv = F.reshape(-1)
    fi = torch.arange(len(fv), device=F.device) // 3
    r = _unit(_rotate(pd1f[fi], fn[fi], z[fv]))
    back = (fn[fi] * z[fv]).sum(-1) <= -0.99999
    cx = torch.where(back, pd1f[fi, 0], (x[fv] * r).sum(-1))
    cy = torch.where(back, pd1f[fi, 1], (y[fv] * r).sum(-1))
    th = 4 * torch.atan2(cy, cx)
    acc = row_sum(fv, torch.stack([th.cos(), th.sin()], 1), len(z))
    a = torch.atan2(acc[:, 1], acc[:, 0]) / 4
    return _cross_axes(_unit(x * a.cos()[:, None] + y * a.sin()[:, None]), z)


def build_graph(P: torch.Tensor, F: torch.Tensor, face_dir: torch.Tensor) -> tuple[FieldGraph, torch.Tensor]:
    """Preprocessed mesh (edges around singular vertices split) and its field graph from a triangle mesh and its
    per-face cross-field direction. Also returns the per-face PD1 of the split mesh (sub-patch graphs inherit it)."""
    pd1f = _unit(face_dir)
    sing = singular_index(F, opposite_halfedges(F), pd1f, _normals(P, F)[0], len(P)) != 0
    newP, newF, parent = _refine(P.cpu().numpy(), F.cpu().numpy(), sing.cpu().numpy())
    P, F, pd1f = torch.as_tensor(newP, device=P.device), torch.as_tensor(newF, device=P.device), pd1f[torch.as_tensor(parent, device=P.device)]
    fn, vn = _normals(P, F)
    valence = torch.tensor([4, 5, 6, 3], device=P.device)[singular_index(F, opposite_halfedges(F), pd1f, fn, len(P))]
    return link_graph(P, F, vn, _vertex_field(F, fn, vn, pd1f), pd1f, valence), pd1f


def link_graph(P, F, vn, dirs, pd1f, valence) -> FieldGraph:
    """Field graph of a mesh whose vertex normals, vertex field and singularity valences are given (a sub-patch keeps
    its parent's); singular vertices are found on this mesh, so patch borders get their own."""
    dev, nv = P.device, len(P)
    opp = opposite_halfedges(F)
    singular = singular_index(F, opp, pd1f, _normals(P, F)[0], nv) != 0
    half_a, half_b = F.reshape(-1), F.roll(-1, 1).reshape(-1)
    border = torch.zeros(nv, dtype=torch.bool, device=dev)
    border[torch.cat([half_a[opp < 0], half_b[opp < 0]])] = True

    # candidate vertex pairs: 1-ring without border edges and internal singular vertices, plus the 2-ring through and
    # onto non-border vertices
    internal_sing = singular & ~border
    ok = (opp >= 0) & ~internal_sing[half_a] & ~internal_sing[half_b]
    ring1 = torch.unique(torch.stack([half_a[ok], half_b[ok]], 1), dim=0)          # sorted by (i, k)
    start = torch.searchsorted(ring1[:, 0].contiguous(), torch.arange(nv + 1, device=dev))
    via = ring1[~border[ring1[:, 1]]]
    rep, at = _ranges(start[via[:, 1]], start[via[:, 1] + 1] - start[via[:, 1]])
    i2, k2 = via[rep, 0], ring1[at, 1]
    keep2 = ~border[k2] & (k2 != i2)
    pairs, inv = torch.unique(torch.cat([ring1, torch.stack([i2[keep2], k2[keep2]], 1)]), dim=0, return_inverse=True)
    is_direct = torch.zeros(len(pairs), dtype=torch.bool, device=dev)
    is_direct[inv[:len(ring1)]] = True

    i, k = pairs[:, 0], pairs[:, 1]
    t = _unit(P[k] - P[i])
    n0 = _unit(vn[i])
    tn = (t * n0).sum(-1, keepdim=True)
    t = torch.where(tn.abs() > 0.1, _unit(t - tn * n0), t)
    cos = ((dirs[i] * t[:, None]).sum(-1) / (dirs[i].norm(dim=-1) * t.norm(dim=-1, keepdim=True))).clamp(-1, 1)
    ang = torch.rad2deg(torch.acos(cos))                                    # (pairs, 4)
    r = _rotate(dirs[i], vn[i][:, None].expand(-1, 4, -1), vn[k][:, None].expand(-1, 4, -1))
    d1 = 3 - (dirs[k][:, None] * r[:, :, None]).sum(-1).flip(-1).argmax(-1)       # closest direction, ties to the last
    pi, d = torch.nonzero(ang < 90, as_tuple=True)                       # links (pair, direction)

    # drop links into dead ends (nodes without links) until none is left, except into border nodes
    while True:
        src, dst = 4 * i[pi] + d, 4 * k[pi] + d1[pi, d]
        drop = (torch.bincount(src, minlength=4 * nv) == 0)[dst] & ~border[dst // 4]
        if not bool(drop.any()):
            break
        pi, d = pi[~drop], d[~drop]
    order = torch.argsort(ang[pi, d], stable=True)
    order = order[torch.argsort(src[order], stable=True)]
    pi, d = pi[order], d[order]
    indptr = torch.searchsorted(src[order], torch.arange(4 * nv + 1, device=dev))
    return FieldGraph(P, F, vn, dirs, valence, border, singular, indptr, dst[order], ang[pi, d], (P[k] - P[i]).norm(dim=-1)[pi], is_direct[pi])


class Queries:
    """Batched shortest-path queries on a FieldGraph with xfield_tracer's VertexFieldQuery::ShortestPath results: each
    query is relaxed frontier by frontier on its own distance row (pruned by the best answer found so far), then the
    path is read back taking, at every node, the predecessor Dijkstra would have settled it from (smallest distance,
    then largest node id)."""

    def __init__(self, g: FieldGraph):
        self.n = 4 * g.P.shape[0]
        dev = g.dst.device
        src = torch.repeat_interleave(torch.arange(self.n, device=dev), g.indptr[1:] - g.indptr[:-1])
        drift = g.dist * (1 + DRIFT * (g.angle / MAX_ANGLE) ** 2)
        self.w = {"drift": torch.where(g.angle > MAX_ANGLE, torch.inf, drift), "length": g.dist, "edge": torch.where(g.direct, g.dist, torch.inf)}
        order = torch.argsort(g.dst, stable=True)
        indeg = torch.bincount(g.dst, minlength=self.n)
        self.in_ptr = torch.cat([indeg.new_zeros(1), torch.cumsum(indeg, 0)]).int()
        self.in_tail = src[order].int()
        self.w_in = {k: w[order] for k, w in self.w.items()}
        node, slot = _ranges(torch.zeros_like(indeg), indeg)
        self.closers = torch.full((self.n, int(indeg.max())), -1, dtype=torch.long, device=dev)
        self.closers[node, slot] = torch.where(torch.isfinite(self.w_in["drift"]), src[order], -1)   # nodes a loop may close from
        self.node_border = g.border.repeat_interleave(4)
        self.indptr, self.dst = g.indptr.int(), g.dst.int()
        self.rows = max(1, BUDGET // (self.n * 28))      # distances plus the search's queues, far piles and list
        mean = float(g.dist.mean()) if len(g.dist) else 1.0     # sub-patches of singular vertices only have no links
        self.hop_reach = HOP_REACH * mean          # cap of the sparse path expansion searches
        self.delta = {"drift": 32 * mean, "length": 8 * mean, "edge": 8 * mean}   # bucket widths

    def _search(self, starts, kind, active, end=None, stop=None, reach=None, blocked=None, closer=None):
        """Delta-stepping from one start node per row, each row pruned by its best answer so far (lowered when it
        reaches its end node and by loop closers: closer = start node per row), stopping at stop nodes and not expanding
        nodes beyond reach (B,): the sorted row * n + node keys of the nodes reached and their distances, in a hash table
        within the budget."""
        bound = torch.full((len(starts),), torch.inf, dtype=torch.float64, device=starts.device)
        return ck.shortest_paths_sparse(self.indptr, self.dst, self.w[kind], starts, bound, self.delta[kind], self.n, active=active, blocked=blocked,
                                        stop=stop, end=end, closer=closer, reach=reach, max_slots=BUDGET // 32)

    def _walk_back(self, dist, starts, ends, reach=None, stop=None):
        """Drift-weighted node paths start -> end per row (None where end < 0) in a dense (B, N) field or sorted (keys,
        distances); predecessors beyond reach (B,) or stop nodes do not count."""
        nodes, lengths = ck.trace_paths(self.in_ptr, self.in_tail, self.w_in["drift"], starts, ends, dist, self.n, reach=reach, stop=stop)
        lengths = lengths.cpu().numpy()
        return [p.tolist() if k else None for p, k in zip(np.split(nodes.cpu().numpy(), np.cumsum(lengths)[:-1]), lengths)]

    def _at(self, dist, r, v):
        """Distances of (row, node) pairs in a dense (B, N) field or sorted (keys, distances)."""
        if isinstance(dist, torch.Tensor):
            return dist[r, v]
        keys, kd = dist
        k = r * self.n + v
        pos = torch.searchsorted(keys, k.reshape(-1)).clamp_max(len(keys) - 1).view(k.shape)
        return torch.where(keys[pos] == k, kd[pos], torch.inf)

    def loops(self, sources, active, max_weight=None):
        """FindLoop: per source the drift-weighted path back to a node linked to the source, choosing the closest such
        node (the closing link's own weight does not count). The loop is returned without repeating the start. Loops
        known to be short (max_weight: nodes farther are not expanded) are searched in hash tables, the others, which
        reach much of the mesh, on dense (B, N) distance rows."""
        out, c, step = [], 0, self.rows
        while c < len(sources):
            s = sources[c:c + step]
            rows = torch.arange(len(s), device=s.device)
            if max_weight is None:
                dist = torch.full((len(s), self.n), torch.inf, dtype=torch.float64, device=s.device)
                dist[rows, s] = 0
                ck.shortest_paths(self.indptr, self.dst, self.w["drift"], dist, rows * self.n + s, dist.new_full((len(s),), torch.inf),
                                  self.delta["drift"], active=active, closer=s)
            else:
                dist = self._search(s, "drift", active, reach=torch.full((len(s),), max_weight, dtype=torch.float64, device=s.device), closer=s)
                # the next chunk's results (about 32 bytes per node reached) should fit the budget: sized from this one and
                # growing at most 4x, as the nodes reached per row vary
                step = max(1, min(4 * len(s), int(BUDGET * len(s) / (32 * max(1, len(dist[0]))))))
            u = self.closers[s]
            du = torch.where((u >= 0) & active[s][:, None], self._at(dist, rows[:, None].expand_as(u), u), torch.inf)
            best = du.min(1, keepdim=True).values
            end = torch.where((du == best) & torch.isfinite(best), u, -1).max(1).values
            out += self._walk_back(dist, s, end)
            c += len(s)
        return out

    def first_hit(self, sources, active, selected, max_weight=None):
        """TraceDijkstraPath: drift-weighted path from each source to the closest selected node (the source itself
        deselected), links over 45 degrees skipped, nodes beyond max_weight (per source) not expanded. None where
        unreachable. These searches stop at the first receiver, so they run sparse."""
        keys, kd = self._search(sources, "drift", active, stop=selected, reach=max_weight)
        q, v = keys // self.n, keys % self.n
        hit = selected[v] & (v != sources[q])
        q, v, d = q[hit], v[hit], kd[hit]
        best = torch.full((len(sources),), torch.inf, dtype=torch.float64, device=sources.device).scatter_reduce_(0, q, d, "amin")
        end = torch.full((len(sources),), -1, dtype=torch.long, device=sources.device).scatter_reduce_(0, q, torch.where(d == best[q], v, -1), "amax")
        return self._walk_back((keys, kd), sources, end, max_weight, selected)

    def between(self, starts, ends, active):
        """GetSubSequence: shortest path over mesh-edge links from each start node to its end node, entering border
        vertices only at the end, as concatenated nodes and offsets per row (empty where unreachable). The searches run
        up to hop_reach; the rows the cap cut short are redone without it. Also returns what the searches leave in
        xfield_tracer's shared distance array as (row, node, distance): every node settled before the end (closer, or as
        close with a larger id) and every node those relax, or everything reached by a search without an end."""
        n, reach = self.n, self.hop_reach
        keys, kd = self._search(starts, "edge", active, end=ends, blocked=self.node_border,
                                reach=torch.full((len(starts),), reach, dtype=torch.float64, device=starts.device))
        rows = torch.arange(len(starts), device=starts.device)
        end_d = self._at((keys, kd), rows, ends)
        far = torch.nonzero(torch.isin(rows, keys[kd > reach] // n) & ~torch.isfinite(end_d))[:, 0]
        if len(far):
            fk, fd = self._search(starts[far], "edge", active, end=ends[far], blocked=self.node_border)
            keep = ~torch.isin(keys // n, far)
            keys, order = torch.sort(torch.cat([keys[keep], far[fk // n] * n + fk % n]))
            kd = torch.cat([kd[keep], fd])[order]
            end_d = self._at((keys, kd), rows, ends)
        found = torch.isfinite(end_d)
        nodes, lengths = ck.trace_paths(self.in_ptr, self.in_tail, self.w_in["edge"], starts, torch.where(found, ends, -1), (keys, kd), n)
        paths = nodes.cpu().numpy(), torch.cat([lengths.new_zeros(1), torch.cumsum(lengths, 0)]).cpu().numpy()

        q, v = keys // n, keys % n
        settled = ~found[q] | (kd < end_d[q]) | ((kd == end_d[q]) & (v > ends[q]))
        grow = settled & found[q]
        qq, to, cand = self._relax_once(q[grow], v[grow], kd[grow], active, ends)
        wk, inv = torch.unique(torch.cat([keys[settled], qq * n + to]), return_inverse=True)
        wd = torch.full((len(wk),), torch.inf, dtype=torch.float64, device=q.device).scatter_reduce_(0, inv, torch.cat([kd[settled], cand]), "amin")
        return paths, (wk // n, wk % n, wd)

    def _relax_once(self, q, v, d, active, ends):
        """Mesh-edge links out of the settled (row q, node v, distance d) entries of GetSubSequence searches, expanded a
        budget's worth of arcs (about 64 bytes each) at a time."""
        deg = self.indptr[v + 1] - self.indptr[v]
        ends_at = torch.cumsum(deg, 0)
        step = BUDGET // 64
        total = int(ends_at[-1]) if len(deg) else 0
        cuts = [0] + torch.searchsorted(ends_at, torch.arange(1, max(1, -(-total // step)), device=q.device) * step).tolist() + [len(q)]
        out = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            rep, arc = _ranges(self.indptr[v[lo:hi]], deg[lo:hi])
            qq, to = q[lo:hi][rep], self.dst[arc]
            cand = d[lo:hi][rep] + self.w["edge"][arc]
            ok = torch.isfinite(cand) & active[to] & (~self.node_border[to] | (to == ends[qq]))
            out.append((qq[ok], to[ok], cand[ok]))
        return tuple(torch.cat(t) for t in zip(*out))

    def distances(self, sources, active, dist=None):
        """UpdateDistancesFrom: link-length distance from the nearest source (decrease-only when a previous field is
        given). Nodes not reached stay +inf (or keep their previous value)."""
        dist = torch.full((self.n,), torch.inf, dtype=torch.float64, device=sources.device) if dist is None else dist.clone()
        dist[sources] = 0
        ck.shortest_paths(self.indptr, self.dst, self.w["length"], dist.view(1, -1), sources, dist.new_full((1,), torch.inf),
                          self.delta["length"], active=active)
        return dist

    def walk(self, sources, active, selected):
        """TraceToSelected: from each source follow every node's smallest-angle link until a selected node; fails on a
        node without links, an inactive one, or one already marked (visited nodes and their active reverses)."""
        if not len(self.dst):
            return [None] * len(sources)
        nxt0 = torch.where(self.indptr[:-1] < self.indptr[1:], self.dst[self.indptr[:-1].clamp_max(len(self.dst) - 1)], -1)
        twin = torch.arange(self.n, device=sources.device) ^ 2
        out = []
        for c in range(0, len(sources), 160 * self.rows):                  # a bit per node and walk
            paths, k = ck.walk_successors(nxt0, sources[c:c + 160 * self.rows], selected, active=active, twin=twin)
            paths = paths.cpu().numpy()
            out += [paths[r, :x].tolist() if x else None for r, x in enumerate(k.tolist())]
        return out
