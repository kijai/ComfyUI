"""Integer subdivision of every subside with QuadWild's Bi-MDF quantization objective (Heistermann et al. 2023, "Min-
Deviation-Flow in Bi-directed Graphs for T-Mesh Quantization"): the quadratic deviation of each subside from its target
edge count plus, per patch, the cheapest emergency flow of its Bi-MDF subgraph fitting its side lengths to a pattern (a
closed form per valence, _regularity). Minimised by local search from an even-perimeter rounding: Bellman-Ford on the GPU
over turn costs finds negative cycles of +-1 and paths of +-2 subside changes (both keep every perimeter even), and the
exactly improving, patch-disjoint ones are applied each round."""
from __future__ import annotations

import heapq
import itertools

import comfy_kitchen as ck
import numpy as np
import torch


def _tri(t, w_sob, w_loop):
    """Patch cost of a triangle of inner edges (sides t (..., 3), even sum): every side must stay below the sum of the other
    two minus one, via singularity-on-boundary edges (+1 on the other two sides) and side loops (-2) on the longest side."""
    sob = torch.tensor(list(itertools.product((0, 1), repeat=3)), device=t.device)
    tt = t[..., None, :] + sob.sum(1, keepdim=True) - sob
    e = 2 * tt - tt.sum(-1, keepdim=True)
    emax, m = e.max(-1)
    loops = (emax + 2).clamp_min(0) // 2
    tt = tt - 2 * loops[..., None] * torch.nn.functional.one_hot(m, 3)
    ok = ((2 * tt - tt.sum(-1, keepdim=True)) <= -2).all(-1)
    return torch.where(ok, w_sob * sob.sum(1) + w_loop * loops, torch.inf).amin(-1).double()


def _regularity(S):
    """Min emergency cost of a patch with side sums S (m, n) for Bi-MDF regularity weight 1 (QuadRetopology make_bimdf)."""
    n = S.shape[1]
    if n == 3:
        return _tri(S, 1, 4)
    if n == 4:
        return 2 * ((S[:, 0] - S[:, 2]).abs() + (S[:, 1] - S[:, 3]).abs()).double()
    if n == 5:
        return (2 * (S + S.roll(-1, 1)) - S.sum(1, keepdim=True) + 1).clamp_min(0).sum(1).double()
    # hexagon: two triangles of inner edges, an odd pair of triangle sums needs one neighbour or opposite link
    links = torch.tensor([(i, (i + 1) % 6) for i in range(6)] + [(i, i + 3) for i in range(3)], device=S.device)
    shift = torch.cat([S.new_zeros(1, 6), -torch.nn.functional.one_hot(links, 6).sum(1)])       # no link, then each one
    link_cost = torch.tensor([0.0] + [3.0] * 6 + [6.0] * 3, dtype=torch.float64, device=S.device)
    t = S[:, None] + shift
    odd = S[:, 0::2].sum(1, keepdim=True) % 2 == 1
    use = odd == (torch.arange(len(links) + 1, device=S.device) > 0)
    total = link_cost + _tri(t[..., 0::2], 0.5, 1) + _tri(t[..., 1::2], 0.5, 1)
    return torch.where(use, total, torch.inf).amin(1)


def _make_even(charts, subsides, s, quad, target):
    """Greedy T-join: pair odd-perimeter patches along the cheapest paths of +-1 subside moves."""
    nb = len(charts)                                  # node id for the outer boundary
    adj = {}
    for k, sub in enumerate(subsides):
        a, b = sub.charts[0], sub.charts[1] if sub.charts[1] >= 0 else nb
        adj.setdefault(a, []).append((b, k))
        adj.setdefault(b, []).append((a, k))

    def move(k):
        up = quad[k] * (2 * (s[k] - target[k]) + 1)
        down = quad[k] * (1 - 2 * (s[k] - target[k])) if s[k] > 1 else np.inf
        return (up, 1) if up <= down else (down, -1)

    odd = {ci for ci, c in enumerate(charts) if sum(int(s[k]) for k in c.subsides) % 2}
    price = [max(move(k)[0], 0.0) for k in range(len(s))]             # each subside's cheaper move, kept current
    while odd:
        src = odd.pop()
        dist, prev = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u != src and (u in odd or u == nb):
                break
            for v, k in adj.get(u, ()):
                nd = d + price[k] + 1e-9
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = (u, k)
                    heapq.heappush(pq, (nd, v))
        else:
            raise RuntimeError("quad remesh quantization: odd patch perimeter with no partner to pair with")
        odd.discard(u)
        while u != src:
            u, k = prev[u]
            s[k] += move(k)[1]
            price[k] = max(move(k)[0], 0.0)
    return s


def quantize(charts, subsides, edge_length: float, alpha: float = 0.05, device="cpu", start: np.ndarray | None = None) -> np.ndarray:
    """Even-perimeter integer subdivisions for all subsides (array indexed by subside id); the local search starts
    from the rounded targets, or from start when given."""
    dev = torch.device(device)
    active = [ci for ci, c in enumerate(charts) if len(c.faces)]
    nside = np.array([len(charts[ci].sides) for ci in active])
    if ((nside < 3) | (nside > 6)).any():
        raise ValueError(f"quad remesh: patch with {nside[(nside < 3) | (nside > 6)][0]} sides")
    n = torch.as_tensor(nside, dtype=torch.float64)
    weight = ((1 - alpha) / n ** 2 * torch.where(n == 4, 1.0, 0.9)).numpy()
    row = {ci: r for r, ci in enumerate(active)}
    target = np.array([max(1.0, sub.length / edge_length) for sub in subsides])
    quad = np.array([sum(alpha / len(charts[c].subsides) for c in sub.charts if c >= 0) for sub in subsides])
    occ_row = np.array([row.get(c, -1) for sub in subsides for c in sub.charts])          # occurrence 2k+d
    occ_side = np.array([j for sub in subsides for j in sub.chart_sides])
    s = _make_even(charts, subsides, np.maximum(1, np.rint(target) if start is None else start).astype(np.int64), quad, target)

    # turn graph: node 4k+2d+(sigma<0) = subside k changed by sigma, entering the patch at its end d, and the boundary hub
    # 4 * len(subsides); arcs leave through one occurrence of a patch and enter through another of a different subside,
    # ordered by patch, then by the two occurrences' places in the patch, then by a and b
    occs = [[2 * k + d for k in charts[ci].subsides for d in (0, 1) if subsides[k].charts[d] == ci] for ci in active]
    size = np.array([len(o) for o in occs])
    occ = np.concatenate(occs).astype(np.int64)
    m = np.repeat(size, size)
    i = np.repeat(np.arange(len(occ)), m)
    j = np.repeat(np.repeat(np.cumsum(size) - size, size), m) + np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)
    oi, oo, r = occ[i], occ[j], np.repeat(np.arange(len(active)), size)[i]
    keep = oi // 2 != oo // 2
    oi, oo, r = oi[keep, None], oo[keep, None], r[keep, None]
    a, b = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    hub = 4 * len(subsides)
    o = np.repeat(np.nonzero(occ_row < 0)[0], 2)
    ob = np.tile([0, 1], len(o) // 2)
    hubs = np.full_like(o, hub)
    # arcs within the patches, then the boundary ones into the hub, then those out of it
    tail = np.concatenate([(2 * oi + a).reshape(-1), 2 * o + ob, hubs])
    head = np.concatenate([(2 * (oo ^ 1) + b).reshape(-1), hubs, 2 * (o ^ 1) + ob])
    tab = np.concatenate([(r * 144 + occ_side[oi] * 24 + a * 12 + occ_side[oo] * 2 + b).reshape(-1), np.full(2 * len(o), -1)])
    out = np.concatenate([(2 * (oo // 2) + b).reshape(-1), np.full_like(o, -1), 2 * (o // 2) + ob])
    order = np.argsort(head, kind="stable")                                       # arcs by head, for bellman_ford
    indptr = np.concatenate([[0], np.cumsum(np.bincount(head, minlength=hub + 1))])
    tail, tab, out = tail[order], tab[order], out[order]
    T = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
    tail_d, indptr_d, tail_list = T(tail, torch.int32), T(indptr, torch.int32), tail.tolist()
    tab, out = T(tab), T(out)
    occ_row_d, occ_side_d = T(occ_row), T(occ_side)
    quad_d, target_d = T(quad, torch.float64), T(target, torch.float64)
    node_occ, node_b = torch.arange(hub, device=dev) // 2, torch.arange(hub, device=dev) % 2
    single = torch.stack([torch.eye(6, dtype=torch.long, device=dev), -torch.eye(6, dtype=torch.long, device=dev)], 1).reshape(12, 6)
    pair = (single[:, None] + single[None]).reshape(144, 6)
    steps = torch.cat([torch.zeros(1, 6, dtype=torch.long, device=dev)] + [k * m for k in (1, 2) for m in (pair, single)])
    # per patch the cost change of every pair (144) and single (12) side change, of size 1 then 2
    table = torch.zeros(len(active), 312, dtype=torch.float64, device=dev)
    on_patch = np.nonzero(occ_row >= 0)[0]
    side_at = occ_row[on_patch] * 6 + occ_side[on_patch]

    def patch_cost(rows, S):
        """Regularity costs of the patches rows (numpy) with side sums S (len(rows), 6), on S's device."""
        cost = torch.zeros(len(rows), dtype=torch.float64, device=S.device)
        for n in range(3, 7):
            m = np.nonzero(nside[rows] == n)[0]
            if len(m):
                at = torch.as_tensor(m, device=S.device)
                cost[at] = _regularity(S[at, :n]) * torch.as_tensor(weight[rows[m]], device=S.device)
        return cost

    def exact_gains(cands, sv, S):
        """Exact cost change of each candidate (a dict subside -> delta), inf when a subside drops below one."""
        if not cands:
            return np.zeros(0)
        lens = np.array([len(d) for d in cands])
        ks = np.fromiter((k for d in cands for k in d), dtype=np.int64, count=lens.sum())
        dv = np.fromiter((x for d in cands for x in d.values()), dtype=np.int64, count=lens.sum())
        starts = np.cumsum(lens) - lens
        nx = sv[ks] + dv
        iso_k = quad[ks] * ((nx - target[ks]) ** 2 - (sv[ks] - target[ks]) ** 2)
        gains = np.array([iso_k[i:j].sum() for i, j in zip(starts.tolist(), (starts + lens).tolist())])
        ok = np.minimum.reduceat(nx, starts) >= 1
        gains[~ok] = np.inf
        # patch side changes, rows in the order the candidate's subsides first reach them
        o = np.stack([2 * ks, 2 * ks + 1], 1).reshape(-1)
        owner = np.repeat(np.arange(len(cands)), 2 * lens)
        hit = (occ_row[o] >= 0) & ok[owner]
        o, i, d = o[hit], owner[hit], np.repeat(dv, 2)[hit]
        key, first, inv = np.unique(i * len(active) + occ_row[o], return_index=True, return_inverse=True)
        rowS = np.zeros((len(key), 6), dtype=np.int64)
        np.add.at(rowS, (inv, occ_side[o]), d)
        by_first = np.argsort(first)
        own, rows = np.divmod(key[by_first], len(active))
        old = torch.as_tensor(S[rows])                                    # few rows: costs on the host
        c = patch_cost(np.concatenate([rows, rows]), torch.cat([old + torch.as_tensor(rowS[by_first]), old]))
        np.add.at(gains, own, (c[:len(rows)] - c[len(rows):]).numpy())
        return gains

    def patch_of(v):
        return -1 if v == hub else occ_row[v // 2]

    def moves_of(nodes, k, par):
        """Subside changes of a turn-graph walk (closed, or a path), plus those of every simple cycle it contains (turn
        costs are exact for a walk that enters each patch once, but double count at a patch it revisits), and the arcs
        of the walk that enter a patch it already went through."""
        first = patch_of(nodes[0] ^ 2) if nodes[0] != hub else -1
        seqs, stack, at, seen, arcs = [], [], {first: -1}, {first}, []
        for v in nodes:
            c = patch_of(v)
            if c in seen and c >= 0:
                arcs.append(par[v])
            seen.add(c)
            if c in at:
                seqs.append(stack[at[c] + 1:] + [v])
                del stack[at[c] + 1:]
                at = {key: p for key, p in at.items() if p < len(stack)}
            else:
                stack.append(v)
                at[c] = len(stack) - 1
        if stack:
            seqs.append(stack)
        if len(seqs) > 1:
            seqs.append(nodes)
        result = []
        for seq in seqs:
            d = {}
            for v in seq:
                if v != hub:
                    d[v // 4] = d.get(v // 4, 0) + k * (1 - 2 * (v % 2))
            d = {key: val for key, val in d.items() if val}
            if d:
                result.append(d)
        return result, arcs

    banned = {k: torch.zeros(len(tail), dtype=torch.bool, device=dev) for k in (1, 2)}
    dirty = np.arange(len(active))
    search = (1, 2)
    while True:
        S = np.bincount(side_at, weights=s[on_patch // 2], minlength=6 * len(active)).reshape(-1, 6).astype(np.int64)
        if len(dirty):
            rt = T(dirty)
            c = patch_cost(np.repeat(dirty, len(steps)), (T(S)[rt, None] + steps).reshape(-1, 6)).view(len(dirty), -1)
            table[rt] = c[:, 1:] - c[:, :1]
        cands, bans = [], {1: [], 2: []}
        x = T(s, torch.float64)
        for k in search:
            t2, t1 = table[:, 156 * k - 156:156 * k - 12].reshape(-1), table[:, 156 * k - 12:156 * k].reshape(-1)
            up = quad_d * ((x + k - target_d) ** 2 - (x - target_d) ** 2)
            down = torch.where(x - k >= 1, quad_d * ((x - k - target_d) ** 2 - (x - target_d) ** 2), torch.inf)
            isok = torch.stack([up, down], 1).reshape(-1)          # index 2k+b
            cost = torch.where(tab >= 0, t2[tab.clamp_min(0)], 0.0) + torch.where(out >= 0, isok[out.clamp_min(0)], 0.0)
            cost = cost.masked_fill(banned[k], torch.inf)

            # +-2 paths: start by leaving the patch at the other end of a node's subside, end by entering its patch
            def single_cost(occ):
                r = occ_row_d[occ]
                return torch.where(r >= 0, t1[r.clamp_min(0) * 12 + occ_side_d[occ] * 2 + node_b], 0.0)
            if k == 1:
                dist = torch.zeros(hub + 1, dtype=torch.float64, device=dev)
            else:
                dist = torch.cat([single_cost(node_occ ^ 1) + isok[2 * (node_occ // 2) + node_b], torch.full((1,), torch.inf, dtype=torch.float64, device=dev)])
            # Jacobi Bellman-Ford, stopped early once the parent graph has cycles, which are then negative
            parent, on = ck.bellman_ford(indptr_d, tail_d, cost, dist, hub + 1 if k == 1 else 64)
            par = parent.cpu().numpy()
            seen, walks = set(), []
            for v in torch.nonzero(on)[:, 0].tolist():
                cyc = []
                while v not in seen:
                    seen.add(v)
                    cyc.append(v)
                    v = tail_list[par[v]]
                if cyc:
                    walks.append(cyc[::-1])
            if k == 2:
                total = dist[:hub] + single_cost(node_occ)
                neg = torch.nonzero(total < -1e-9)[:, 0]
                pl = par.tolist()
                for v in neg[total[neg].argsort()].tolist()[:4096]:
                    path, seen = [v], {v}
                    while pl[v] >= 0:
                        v = tail_list[pl[v]]
                        if v in seen:
                            break
                        seen.add(v)
                        path.append(v)
                    else:
                        walks.append(path[::-1])
            for w in walks:
                moves, arcs = moves_of(w, k, par)
                cands += moves
                bans[k] += arcs
        gains = exact_gains(cands, s, S)
        used = set()
        for i in np.argsort(gains):
            if gains[i] >= -1e-9:
                break
            pts = {occ_row[o] for k in cands[i] for o in (2 * k, 2 * k + 1) if occ_row[o] >= 0}
            if pts & used:
                continue
            used |= pts
            for k, dk in cands[i].items():
                s[k] += dk
        dirty = np.array(sorted(used))
        if used:
            for k in banned:
                banned[k].zero_()
            search = (1, 2)
        elif bans[1] or bans[2]:
            # a search whose bans did not change would find the same non-improving moves again
            search = tuple(k for k in (1, 2) if bans[k])
            for k in search:
                banned[k][T(bans[k])] = True
        else:
            break
    return s
