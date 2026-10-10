"""Curvature-aligned 4-RoSy cross field, a PyTorch port of QuadWild's organic field setup (libigl principal_curvature with a
3-ring quadric fit, anisotropy-weighted soft constraints, libigl/CoMISo mixed-integer N-RoSy smoothing). CoMISo rounds the
integer period jumps greedily in tiny batches; here all jumps under a rising residue threshold are rounded together and the
angles re-solved with conjugate gradients. Then opposite singularity pairs closer than about a quad are cancelled: below
the quad size they only cost patches and irregular vertices."""
from __future__ import annotations

import math
from typing import Tuple

import comfy_kitchen as ck
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra

HALF_PI = math.pi / 2
# angle deficit within a vertex's 2-ring above which the vertex is the point of a spike
TIP_DEFICIT = math.radians(150)
# vertices whose k-rings are expanded and fitted at once (about 2 KB each for 3-rings)
RING_CHUNK = 65536


class RowSum:
    """Sums of values (along the first dimension) by row index in a fixed order: sorted by index, each row's values added
    in sequence. index_add_ on CUDA adds in thread order, so its float sums vary in the last bits from run to run, which
    the patch layout turns into different quads. Built once per index for repeated sums."""

    def __init__(self, index: torch.Tensor, rows: int):
        self.order = torch.argsort(index, stable=True)
        self.lengths = torch.bincount(index, minlength=rows)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return torch.segment_reduce(values[self.order], "sum", lengths=self.lengths, axis=0)


def row_sum(index: torch.Tensor, values: torch.Tensor, rows: int) -> torch.Tensor:
    return RowSum(index, rows)(values)


def _normals(verts: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit face normals and area-weighted unit vertex normals."""
    p = verts[faces]
    fn = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    vn = row_sum(faces.reshape(-1), fn.repeat_interleave(3, 0), len(verts))
    return (torch.nn.functional.normalize(fn, dim=1, eps=1e-30), torch.nn.functional.normalize(vn, dim=1, eps=1e-30))


def _edge_pairs(faces: torch.Tensor):
    """The two half-edges (3f+k: faces[f,k] -> faces[f,k+1]) of every edge of exactly two faces, in edge key order, and
    the half-edges' end vertices; an edge of more than two faces counts as a border."""
    nv = int(faces.max()) + 1
    a, b = faces.reshape(-1), faces.roll(-1, 1).reshape(-1)
    key = torch.minimum(a, b) * nv + torch.maximum(a, b)
    order = torch.argsort(key)
    _, count = torch.unique_consecutive(key[order], return_counts=True)
    pair = (torch.cumsum(count, 0) - count)[count == 2]
    return order[pair], order[pair + 1], a, b


def opposite_halfedges(faces: torch.Tensor) -> torch.Tensor:
    """For half-edge 3f+k the opposite half-edge index, -1 on borders."""
    h0, h1, _, _ = _edge_pairs(faces)
    opp = torch.full((faces.numel(),), -1, dtype=torch.long, device=faces.device)
    opp[h0], opp[h1] = h1, h0
    return opp


def _angle_defects(verts: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Angle deficit (2 pi minus the corner angles around it) per vertex, 0 on the border vertices, and the border mask."""
    c = verts[faces]
    u, w = c.roll(-1, 1) - c, c.roll(-2, 1) - c
    angles = row_sum(faces.T.reshape(-1), torch.atan2(torch.linalg.cross(u, w).norm(dim=2), (u * w).sum(2)).T.reshape(-1), len(verts))
    border = torch.zeros(len(verts), dtype=torch.bool, device=verts.device)
    border[faces.reshape(-1)[opposite_halfedges(faces) < 0]] = True
    return torch.where(border, 0.0, 2 * math.pi - angles), border


def _rings(faces: torch.Tensor, nv: int, rings: int):
    """Sorted keys vertex * nv + member of the k-rings (the vertex itself included) as (lo, hi, keys) per chunk of
    vertices lo..hi."""
    dev = faces.device
    a, b = faces.reshape(-1), faces.roll(-1, 1).reshape(-1)
    ring1 = torch.unique(torch.cat([a * nv + b, b * nv + a, torch.arange(nv, device=dev) * (nv + 1)]))
    start = torch.searchsorted(ring1, torch.arange(nv + 1, device=dev) * nv)
    for lo in range(0, nv, RING_CHUNK):
        hi = min(lo + RING_CHUNK, nv)
        ring = ring1[start[lo]:start[hi]]
        for _ in range(rings - 1):
            i, k = ring // nv, ring % nv
            deg = start[k + 1] - start[k]
            rep = torch.repeat_interleave(torch.arange(len(ring), device=dev), deg)
            off = torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(deg, 0) - deg, deg)
            ring = torch.unique(i[rep] * nv + ring1[start[k[rep]] + off] % nv)
        yield lo, hi, ring


def principal_curvature(verts: torch.Tensor, faces: torch.Tensor, n: torch.Tensor):
    """Per-vertex first principal direction and curvatures k1 >= k2 (all zero with fewer than 6 neighbours) from a
    quadric z = au^2+buv+cv^2+du+ev fitted to the 3-ring in the tangent frame of the vertex normal n. Neighbours whose
    normals face away are dropped when at least 6 remain."""
    nv, dev = verts.shape[0], verts.device
    a, b = faces.reshape(-1), faces.roll(-1, 1).reshape(-1)
    first = torch.full((nv,), nv, dtype=torch.long, device=dev).scatter_reduce_(0, torch.cat([a, b]), torch.cat([b, a]), reduce="amin")
    # frame: x = first edge neighbour projected to the tangent plane, y = n x x
    d = verts[first] - verts
    x = torch.nn.functional.normalize(d - (d * n).sum(1, keepdim=True) * n, dim=1, eps=1e-30)
    y = torch.nn.functional.normalize(torch.linalg.cross(n, x), dim=1, eps=1e-30)
    ata = torch.empty(nv, 5, 5, dtype=verts.dtype, device=dev)
    atb = torch.empty(nv, 5, dtype=verts.dtype, device=dev)
    count = torch.empty(nv, dtype=torch.long, device=dev)
    for lo, hi, ring in _rings(faces, nv, 3):
        ci, cj = ring // nv, ring % nv
        front = (n[cj] * n[ci]).sum(1) > 0
        keep = front | (torch.bincount(ci[front] - lo, minlength=hi - lo) < 6)[ci - lo]
        ci, cj = ci[keep], cj[keep]
        rel = verts[cj] - verts[ci]
        u, v, z = (rel * x[ci]).sum(1), (rel * y[ci]).sum(1), (rel * n[ci]).sum(1)
        phi = torch.stack([u * u, u * v, v * v, u, v], 1)
        by_vertex = RowSum(ci - lo, hi - lo)
        count[lo:hi] = by_vertex.lengths
        ata[lo:hi] = by_vertex(phi[:, :, None] * phi[:, None, :])
        atb[lo:hi] = by_vertex(phi * z[:, None])
    qa, qb, qc, qd, qe = (torch.linalg.pinv(ata) @ atb[:, :, None])[:, :, 0].unbind(1)

    E, Fm, G = 1 + qd * qd, qd * qe, 1 + qe * qe
    nz = 1 / torch.sqrt(qd * qd + qe * qe + 1)
    L, M, N = 2 * qa * nz, qb * nz, 2 * qc * nz
    den = E * G - Fm * Fm
    m = torch.stack([torch.stack([L * G - M * Fm, M * E - L * Fm], 1), torch.stack([M * E - L * Fm, N * E - M * Fm], 1)], 1) / den[:, None, None]
    val, vec = torch.linalg.eigh(m)                     # ascending: negated, index 0 is the larger curvature
    valid = (count >= 6)[:, None]
    k = torch.where(valid, -val, 0.0)
    pd1 = torch.nn.functional.normalize(vec[:, 0, 0, None] * x + vec[:, 1, 0, None] * y, dim=1, eps=1e-30)
    return torch.where(valid, pd1, 0.0), k[:, 0], k[:, 1]


def surface_tips(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Vertices at the point of a spike: the angle deficit summed over the vertex's 2-ring exceeds TIP_DEFICIT and is
    the largest within that 2-ring (curvature concentrated at a point; rounded ends spread it out)."""
    n = len(verts)
    deficit, _ = _angle_defects(verts, faces)
    pair = torch.cat([ring for _, _, ring in _rings(faces, n, 2)])
    a, b = pair // n, pair % n
    ring = row_sum(a, deficit[b], n)
    most = ring.clone().scatter_reduce_(0, a, ring[b], "amax")
    return torch.nonzero((ring > TIP_DEFICIT) & (ring >= most))[:, 0]


def _angle_system(fi, fj, target, soft, smooth):
    """solve(theta, mask, c, iters, tol) for (1-a) sum_e (t_i - t_j + c_e)^2 + a/2 sum_f w_f (t_f - s_f)^2 over the
    rounded edges (a 0/1 mask over all interior edges): conjugate gradients on the device from theta (updated in place),
    fewer than iters iterations once the residual is below tol. The CSR matrix over the faces has a fixed pattern: the
    diagonal, then one entry per interior edge of the face, zero while the edge is not rounded."""
    nf, ne, dev = len(target), len(fi), fi.device
    face, order = torch.sort(torch.cat([fi, fj]), stable=True)
    count = torch.bincount(face, minlength=nf)
    slot = torch.arange(len(face), device=dev) - (torch.cumsum(count, 0) - count)[face]
    edge = torch.zeros(nf, 3, dtype=torch.long, device=dev)          # interior edges around each face
    sign = torch.zeros(nf, 3, dtype=target.dtype, device=dev)        # +1 as its face i, -1 as face j, 0 none
    edge[face, slot] = order % ne
    sign[face, slot] = torch.where(order < ne, 1.0, -1.0).to(target.dtype)
    faces = torch.arange(nf, device=dev)
    row, order = torch.sort(torch.cat([faces, fi, fj]), stable=True)
    cols = torch.cat([faces, fj, fi])[order]
    entry_edge = torch.where(order < nf, -1, (order - nf) % ne)       # -1 on the diagonal
    indptr = torch.cat([count.new_zeros(1), torch.cumsum(count + 1, 0)])

    def solve(theta, mask, c, iters, tol):
        rounded = (mask[edge] * sign.abs()).sum(1)
        vals = torch.where(entry_edge < 0, (soft + smooth * rounded)[row], -smooth * mask[entry_edge.clamp_min(0)])
        b = soft * target + smooth * ((-c * mask)[edge] * sign).sum(1)
        ck.conjugate_gradient(indptr, cols, vals, b, theta, iters, tol)
        return theta
    return solve


def miq_field(solve, fi: torch.Tensor, fj: torch.Tensor, k: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integer period jumps (all rounded) and the face angles of the last rounding solve for the energy
    (1-a) sum_e (t_i - t_j + k_e + pi/2 p_e)^2 + a/2 sum_f w_f (t_f - s_f)^2, from the angles theta. Free jumps follow
    the angles exactly, so only rounded edges enter the angle system."""
    fixed = torch.zeros_like(k, dtype=torch.bool)
    p = torch.zeros_like(k)
    for tau in [1e-4] + [0.5 * (i + 1) / 50 for i in range(50)]:
        while True:
            pc = -(theta[fi] - theta[fj] + k) / HALF_PI
            new = ~fixed & ((pc - torch.round(pc)).abs() <= tau)
            if not bool(new.any()):
                break
            p = torch.where(new, torch.round(pc), p)
            fixed = fixed | new
            theta = solve(theta, fixed.to(theta.dtype), k + HALF_PI * p, 15, 1e-6)
    return theta, p


def cancel_dipoles(verts: torch.Tensor, faces: torch.Tensor, edge: torch.Tensor, k: torch.Tensor, p: torch.Tensor,
                   reach: float) -> torch.Tensor:
    """Period jumps with the interior +1/4 and -1/4 singularities closer than reach along the mesh edges cancelled in
    pairs, closest first: the jumps along the shortest edge path between a pair shift by one, which carries the +1/4
    over to the -1/4 vertex. The index in quarters of a vertex is (angle defect + sum s k_e) / (pi/2) + sum s p_e over its
    edges, s = +1 where the edge leaves the vertex. The +1/4 singularities within the 2-ring of a spike's point stay: the
    tip rings traced around the point need them inside."""
    nv = verts.shape[0]
    deficit, border = _angle_defects(verts, faces)
    a, b = edge[:, 0], edge[:, 1]
    turn = deficit + row_sum(a, k, nv) - row_sum(b, k, nv)
    q = torch.round(turn / HALF_PI).long() + (torch.bincount(a, p, nv) - torch.bincount(b, p, nv)).long()
    keep = torch.zeros(nv, dtype=torch.bool, device=verts.device)
    keep[surface_tips(verts, faces)] = True
    for _ in range(2):
        keep[torch.cat([a[keep[b]], b[keep[a]]])] = True
    plus, minus = (~border & ~keep & (q == 1)).cpu().numpy(), (~border & (q == -1)).cpu().numpy()
    # the pairing runs on the host (scipy's Dijkstra)
    length = np.linalg.norm((verts[b] - verts[a]).cpu().numpy(), axis=1)
    a, b = edge.T.cpu().numpy()
    jumps = p.cpu().numpy().astype(np.int64)
    graph = sp.csr_matrix((np.concatenate([length, length]), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(nv, nv))
    key = np.minimum(a, b) * nv + np.maximum(a, b)                    # ascending: the edges come in key order
    while plus.any() and minus.any():
        dist, pred, src = dijkstra(graph, indices=np.nonzero(plus)[0], limit=reach, return_predecessors=True, min_only=True)
        ends = np.nonzero(minus & np.isfinite(dist))[0]
        if not len(ends):
            break
        tail, head = [], []
        for v in ends[np.argsort(dist[ends], kind="stable")].tolist():
            s = int(src[v])
            if plus[s]:
                plus[s] = minus[v] = False
                while v != s:
                    tail.append(int(pred[v]))
                    head.append(v)
                    v = tail[-1]
        tail, head = np.array(tail), np.array(head)
        e = np.searchsorted(key, np.minimum(tail, head) * nv + np.maximum(tail, head))
        np.add.at(jumps, e, np.where(a[e] == tail, -1, 1))
    return torch.as_tensor(jumps, dtype=p.dtype, device=p.device)


def cross_field(verts: torch.Tensor, faces: torch.Tensor, reach: float) -> torch.Tensor:
    """Per-face unit direction of the cross field (one of its four branches), (F, 3), with opposite singularity pairs
    closer than reach cancelled."""
    verts = verts.double()
    fn, vn = _normals(verts, faces)
    e1 = torch.nn.functional.normalize(verts[faces[:, 1]] - verts[faces[:, 0]], dim=1, eps=1e-30)
    e2 = torch.linalg.cross(fn, e1)
    # soft target angle per face (in the e1, e2 frame): the 4-RoSy average of its vertices' first principal directions
    # turned into the face plane (Rodrigues, vertex normal onto face normal), weighted by the curvature anisotropy
    pd1, k1, k2 = principal_curvature(verts, faces, vn)
    trim = torch.quantile(torch.cat([k1.abs(), k2.abs()]).float(), 0.95).to(k1.dtype).clamp_min(1e-30)
    weight = (((k1 / trim).clamp(-1, 1) - (k2 / trim).clamp(-1, 1)) / 2).clamp_min(1e-5)[faces].mean(1)
    d, n = pd1[faces], vn[faces]
    axis = torch.linalg.cross(n, fn[:, None].expand_as(n))
    sin = axis.norm(dim=2, keepdim=True)
    cos = (n * fn[:, None]).sum(2, keepdim=True)
    r = axis / sin.clamp_min(1e-30)
    d = torch.where(sin > 1e-12, d * cos + torch.linalg.cross(r, d) * sin + r * (r * d).sum(2, keepdim=True) * (1 - cos), d)
    a = torch.atan2((d * e2[:, None]).sum(2), (d * e1[:, None]).sum(2))
    w = torch.polar(torch.ones_like(a), 4 * a)
    target = torch.angle(w[:, 0] + w[:, 1] + w[:, 2]) / 4
    # interior edges as oriented in face fi, each once, in key order
    h0, h1, tail, head = _edge_pairs(faces)
    fi, fj = h0 // 3, h1 // 3
    edge = torch.stack([tail[h0], head[h0]], 1)
    # k_ij such that t_i - t_j + k_ij = 0 for a direction parallel transported across the shared edge
    d = verts[edge[:, 1]] - verts[edge[:, 0]]
    k = torch.atan2((d * e2[fj]).sum(1), (d * e1[fj]).sum(1)) - torch.atan2((d * e2[fi]).sum(1), (d * e1[fi]).sum(1))
    solve = _angle_system(fi, fj, target, 0.3 * weight, 2 * (1 - 0.3))   # soft constraint weight a = 0.3
    theta, p = miq_field(solve, fi, fj, k, target.clone())
    p = cancel_dipoles(verts, faces, edge, k, p, reach)
    theta = solve(theta, torch.ones_like(k), k + HALF_PI * p, 1000, 1e-9)
    return theta.cos()[:, None] * e1 + theta.sin()[:, None] * e2
