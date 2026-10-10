"""Fill every patch with its quad pattern, a port of QuadRetopology's quadrangulate: the patch is mapped onto the pattern's
regular polygon with its border fixed by arc length per subside (LSCM with a fully fixed border is the cotangent-harmonic
map, solved for all patches at once), pattern vertices are located in that map and lifted to 3D, shared subside vertices
are stitched, doublets and degenerate faces removed and the result projected back onto the triangle mesh. smooth() is
QuadWild's final MultiCostraintSmooth."""
from __future__ import annotations

import numpy as np
import torch
import comfy_kitchen as ck

from .field import RowSum, opposite_halfedges, row_sum
from .patterns import pattern


def _harmonic(verts3d, tri, fixed, uv):
    """uv (given on `fixed`, filled in place) with L uv = 0 on the free vertices for the cotangent Laplacian L: conjugate
    gradients over the free vertices, the fixed ones moved to the right-hand side."""
    n = verts3d.shape[0]
    p = verts3d[tri]
    u, v = p.roll(-1, 1) - p, p.roll(-2, 1) - p
    cot = 0.5 * ((u * v).sum(2) / torch.linalg.cross(u, v).norm(dim=2).clamp_min(1e-20))
    c1, c2 = tri.roll(-1, 1), tri.roll(-2, 1)
    # both directions of every edge, corner by corner
    i, j, w = (torch.stack([x.T, y.T], 1).reshape(-1) for x, y in ((c1, c2), (c2, c1), (cot, cot)))
    diag = row_sum(i, w, n)
    free = ~fixed
    nfree = int(free.sum())
    at = torch.cumsum(free, 0) - 1                                    # index among the free vertices
    both = free[i] & free[j]
    rows, order = torch.sort(torch.cat([at[free], at[i[both]]]), stable=True)
    cols = torch.cat([at[free], at[j[both]]])[order]
    vals = torch.cat([diag[free], -w[both]])[order]
    indptr = torch.cat([rows.new_zeros(1), torch.cumsum(torch.bincount(rows, minlength=nfree), 0)])
    edge = free[i] & fixed[j]
    b = row_sum(at[i[edge]], w[edge, None] * uv[j[edge]], nfree)
    y = torch.zeros_like(b)
    ck.conjugate_gradient(indptr, cols, vals, b, y, 5000, 1e-10)
    uv[free] = y
    return uv


def quadrangulate(verts: np.ndarray, faces: np.ndarray, charts, subsides, subdiv: np.ndarray, device="cuda"):
    """Quad mesh (vertices (V, 3) float64, quads (Q, 4), on the device) of the patterns of every chart, stitched along
    the subsides and projected onto the triangle mesh."""
    dev = torch.device(device)
    # patterns and chart-local vertices
    pats, local_tris, local_vids, uv_fixed_idx, uv_fixed_val, base = [], [], [], [], [], 0
    for chart in charts:
        pat = pattern(tuple(int(sum(subdiv[s] for s in side.subsides)) for side in chart.sides))
        if pat is None:
            continue
        vids, inv = np.unique(faces[chart.faces], return_inverse=True)
        local_tris.append(inv.reshape(-1, 3) + base)
        local_vids.append(vids)
        walk = []                                       # (subside, reversed, its pattern vertices) along the border
        for i, side in enumerate(chart.sides):
            start = 0
            for s, rev in zip(side.subsides, side.reversed):
                sub, n, d = subsides[s], int(subdiv[s]), -1 if rev else 1
                ids = pat[2][i][start:start + n + 1]
                walk.append((s, rev, ids))
                a, b = pat[0][ids[0]], pat[0][ids[-1]]
                acc = np.concatenate([[0.0], np.cumsum(sub.steps[::d])])[:-1] / max(sub.length, 1e-30)     # spaced by the quad size
                uv_fixed_idx.append(np.searchsorted(vids, sub.vertices[::d][:-1]) + base)
                uv_fixed_val.append(a + (b - a) * acc[:, None])
                start += n
        pats.append((pat, walk))
        base += len(vids)

    T = lambda x, dt=torch.float64: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    tri = T(np.concatenate(local_tris), torch.long)
    v3 = T(verts[np.concatenate(local_vids)])
    idx = T(np.concatenate(uv_fixed_idx), torch.long)
    fixed = torch.zeros(base, dtype=torch.bool, device=dev)
    fixed[idx] = True
    uv = torch.zeros(base, 2, dtype=torch.float64, device=dev)
    uv[idx] = T(np.concatenate(uv_fixed_val))
    uv = _harmonic(v3, tri, fixed, uv)

    # locate pattern vertices: each chart's uv triangles live at their own height
    lift = 4.0 * T(np.repeat(np.arange(len(local_tris)), [len(t) for t in local_tris]))
    bvh = ck.mesh_bvh(torch.cat([uv[tri], lift[:, None, None].expand(-1, 3, 1)], 2).float())
    q = T(np.concatenate([np.c_[pat[0], np.full(len(pat[0]), 4.0 * k)] for k, (pat, _) in enumerate(pats)])).float()
    _, cp, hit = ck.closest_point_on_mesh(bvh, q)
    t2 = uv[tri[hit]]
    d0, d1, d2 = t2[:, 1] - t2[:, 0], t2[:, 2] - t2[:, 0], cp[:, :2].double() - t2[:, 0]
    den = d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0]
    den = torch.where(den.abs() > 1e-300, den, 1.0)
    bu = (d2[:, 0] * d1[:, 1] - d2[:, 1] * d1[:, 0]) / den
    bv = (d0[:, 0] * d2[:, 1] - d0[:, 1] * d2[:, 0]) / den
    bary = torch.stack([1 - bu - bv, bu, bv], 1).clamp(0, 1)
    bary = bary / bary.sum(1, keepdim=True).clamp_min(1e-30)
    mapped = (bary[:, :, None] * v3[tri[hit]]).sum(1).cpu().numpy()

    # stitch: patch corners first, then each subside's inner vertices once (averaged over its two patches), then the
    # patch interiors
    corner_vertex = {v: k for k, v in enumerate(dict.fromkeys(v for sub in subsides for v in (sub.vertices[0], sub.vertices[-1])))}
    out_v = [verts[v] for v in corner_vertex]
    out_f, subside_verts, qoff = [], {}, 0
    for (ppos, pfaces, _), walk in pats:
        vmap = np.full(len(ppos), -1, dtype=np.int64)
        for s, rev, ids in walk:
            n = len(ids) - 1
            if s not in subside_verts:
                sv = subsides[s].vertices[::-1] if rev else subsides[s].vertices
                chain = [corner_vertex[sv[0]]] + list(range(len(out_v), len(out_v) + n - 1)) + [corner_vertex[sv[-1]]]
                out_v += [mapped[qoff + k] for k in ids[1:n]]
                subside_verts[s] = chain[::-1] if rev else chain
            else:
                chain = subside_verts[s][::-1] if rev else subside_verts[s]
                for k in range(1, n):
                    out_v[chain[k]] = (out_v[chain[k]] + mapped[qoff + ids[k]]) / 2
            vmap[ids] = chain
        inner = np.nonzero(vmap < 0)[0]
        vmap[inner] = np.arange(len(out_v), len(out_v) + len(inner))
        out_v += list(mapped[qoff + inner])
        out_f.append(vmap[pfaces])
        qoff += len(ppos)
    qv, qf = np.array(out_v), np.concatenate(out_f)
    qf = _remove_doublets(qf[np.array([len(set(f)) == 4 for f in qf.tolist()], dtype=bool)])
    used = np.zeros(len(qv), dtype=bool)
    used[qf.reshape(-1)] = True
    qv, qf = qv[used], (np.cumsum(used) - 1)[qf]
    tbvh = ck.mesh_bvh(T(verts).float()[T(faces, torch.long)])
    return ck.closest_point_on_mesh(tbvh, T(qv).float())[1].double(), T(qf, torch.long)


def _remove_doublets(qf):
    """Merge pairs of quads that share both edges at a vertex no other quad uses (QuadRetopology removeDoubletFaces,
    repeated until none is left): the first quad takes the second one's corner opposite that vertex. Edits qf in place."""
    alive = np.ones(len(qf), dtype=bool)
    merged = True
    while merged:
        fi = np.nonzero(alive)[0]
        corners = qf[fi].reshape(-1)
        at = np.nonzero(np.bincount(corners)[corners] == 2)[0]
        merged = False
        # the two corners (flat face*4+k) at each such vertex
        for a, b in at[np.argsort(corners[at], kind="stable")].reshape(-1, 2).tolist():
            f, g, j, k = fi[a // 4], fi[b // 4], a % 4, b % 4
            if alive[f] and alive[g] and {qf[f][j - 1], qf[f][(j + 1) % 4]} == {qf[g][k - 1], qf[g][(k + 1) % 4]}:
                qf[f][j] = qf[g][(k + 2) % 4]
                alive[g] = False
                merged = True
    return qf[alive]


def _quad_templates(p):
    """Square of each quad's area rigidly fitted to it in its PCA plane, then turned to the quad's normal (vcg
    GetRotatedTemplatePos with isotropic GetPolyTemplatePos), the distortion sum|template - p|^2 / area, and the area."""
    def normal(q):
        n = torch.linalg.cross(q.roll(-1, 1) - q, q.roll(-2, 1) - q)
        n = (n / n.norm(dim=2, keepdim=True).clamp_min(1e-30)).sum(1)
        return n / n.norm(dim=1, keepdim=True).clamp_min(1e-30)
    b = p.mean(1, keepdim=True)
    d = p - b
    axes = torch.linalg.eigh(d.transpose(1, 2) @ d)[1].flip(2)            # columns: major, minor, normal
    u = d @ axes
    area = lambda q: 0.5 * torch.linalg.cross(q, q.roll(-1, 1)).norm(dim=2).sum(1)
    scale = (2.0 / area(u).clamp_min(1e-30)).sqrt()[:, None, None]
    t = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=p.dtype, device=p.device).expand(len(p), 4, 2)
    t = torch.where((normal(u)[:, 2] < 0)[:, None, None], t.flip(1), t)
    uu = u[..., :2] * scale
    c = (t * uu).sum((1, 2))
    s = (t[..., 0] * uu[..., 1] - t[..., 1] * uu[..., 0]).sum(1)
    ang = torch.atan2(s, c)
    rot = torch.stack([torch.stack([ang.cos(), -ang.sin()], 1), torch.stack([ang.sin(), ang.cos()], 1)], 1)
    tp = (t @ rot.transpose(1, 2)) / scale
    tmpl = b + tp[..., :1] * axes[:, None, :, 0] + tp[..., 1:] * axes[:, None, :, 1]
    pa = area(d)
    quality = ((tmpl - p) ** 2).sum((1, 2)) / pa.clamp_min(1e-30)
    a, n = normal(tmpl), normal(p)
    w = torch.linalg.cross(a, n)[:, None]
    k = 1.0 / (1.0 + (a * n).sum(1)).clamp_min(1e-12)
    v = tmpl - b
    wv = torch.linalg.cross(w.expand_as(v), v)
    return b + v + wv + k[:, None, None] * torch.linalg.cross(w.expand_as(v), wv), quality, pa


def _back_projection(q, f, tri_v):
    """Closest point of every triangle mesh vertex on the quads (fanned around their centres), each pulling the corners
    of its quad weighted by its position in it: the mean pull per quad vertex and which vertices are pulled at all."""
    p = q[f]
    centre = p.mean(1, keepdim=True).expand(-1, 4, 3)
    fan = torch.stack([p, p.roll(-1, 1), centre], 2).reshape(-1, 3, 3)
    _, cp, hit = ck.closest_point_on_mesh(ck.mesh_bvh(fan.float()), tri_v)
    tri = fan[hit]
    cp = cp.double()
    v0, v1, v2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], cp - tri[:, 0]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = (d00 * d11 - d01 * d01).clamp_min(1e-30)
    b1, b2 = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
    bary = torch.stack([1 - b1 - b2, b1, b2], 1).clamp_min(0)
    quad, corner = hit // 4, hit % 4
    one_hot = torch.nn.functional.one_hot
    wq = bary[:, 2:3] / 4 + bary[:, :1] * one_hot(corner, 4) + bary[:, 1:2] * one_hot((corner + 1) % 4, 4)
    wq = wq / wq.sum(1, keepdim=True).clamp_min(1e-30)
    by_vertex = RowSum(f[quad].reshape(-1), len(q))
    num = by_vertex(((tri_v.double() - cp)[:, None] * wq[..., None] ** 2).reshape(-1, 3))
    wsum = by_vertex(wq.reshape(-1))
    return num / wsum[:, None].clamp_min(1e-30), wsum > 0


def smooth(q, f, verts, faces):
    """QuadWild's MultiCostraintSmooth on a mesh without sharp features: 30 steps that each move every vertex halfway
    towards 0.7 * the distortion-weighted average of its quads' fitted squares + 0.3 * the area-weighted average of its
    quads' other corners (vcg SmoothPCA), then back project the triangle mesh onto the quads and reproject onto the
    triangle mesh. Then 3 back projections along the vertex normals only and without the reprojection, so the quads fit
    the surface instead of interpolating it at their corners. Border vertices stay put. Tensors on one device."""
    dev = q.device
    tri_v = verts.float()
    surf = ck.mesh_bvh(tri_v[faces])
    nv = len(q)
    fixed = torch.zeros(nv, 1, dtype=torch.bool, device=dev)
    fixed[f.reshape(-1)[opposite_halfedges(f) < 0]] = True
    by_vertex = RowSum(f.reshape(-1), nv)
    for _ in range(30):
        p = q[f]
        tmpl, quality, area = _quad_templates(p)
        w = (quality + 1e-5)[:, None].expand(-1, 4).reshape(-1)
        fit = by_vertex(tmpl.reshape(-1, 3) * w[:, None]) / by_vertex(w)[:, None].clamp_min(1e-30)
        others = (p.sum(1, keepdim=True) - p) * area[:, None, None]
        lap = by_vertex(others.reshape(-1, 3)) / by_vertex(3 * area[:, None].expand(-1, 4).reshape(-1))[:, None].clamp_min(1e-30)
        q = torch.where(fixed, q, 0.5 * q + 0.5 * (0.7 * fit + 0.3 * lap))
        pull, pulled = _back_projection(q, f, tri_v)
        q = torch.where(fixed | ~pulled[:, None], q, q + pull)
        q = torch.where(fixed, q, ck.closest_point_on_mesh(surf, q.float())[1].double())
    for _ in range(3):
        pull, pulled = _back_projection(q, f, tri_v)
        p = q[f]
        n = by_vertex(torch.linalg.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]).repeat_interleave(4, 0))
        n = torch.nn.functional.normalize(n, dim=1, eps=1e-30)
        q = torch.where(fixed | ~pulled[:, None], q, q + (pull * n).sum(1, keepdim=True) * n)
    return q
