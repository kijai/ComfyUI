"""Quad remeshing after QuadWild (Pietroni et al. 2021, "Reliable Feature-Line Driven Quad-Remeshing") with its Bi-MDF
quantization (Heistermann et al. 2023): a curvature-aligned cross field, a patch layout traced along it, integer
subdivisions of the patch sides, quad patterns filling every patch, and a final smoothing."""
from __future__ import annotations

import math

import comfy.utils
import comfy_kitchen as ck
import numpy as np
import torch
from tqdm import tqdm

from ..qem_decimate import _repair_nonmanifold_edges, _split_touching_sheets, qem_simplify
from ..remesh import remesh_narrow_band_dc
from .charts import compute_chart_data
from .field import cross_field, opposite_halfedges
from .patterns import pattern
from .quadrangulate import quadrangulate, smooth
from .quantize import quantize
from .tracer import TRACE_STEPS, trace_patches

# quads per (area / edge length^2) of QuadWild layouts, the first guess before the count is measured
_QUADS_PER_EDGE_AREA = 3.5
# fewest quads the patterns of a 3/4/5/6-sided patch take
_MIN_QUADS = {3: 3, 4: 1, 5: 5, 6: 6}
# faces of the mesh the layout is traced on: at least this many, and this many per target quad
_WORK_FACES, _WORK_FACES_PER_QUAD = 170_000, 8
# Remesh Mesh (solid) resolution for inputs that are not closed manifold surfaces
_SOLID_RESOLUTION = 256


def _quads(P, F, label, corners, size, verts, faces, target_quads, alpha, tick):
    """Quads of the traced layout for the relative quad size per vertex: integer subdivisions with the count fitted to
    target_quads, the patterns, and the fit to the input surface."""
    dev = verts.device
    area = 0.5 * np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1)
    edge = math.sqrt(_QUADS_PER_EDGE_AREA * (area / size[F].mean(1) ** 2).sum() / target_quads)
    charts, subsides = compute_chart_data(P, F, label, corners, size)
    s = quantize(charts, subsides, edge, alpha=alpha, device=dev)
    tick()
    # the quad count goes as a / edge^2 above the floor b every patch needs: fit a from the last count, solve again
    floor = sum(_MIN_QUADS.get(len(c.sides), 0) for c in charts if len(c.faces))
    for _ in range(2):
        pats = [pattern(tuple(int(sum(s[k] for k in side.subsides)) for side in c.sides)) for c in charts if len(c.faces)]
        count = sum(len(pat[1]) for pat in pats if pat is not None)
        if count <= floor or target_quads <= floor or abs(count - target_quads) < 0.01 * target_quads:
            break
        scaled = edge * math.sqrt((count - floor) / (target_quads - floor))
        s = quantize(charts, subsides, scaled, alpha=alpha, device=dev, start=np.maximum(1, np.rint(s * edge / scaled)))
        edge = scaled
    tick()
    qv, qf = quadrangulate(P, F, charts, subsides, s, device=dev)
    tick()
    # the final fit goes to the input surface, not the tracer's smoothed copy of it, which has already lost its tips
    qv = smooth(qv, qf, verts, faces)
    tick()
    return qv, qf


def _error_size(P, F, qv, qf, target_quads, adaptive):
    """Relative quad size per vertex for a second pass: (distance of the surface to the first pass's quads, averaged over
    about a quad, over its median) ** (-0.7 adaptive), kept within 0.5 ** (2 adaptive) .. 1.5 ** (2 adaptive)."""
    quads = qv.float()[qf][:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3, 3)
    err = ck.closest_point_on_mesh(ck.mesh_bvh(quads), torch.as_tensor(P, dtype=torch.float32, device=qv.device))[0].double().cpu().numpy()
    area = 0.5 * np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1).sum()
    a, b = F.reshape(-1), np.roll(F, -1, 1).reshape(-1)
    deg = np.bincount(a, minlength=len(P))
    # n neighbour averages spread over about sqrt(n) edges
    for _ in range(int(min(200, area / target_quads / np.linalg.norm(P[b] - P[a], axis=1).mean() ** 2))):
        err = (err + np.bincount(a, weights=err[b], minlength=len(P))) / (1 + deg)
    return np.clip((np.maximum(err, 1e-12) / np.median(err)) ** (-0.7 * adaptive), 0.5 ** (2 * adaptive), 1.5 ** (2 * adaptive))


def _working_mesh(verts, faces, target_quads):
    """The surface the quads are fitted to and the mesh the layout is traced on: inputs with open or non-manifold edges
    are first remeshed as Remesh Mesh (solid) does, a surface far denser than the layout needs is decimated for tracing,
    and sheets touching at a vertex are split apart."""
    if bool((opposite_halfedges(faces) < 0).any()):
        v, f, _ = remesh_narrow_band_dc(verts.float(), faces, resolution=_SOLID_RESOLUTION, sign_mode="solid", qef=False,
                                        drop_inverted_components=False, drop_enclosed_components=False)
        verts, faces = v.double(), f.long()
    work_v, work_f = verts, faces
    limit = max(_WORK_FACES, _WORK_FACES_PER_QUAD * target_quads)
    if faces.shape[0] > 2 * limit:
        v, f, _ = qem_simplify(verts.float(), faces, limit)
        work_v, work_f = v.double(), f.long()
    work_f, src = _split_touching_sheets(work_v, work_f)
    work_v, work_f, _ = _repair_nonmanifold_edges(work_v[src], work_f)
    return verts, faces, work_v, work_f


def quad_remesh(verts: torch.Tensor, faces: torch.Tensor, target_quads: int, alpha: float = 0.05,
                adaptive_size: float = 0.5):
    """Quad mesh (vertices (V, 3) float64, quads (Q, 4)) of a triangle mesh (see _working_mesh), on its device;
    adaptive_size (0..1) re-quantizes the layout with smaller quads where a first pass at even size misses the surface
    and larger ones where it fits."""
    # progress: the field, the tracer's steps, then quantization, the count fit, quadrangulation and the fit per pass
    steps = 1 + TRACE_STEPS + (8 if adaptive_size > 0 else 4)
    pbar, tq = comfy.utils.ProgressBar(steps), tqdm(total=steps, desc="Quad remesh", leave=False)

    def tick():
        pbar.update(1)
        tq.update(1)

    verts, faces, work_v, work_f = _working_mesh(verts, faces, target_quads)
    tri = work_v[work_f]
    total = float(torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=1).sum()) / 2
    # singularity pairs closer than about a quad edge only cost patches: the field cancels them
    field = cross_field(work_v, work_f, math.sqrt(_QUADS_PER_EDGE_AREA * total / target_quads))
    tick()
    P, F, label, corners = trace_patches(work_v, work_f, field, progress_callback=tick)
    qv, qf = _quads(P, F, label, corners, np.ones(len(P)), verts, faces, target_quads, alpha, tick)
    if adaptive_size > 0:
        size = _error_size(P, F, qv, qf, target_quads, adaptive_size)
        qv, qf = _quads(P, F, label, corners, size, verts, faces, target_quads, alpha, tick)
    tq.close()
    return qv, qf
