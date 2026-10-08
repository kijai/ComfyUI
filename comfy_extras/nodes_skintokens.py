"""SkinTokens (TokenRig) automatic rigging: mesh -> skeleton + skinning weights, saved as a skinned GLB."""
import logging
from io import BytesIO

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
from typing_extensions import override

import comfy.model_management
import comfy.sd
import folder_paths
from comfy.ldm.skintokens.tokenizer import CLS
from comfy_api.latest import ComfyExtension, IO, Types
from comfy_extras.nodes_save_3d import get_mesh_batch_item, mesh_item_to_glb_bytes

SkinTokensModel = IO.Custom("SKINTOKENS_MODEL")
PoseData = IO.Custom("KIMODO_POSE_DATA")

NUM_SAMPLES = 54000
NUM_VERTEX_SAMPLES = 16384
# glTF is Y-up; the model was trained on Blender-imported (Z-up) assets
GLTF_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def unitize(x):
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, length, out=np.zeros_like(x), where=length > 1e-13)


def mesh_normals(vertices, faces):
    """trimesh's face normals and corner-angle weighted vertex normals."""
    tris = vertices[faces]
    face_normals = unitize(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]))
    u = unitize(tris[:, 1] - tris[:, 0])
    v = unitize(tris[:, 2] - tris[:, 0])
    w = unitize(tris[:, 2] - tris[:, 1])
    angles = np.empty((len(faces), 3))
    angles[:, 0] = np.arccos(np.clip((u * v).sum(axis=1), -1, 1))
    angles[:, 1] = np.arccos(np.clip((-u * w).sum(axis=1), -1, 1))
    angles[:, 2] = np.pi - angles[:, 0] - angles[:, 1]
    angles[(angles < 1e-8).any(axis=1)] = 0.0

    valid = (face_normals ** 2).sum(axis=1) > 0.5
    corner_faces = faces[valid].ravel()
    contrib = (angles[valid][..., None] * face_normals[valid][:, None]).reshape(-1, 3)
    vertex_normals = np.stack([np.bincount(corner_faces, weights=contrib[:, i], minlength=len(vertices)) for i in range(3)], axis=1)
    return face_normals, unitize(vertex_normals)


def sample_points(vertices, faces, rng):
    """Reference "mix" sampler: random mesh vertices plus area-weighted surface points, with normals."""
    face_normals, vertex_normals = mesh_normals(vertices, faces)
    perm = rng.permutation(len(vertices))[:NUM_VERTEX_SAMPLES]
    num_surface = NUM_SAMPLES - len(perm)

    tris = vertices[faces]
    origins = tris[:, 0]
    edges = tris[:, 1:] - origins[:, None]
    area = np.cumsum(np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1))
    face_index = np.searchsorted(area, rng.random(num_surface) * area[-1])
    lengths = rng.random((num_surface, 2, 1))
    fold = lengths.sum(axis=1)[:, 0] > 1.0
    lengths[fold] -= 1.0
    surface = origins[face_index] + (edges[face_index] * np.abs(lengths)).sum(axis=1)

    points = np.concatenate((vertices[perm], surface))
    normals = np.concatenate((vertex_normals[perm], face_normals[face_index]))
    return torch.from_numpy(points.astype(np.float32)), torch.from_numpy(normals.astype(np.float32))


def transfer_skin(vertices, points, skin, k=8, chunk=65536):
    """Inverse-distance blend of the nearest sampled points onto the mesh vertices."""
    k = min(k, len(points))
    dist, idx = cKDTree(points).query(vertices, k=k, workers=-1)
    if k == 1:
        return skin[idx]
    weights = 1.0 / (dist + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.empty((len(vertices), skin.shape[1]), dtype=np.float32)
    for s in range(0, len(vertices), chunk):
        out[s:s + chunk] = np.einsum("nk,nkj->nj", weights[s:s + chunk], skin[idx[s:s + chunk]])
    return out


def geodesic_falloff(vertices, faces, joints, alpha=0.5, link_dis=1e-5):
    """The reference's voxel_skin postprocess on the seam-welded surface (its voxel grid never links): (V, J) falloff."""
    welded, weld = np.unique(vertices, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    edges = np.sort(weld[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
    tree = cKDTree(welded)
    dist, idx = tree.query(welded, 4, workers=-1)
    dist, idx = dist[:, 1:], idx[:, 1:]
    close = (0 < dist) & (dist < link_dis)
    src = np.concatenate([np.repeat(np.arange(len(welded)), 3)[close.ravel()], edges[:, 0]])
    dst = np.concatenate([idx[close], edges[:, 1]])
    weight = np.concatenate([dist[close], np.linalg.norm(welded[edges[:, 0]] - welded[edges[:, 1]], axis=-1)])
    graph = csr_matrix((weight, (src, dst)), shape=(len(welded), len(welded)))
    dis = shortest_path(graph, method="D", directed=False, indices=tree.query(joints)[1])
    unreachable = np.isinf(dis).all(axis=0)
    k = min(len(joints), 3)
    near_dis, near_joint = cKDTree(joints).query(welded[unreachable], k, workers=-1)
    dis[near_joint.reshape(-1, k), np.repeat(np.flatnonzero(unreachable), k).reshape(-1, k)] = near_dis.reshape(-1, k)
    dis = dis.astype(np.float32)  # (J, V) is the big one; the rest runs in place
    finite = np.isfinite(dis)
    dis[~finite] = dis.max(where=finite, initial=0.0)
    np.maximum(dis, 1e-6, out=dis)
    dis *= alpha * dis + (1 - alpha)
    np.reciprocal(dis, out=dis)
    np.square(dis, out=dis)
    dis /= dis.sum(axis=0)
    return dis.T[weld]


def top_influences(skin, k=4):
    """Largest k weights per vertex renormalized to sum to one, padded to k for glTF."""
    k_used = min(k, skin.shape[1])
    indices = np.argsort(-skin, axis=1)[:, :k_used]
    weights = np.take_along_axis(skin, indices, axis=1)
    total = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    weights[total[:, 0] == 0, 0] = 1.0  # all weights underflowed: bind to the strongest joint
    pad = ((0, 0), (0, k - k_used))
    return np.pad(indices, pad), np.pad(weights, pad)


class LoadSkinTokensModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LoadSkinTokensModel",
            display_name="Load SkinTokens Model",
            category="model/loaders",
            inputs=[
                IO.Combo.Input("model_name", options=folder_paths.get_filename_list("text_encoders")),
            ],
            outputs=[SkinTokensModel.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> IO.NodeOutput:
        return IO.NodeOutput(comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("text_encoders", model_name)))


class SkinTokensRig(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SkinTokensRig",
            display_name="Auto Rig Mesh (SkinTokens)",
            search_aliases=["rig", "rigging", "skeleton", "skinning", "armature", "tokenrig"],
            category="3d",
            description="Generate a skeleton and skinning weights for a mesh with SkinTokens / TokenRig and output "
                        "the rigged mesh as a skinned GLB, and the rig in its rest pose for animation nodes. "
                        "The mesh is expected Y-up (glTF convention). Uses the first mesh of a batch.",
            inputs=[
                SkinTokensModel.Input("model"),
                IO.Mesh.Input("mesh"),
                IO.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                IO.Combo.Input("skeleton_type", options=list(CLS), default="articulation",
                               tooltip="Dataset style of the skeleton: articulation (general assets), vroid (humanoid) or rignet."),
                IO.Int.Input("num_beams", default=10, min=1, max=32, advanced=True),
                IO.Boolean.Input("do_sample", default=True, advanced=True,
                                 tooltip="Sample candidates instead of a deterministic beam search."),
                IO.Int.Input("top_k", default=5, min=0, max=1000, advanced=True),
                IO.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01, advanced=True),
                IO.Float.Input("temperature", default=1.0, min=0.01, max=5.0, step=0.05, advanced=True),
                IO.Float.Input("repetition_penalty", default=2.0, min=0.5, max=5.0, step=0.05, advanced=True),
                IO.Boolean.Input("skin_cleanup", default=True,
                                 tooltip="Fade each bone's weights with the distance along the mesh surface, as the reference's postprocess "
                                         "does. Stops weights leaking between nearby parts, e.g. a held prop following the head."),
            ],
            outputs=[IO.File3DGLB.Output(display_name="model_3d"), PoseData.Output(display_name="rig")],
        )

    @classmethod
    def execute(cls, model, mesh, seed, skeleton_type, num_beams, do_sample, top_k, top_p, temperature, repetition_penalty, skin_cleanup) -> IO.NodeOutput:
        if mesh.vertices.shape[0] > 1:
            logging.warning("SkinTokensRig uses the first mesh of a batch of %d.", mesh.vertices.shape[0])
        mesh_vertices, faces = get_mesh_batch_item(mesh, 0)[:2]
        mesh_vertices = mesh_vertices.cpu().double().numpy()
        vertices = mesh_vertices @ GLTF_TO_ZUP.T
        faces = faces.cpu().long().numpy()

        # normalize into [-1, 1] like the reference "affine" transform
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        center = (lo + hi) / 2
        scale = (hi - lo).max() / 2
        vertices = (vertices - center) / scale
        points, normals = sample_points(vertices, faces, np.random.default_rng(seed))

        comfy.model_management.load_model_gpu(model)
        joints, parents, skin = model.model.diffusion_model.rig(points, normals, model.load_device, model.model.get_dtype_inference(), cls=skeleton_type, seed=seed,
                                                num_beams=num_beams, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                                temperature=temperature, repetition_penalty=repetition_penalty)

        skin = transfer_skin(vertices, points.numpy(), skin)
        if skin_cleanup:
            skin = skin * geodesic_falloff(vertices, faces, joints)
        indices, weights = top_influences(skin)
        joints = (joints * scale + center) @ GLTF_TO_ZUP
        glb = mesh_item_to_glb_bytes(mesh, 0, skin={"joints": joints, "parents": parents, "indices": indices, "weights": weights})

        bind = np.zeros((len(parents), 8), dtype=np.float32)
        bind[:, :3] = joints
        bind[:, 6:] = 1.0  # identity rotation (xyzw), unit scale
        faces = faces.astype(np.uint32)
        rig = {
            "frames": [[{"pred_global_rots": np.tile(np.eye(3, dtype=np.float32), (len(parents), 1, 1)), "pred_joint_coords": bind[:, :3]}]],
            "faces": faces,
            "_skeleton_override": {
                "parents": np.asarray(parents, dtype=np.int32),
                "bind_global_m": bind,
                "lbs_compact_joints": np.pad(indices, ((0, 0), (0, 4))).astype(np.uint16),
                "lbs_compact_weights": np.pad(weights, ((0, 0), (0, 4))).astype(np.float32),
                "lbs_compact_max_inf": 4,
                "rest_verts_m": mesh_vertices.astype(np.float32),
                "faces": faces,
                "per_frame_y_down": False,
            },
        }
        return IO.NodeOutput(Types.File3D(BytesIO(glb), file_format="glb"), rig)


class SkinTokensExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [LoadSkinTokensModel, SkinTokensRig]


async def comfy_entrypoint() -> SkinTokensExtension:
    return SkinTokensExtension()
