"""SCAIL / SCAIL-2 nodes: the WanSCAILToVideo conditioning node and the SAM3
preprocessing that turns video tracks into the bundle the SCAIL-2 model consumes.

Enhanced version with:
- 3 reference encoding modes (1+4n batch, per-frame, hybrid)
- 1+4n mask expansion
- RoPE downsample patch (full-resolution RoPE -> avg_pool2d downscale for pose branch)
- SCAIL context window support (model forward patch + SCAILContextHandler)
"""

from typing_extensions import override

import torch
import torch.nn.functional as F
import logging

import nodes
import node_helpers
import comfy.model_management
import comfy.utils
import comfy.patcher_extension
from comfy_api.latest import ComfyExtension, io
from comfy.ldm.sam3.tracker import unpack_masks

# Context window infrastructure (from comfy/context_windows.py)
from comfy.context_windows import (
    IndexListContextWindow, IndexListContextHandler,
    ContextSchedule, ContextFuseMethod, ContextFuseMethods,
    ContextSchedules,
    get_matching_context_schedule, get_matching_fuse_method,
    get_shape_for_dim, match_weights_to_dim,
    create_prepare_sampling_wrapper, create_sampler_sample_wrapper,
)

SAM3TrackData = io.Custom("SAM3_TRACK_DATA")


# Model was trained on these exact colors; deviating degrades multi-identity quality.
DEFAULT_PALETTE = [
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (1.0, 1.0, 0.0),  # Yellow
]


# =============================================================================
# RoPE pose-only downsample patch
# Full-res RoPE first, then avg_pool2d downsample for the pose branch only.
# Matches WanAnimatePlus commit 22555324 (Original SCAIL-2 path).
# =============================================================================
_SCAIL_ROPE_PATCH_APPLIED = False


def _apply_rope_downsample_patch():
    """Patch WanModel.rope_encode + SCAILWanModel.rope_encode to implement
    'full-res RoPE first, then avg_pool2d downsample' for the pose branch only.

    Merged with the branch's existing replacement-mode / animation-mode logic.
    Only the pose branch gets the downsampled RoPE — main/reserved frames
    use the standard non-downsampled RoPE.
    """
    global _SCAIL_ROPE_PATCH_APPLIED
    if _SCAIL_ROPE_PATCH_APPLIED:
        return
    try:
        import comfy.ldm.wan.model as _mod

        # ---- Patch WanModel.rope_encode (base class) ----
        # Adds avg_pool2d logic when _scail_rope_downsample flag is set in transformer_options
        _orig_wan_rope = _mod.WanModel.rope_encode

        def _patched_wan_rope(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None,
                              device=None, dtype=None, transformer_options={}):
            need_downsample = transformer_options.get("_scail_rope_downsample", False)
            if not need_downsample:
                return _orig_wan_rope(self, t, h, w, t_start, steps_t, steps_h, steps_w,
                                       device, dtype, transformer_options)

            # --- Full-resolution RoPE generation (2x spatial) ---
            h2 = h * 2
            w2 = w * 2
            steps_h2 = steps_h * 2 if steps_h is not None else None
            steps_w2 = steps_w * 2 if steps_w is not None else None

            freqs_2x = _orig_wan_rope(self, t, h2, w2, t_start, steps_t, steps_h2, steps_w2,
                                       device, dtype, transformer_options)
            # --- Infer spatial decomposition ---
            patch_size = self.patch_size
            t_p = steps_t if steps_t is not None else ((t + patch_size[0] // 2) // patch_size[0])
            h_p_2x = steps_h2 if steps_h2 is not None else ((h2 + patch_size[1] // 2) // patch_size[1])
            w_p_2x = steps_w2 if steps_w2 is not None else ((w2 + patch_size[2] // 2) // patch_size[2])

            # --- avg_pool2d downsample on spatial dimensions ---
            d_rope = freqs_2x.shape[3]
            freqs_grid = freqs_2x.reshape(1, t_p, h_p_2x, w_p_2x, d_rope, 2, 2)

            cos = freqs_grid[..., 0, 0].contiguous()
            sin = freqs_grid[..., 1, 0].contiguous()

            B, Tp, H2, W2, D = cos.shape
            cos_4d = cos.permute(0, 1, 4, 2, 3).reshape(1, Tp * D, H2, W2)
            sin_4d = sin.permute(0, 1, 4, 2, 3).reshape(1, Tp * D, H2, W2)

            cos_pooled = F.avg_pool2d(cos_4d, 2, 2)
            sin_pooled = F.avg_pool2d(sin_4d, 2, 2)

            H, W = cos_pooled.shape[2], cos_pooled.shape[3]

            cos_5d = cos_pooled.reshape(1, Tp, D, H, W).permute(0, 1, 3, 4, 2)
            sin_5d = sin_pooled.reshape(1, Tp, D, H, W).permute(0, 1, 3, 4, 2)

            freqs_out_grid = torch.stack([
                torch.stack([cos_5d, -sin_5d], dim=-1),
                torch.stack([sin_5d,  cos_5d], dim=-1),
            ], dim=-2)

            return freqs_out_grid.reshape(1, Tp * H * W, 1, d_rope, 2, 2)

        _mod.WanModel.rope_encode = _patched_wan_rope

        # ---- Patch SCAILWanModel.rope_encode to inject flag ONLY into pose branch ----
        # Preserves the branch's existing replacement-mode and animation-mode logic,
        # but adds avg_pool2d downsample for the pose RoPE.
        _orig_scail_rope = _mod.SCAILWanModel.rope_encode

        def _patched_scail_rope(self, t, h, w, t_start=0, steps_t=None, steps_h=None, steps_w=None,
                                 device=None, dtype=None, pose_latents=None, reference_latent=None,
                                 ref_mask_flag=None, transformer_options={}):
            if pose_latents is None:
                # No pose -> no downsample needed, pass through to original
                return _orig_scail_rope(self, t, h, w, t_start, steps_t, steps_h, steps_w,
                                         device, dtype, pose_latents=None,
                                         reference_latent=reference_latent,
                                         ref_mask_flag=ref_mask_flag,
                                         transformer_options=transformer_options)

            F_pose, H_pose, W_pose = pose_latents.shape[-3], pose_latents.shape[-2], pose_latents.shape[-1]
            ref_t_patches = 0
            if reference_latent is not None:
                ref_t_patches = (reference_latent.shape[2] + (self.patch_size[0] // 2)) // self.patch_size[0]

            main_t_patches = t - ref_t_patches
            video_t_start = max(ref_t_patches - 1, 0)

            # --- Replacement mode path (ref_mask_flag=False) ---
            if ref_mask_flag is not None and not bool(ref_mask_flag):
                REF_ROPE_H = 120.0
                POSE_ROPE_W = 120.0

                parts = []
                if ref_t_patches > 0:
                    ref_tf = {"rope_options": {"shift_y": REF_ROPE_H, "shift_x": 0.0, "scale_y": 1.0, "scale_x": 1.0}}
                    parts.append(_mod.WanModel.rope_encode(self, ref_t_patches, h, w, t_start=0, device=device, dtype=dtype, transformer_options=ref_tf))
                if main_t_patches > 0:
                    parts.append(_mod.WanModel.rope_encode(self, main_t_patches, h, w, t_start=video_t_start, device=device, dtype=dtype, transformer_options=transformer_options))
                if F_pose > 0:
                    # Pose branch: generate full-resolution RoPE then avg_pool2d downsample.
                    # No scale/shift on coordinates — dense 0..N-1 grid with x-offset=120.0,
                    # matching WanAnimatePlus commit 22555324 (Original SCAIL-2 path).
                    pose_tf = {"rope_options": {"shift_y": 0.0, "shift_x": 120.0, "scale_y": 1.0, "scale_x": 1.0},
                               "_scail_rope_downsample": True}
                    parts.append(_mod.WanModel.rope_encode(self, F_pose, H_pose, W_pose, t_start=0, device=device, dtype=dtype, transformer_options=pose_tf))
                return torch.cat(parts, dim=1)

            # --- Animation mode path (default) ---
            main_freqs = _mod.WanModel.rope_encode(self, t, h, w, t_start=t_start, steps_t=steps_t, steps_h=steps_h, steps_w=steps_w,
                                                     device=device, dtype=dtype, transformer_options=transformer_options)

            if F_pose > 0:
                # Pose frames: WITH downsample (only if there are pose frames)
                # No scale/shift — dense 0..N-1 grid with x-offset=120.0 (Original SCAIL-2 path)
                pose_tf = {"rope_options": {"shift_y": 0.0, "shift_x": 120.0, "scale_y": 1.0, "scale_x": 1.0},
                           "_scail_rope_downsample": True}
                pose_freqs = _mod.WanModel.rope_encode(self, F_pose, H_pose, W_pose, t_start=t_start + ref_t_patches,
                                                         device=device, dtype=dtype, transformer_options=pose_tf)
                return torch.cat([main_freqs, pose_freqs], dim=1)

            return main_freqs

        _mod.SCAILWanModel.rope_encode = _patched_scail_rope

        _SCAIL_ROPE_PATCH_APPLIED = True
        logging.info("[WanSCAIL_MultiRef] RoPE pose-only downsample patch applied")
    except Exception as e:
        logging.warning("[WanSCAIL_MultiRef] Failed to patch RoPE downsample: %s", e)


# =============================================================================
# SCAIL model forward patch: slices SCAIL conditioning at model entry
# =============================================================================
_SCAIL_PATCH_APPLIED = False


def _apply_scail_model_patch():
    """Patch SCAILWanModel._forward to slice ref_mask_latents and driving_mask_28ch
    at model entry, matching windowed x. Also handles ref_mask_flag passthrough."""
    global _SCAIL_PATCH_APPLIED
    if _SCAIL_PATCH_APPLIED:
        return
    try:
        import comfy.ldm.wan.model as _mod
        _orig_forward = _mod.SCAILWanModel._forward

        def _patched_forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None,
                             transformer_options={}, pose_latents=None, ref_mask_latents=None,
                             sam_latents=None, **kwargs):
            window = transformer_options.get("context_window", None) if transformer_options else None

            # ComfyUI renames conditioning dict key "ref_mask_28ch" -> model named param
            # "ref_mask_latents" via Python function argument binding.
            # We slice it here to match the windowed x.
            if ref_mask_latents is not None and window is not None and hasattr(window, "index_list"):
                # ref_mask_latents: (B, 28, N+T_full, H, W) - ComfyUI movedim'ed to channel-first
                # n_ref = total frames in ref_mask_latents minus video frames = N
                n_ref = max(0, ref_mask_latents.shape[2] - window.total_frames)
                indices = window.index_list
                ref_part = ref_mask_latents[:, :, :n_ref]
                video_part = ref_mask_latents[:, :, n_ref:]

                # --- Overlap tracking log ---
                if hasattr(window, "original_indices") and len(video_part.shape) >= 3:
                    oi = window.original_indices
                    if oi and len(oi) > 1:
                        max_idx = video_part.shape[2] - 1
                        idx0 = min(oi[0], max_idx)
                        idx1 = min(oi[-1], max_idx)
                        first_frame_val = video_part[:, :, idx0].mean().item()
                        last_frame_val = video_part[:, :, idx1].mean().item()
                        logging.info(
                            "[SCAIL_OVERLAP] ref_mask overlap boundary: window orig=[%d..%d], "
                            "first_frame_mean=%.6f, last_frame_mean=%.6f",
                            oi[0], oi[-1], first_frame_val, last_frame_val
                        )
                # --- End overlap tracking ---

                safe_indices = [i for i in indices if i < video_part.shape[2]]
                video_part = video_part[:, :, safe_indices]
                ref_mask_latents = torch.cat([ref_part, video_part], dim=2)

            # driving_mask_28ch is NOT a named param of _forward, stays in kwargs
            _driving_from_kw = kwargs.pop("driving_mask_28ch", None)
            if _driving_from_kw is not None and window is not None and hasattr(window, "index_list"):
                _driving_from_kw = _driving_from_kw[:, window.index_list]

            ref_mask_flag = kwargs.pop("ref_mask_flag", None)

            return _orig_forward(self, x, timestep, context, clip_fea=clip_fea,
                                  time_dim_concat=time_dim_concat,
                                  transformer_options=transformer_options,
                                  pose_latents=pose_latents,
                                  ref_mask_latents=ref_mask_latents,
                                  sam_latents=sam_latents,
                                  driving_mask_28ch=_driving_from_kw,
                                  ref_mask_flag=ref_mask_flag,
                                  **kwargs)

        _mod.SCAILWanModel._forward = _patched_forward
        _SCAIL_PATCH_APPLIED = True
        logging.info("[SCAILContext] Model patch applied")
    except Exception as e:
        logging.warning("[SCAILContext] Failed to patch SCAIL model: %s", e)


# =============================================================================
# Mask helper functions (existing, reused)
# =============================================================================
def _unpack(track_data):
    packed = track_data["packed_masks"]
    if packed is None or packed.shape[1] == 0:
        return None
    return unpack_masks(packed)


def _first_appearance_cx_area(masks_bool):
    """Per object: first frame it appears in, plus centroid-x and area in that frame."""
    m = masks_bool.float()
    T, H, W = m.shape[0], m.shape[-2], m.shape[-1]
    grid_x = torch.arange(W, device=m.device, dtype=m.dtype).view(1, 1, 1, W)
    area_t = m.sum(dim=(-1, -2))
    cx_t = (m * grid_x).sum(dim=(-1, -2)) / area_t.clamp(min=1)
    present = area_t > 0
    frame_idx = torch.arange(T, device=m.device).unsqueeze(1)
    first_t = torch.where(present, frame_idx, T).amin(dim=0)
    sel = first_t.clamp(max=T - 1).unsqueeze(0)
    cx = cx_t.gather(0, sel).squeeze(0)
    area = area_t.gather(0, sel).squeeze(0)
    return first_t.tolist(), (cx / W).tolist(), (area / (H * W)).tolist()


def _subset_track_data(track_data, obj_indices):
    out = dict(track_data)
    packed = track_data["packed_masks"]
    if packed is None or not obj_indices:
        out["packed_masks"] = None
        if "scores" in out:
            out["scores"] = []
        return out
    out["packed_masks"] = packed[:, obj_indices].contiguous()
    scores = track_data.get("scores")
    if scores is not None:
        out["scores"] = [scores[i] for i in obj_indices if i < len(scores)]
    return out


def _render_colored_masks(track_data, background="black"):
    packed = track_data["packed_masks"]
    H, W = track_data["orig_size"]
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    bg_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    if packed is None or packed.shape[1] == 0:
        T = track_data.get("n_frames", 1) if packed is None else packed.shape[0]
        out = torch.empty(T, H, W, 3, device=device, dtype=dtype)
        out[..., 0], out[..., 1], out[..., 2] = bg_rgb[0], bg_rgb[1], bg_rgb[2]
        return out
    T, N_obj = packed.shape[0], packed.shape[1]
    colors = torch.tensor(
        [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(N_obj)],
        device=device, dtype=dtype,
    )
    masks_full = unpack_masks(packed.to(device)).float()
    Hm, Wm = masks_full.shape[-2], masks_full.shape[-1]
    masks_full = F.interpolate(
        masks_full.view(T * N_obj, 1, Hm, Wm), size=(H, W), mode="nearest"
    ).view(T, N_obj, H, W) > 0.5
    any_mask = masks_full.any(dim=1)
    color_overlay = colors[masks_full.to(torch.uint8).argmax(dim=1)]
    bg_tensor = torch.tensor(bg_rgb, device=device, dtype=color_overlay.dtype).view(1, 1, 1, 3)
    return torch.where(any_mask.unsqueeze(-1), color_overlay, bg_tensor.expand_as(color_overlay))


def _render_mask_as_identity(mask, background="black"):
    """Plain comfy MASK (B,H,W) or (H,W) -> (B,H,W,3) rendered as a single identity (palette[0])
    on the given background. A batch is treated as multiple views of that one subject."""
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=device, dtype=dtype)
    B, H, W = mask.shape
    bg_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    color = torch.tensor(DEFAULT_PALETTE[0], device=device, dtype=dtype).view(1, 1, 1, 3)
    bg = torch.tensor(bg_rgb, device=device, dtype=dtype).view(1, 1, 1, 3)
    return torch.where((mask > 0.5).unsqueeze(-1), color.expand(B, H, W, 3), bg.expand(B, H, W, 3))


def _extract_mask_to_28ch(rgb_video):
    """Colored RGB mask (T, H, W, 3) in [0, 1] -> SCAIL-2 28-channel binary latent
    (1, T_lat, 28, H_lat, W_lat). 7 per-color binary channels (white/r/g/b/y/m/c)
    threshold-extracted at 225/255, 8x spatial downsample, 4-frame temporal stacking."""
    T, H, W, _ = rgb_video.shape
    _ON_THRESH = 225.0 / 255.0
    mask = rgb_video.movedim(-1, 1).float()
    R = (mask[:, 0:1] > _ON_THRESH).float()
    G = (mask[:, 1:2] > _ON_THRESH).float()
    B = (mask[:, 2:3] > _ON_THRESH).float()
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    binary_7ch = torch.cat([
        R * G * B,    # white
        R * nG * nB,  # red
        nR * G * nB,  # green
        nR * nG * B,  # blue
        R * G * nB,   # yellow
        R * nG * B,   # magenta
        nR * G * B,   # cyan
    ], dim=1)
    H_lat, W_lat = H, W
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2
    binary_7ch = torch.nn.functional.interpolate(binary_7ch, size=(H_lat, W_lat), mode='area')
    T_latent = (T - 1) // 4 + 1
    padded = torch.cat([binary_7ch[:1].repeat(4, 1, 1, 1), binary_7ch[1:]], dim=0)
    out = padded.view(T_latent, 28, H_lat, W_lat)
    return out.unsqueeze(0)


# =============================================================================
# SCAILContextHandler — SCAIL-aware conditioning slicing for context windows
# =============================================================================
class SCAILContextHandler(IndexListContextHandler):
    """Context handler with SCAIL-aware conditioning slicing.

    SCAIL conditioning field time-dimension layout:
    - ref_mask_28ch:   (B, N+T, 28, H, W), time in dim=1  (N ref + T video)
    - driving_mask_28ch: (B, T, 28, H, W), time in dim=1
    - pose_video_latent: wrapped .cond (B, C, T, H, W), time in dim=2
    - reference_latents: list[Tensor], keep untouched (global reference)
    - ref_mask_flag: bool, keep untouched
    - clip_vision_output: object, keep untouched
    """

    def get_resized_cond(self, cond_in: list[dict], x_in: torch.Tensor,
                         window: IndexListContextWindow, device=None) -> list:
        if cond_in is None:
            return None
        resized_cond = []
        if self.split_conds_to_windows and len(cond_in) > 1:
            region = window.get_region_index(len(cond_in))
            cond_in = [cond_in[region]]
        for actual_cond in cond_in:
            resized_actual_cond = actual_cond.copy()
            for key in actual_cond:
                try:
                    cond_item = actual_cond[key]
                    # --- Level 1: plain tensors ---
                    if isinstance(cond_item, torch.Tensor):
                        # SCAIL fields: pass through — model patch handles slicing
                        if key in ("ref_mask_28ch", "driving_mask_28ch"):
                            resized_actual_cond[key] = cond_item.to(device)
                        # Standard: time in dim=self.dim (=2)
                        elif (self.dim < cond_item.ndim
                              and cond_item.size(self.dim) == x_in.size(self.dim)):
                            resized_actual_cond[key] = window.get_tensor(
                                cond_item, device, retain_index_list=self.cond_retain_index_list
                            )
                        # Fallback dim=1
                        elif cond_item.ndim >= 2 and cond_item.size(1) == x_in.size(self.dim):
                            resized_actual_cond[key] = window.get_tensor(
                                cond_item, device, dim=1,
                                retain_index_list=self.cond_retain_index_list
                            )
                        else:
                            resized_actual_cond[key] = cond_item.to(device)

                    # --- Level 2: control objects ---
                    elif key == "control":
                        resized_actual_cond[key] = self.prepare_control_objects(cond_item, device)

                    # --- Level 3: dict items (where SCAIL .cond fields live) ---
                    elif isinstance(cond_item, dict):
                        new_cond_item = cond_item.copy()
                        for cond_key, cond_value in new_cond_item.items():
                            # ----- pose_video_latent (time in dim=2, .cond wrapped) -----
                            if (cond_key == "pose_video_latent"
                                    and hasattr(cond_value, "cond")
                                    and isinstance(cond_value.cond, torch.Tensor)):
                                indices = range(self.context_length)
                                if cond_value.cond.ndim == 5:
                                    idx = (slice(None), slice(None), list(indices),
                                           slice(None), slice(None))
                                elif cond_value.cond.ndim == 4:
                                    idx = (slice(None), list(indices), slice(None), slice(None))
                                else:
                                    idx = (slice(None), list(indices))
                                sliced = cond_value.cond[idx]
                                new_cond_item[cond_key] = cond_value._copy_with(sliced)
                                continue

                            # ----- SCAIL dict-level fields: pass through, model patch handles -----
                            if cond_key in ("driving_mask_28ch", "ref_mask_28ch"):
                                if isinstance(cond_value, torch.Tensor):
                                    new_cond_item[cond_key] = cond_value.to(device)
                                    continue
                                elif hasattr(cond_value, "cond") and isinstance(cond_value.cond, torch.Tensor):
                                    new_cond_item[cond_key] = cond_value
                                    continue

                            # ----- clip_vision_output: passthrough -----

                            # ----- Callback hooks -----
                            handled = False
                            for callback in comfy.patcher_extension.get_all_callbacks(
                                "resize_cond_item", self.callbacks
                            ):
                                result = callback(cond_key, cond_value, window, x_in, device,
                                                  new_cond_item)
                                if result is not None:
                                    new_cond_item[cond_key] = result
                                    handled = True
                                    break
                            if handled:
                                continue

                            # ----- face_pixel_values (time in pixel frames) -----
                            if (cond_key == "face_pixel_values"
                                    and hasattr(cond_value, "cond")
                                    and isinstance(cond_value.cond, torch.Tensor)):
                                pixel_tensor = cond_value.cond
                                oi = window.original_indices if hasattr(window, "original_indices") and window.original_indices else window.index_list
                                pixel_start = max(oi[0] * 4, 0)
                                pixel_end = min((oi[-1] + 1) * 4, pixel_tensor.shape[2])
                                sliced_pixels = pixel_tensor[:, :, pixel_start:pixel_end, :, :] if pixel_start < pixel_tensor.shape[2] else torch.zeros_like(pixel_tensor[:, :, :0, :, :])
                                new_cond_item[cond_key] = cond_value._copy_with(sliced_pixels)
                                continue

                            # ----- Generic dict-cond handler -----
                            if isinstance(cond_value, torch.Tensor):
                                if (self.dim < cond_value.ndim
                                        and cond_value.size(self.dim) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = window.get_tensor(
                                        cond_value, device,
                                        retain_index_list=self.cond_retain_index_list
                                    )
                            elif hasattr(cond_value, "cond") and isinstance(
                                    cond_value.cond, torch.Tensor):
                                cond_t = cond_value.cond
                                if (self.dim < cond_t.ndim
                                        and cond_t.size(self.dim) == x_in.size(self.dim)):
                                    new_cond_item[cond_key] = cond_value._copy_with(
                                        window.get_tensor(cond_t, device,
                                                          retain_index_list=self.cond_retain_index_list)
                                    )
                            elif cond_key == "num_video_frames":
                                new_cond_item[cond_key] = cond_value._copy_with(cond_value.cond)
                                new_cond_item[cond_key].cond = window.context_length
                        resized_actual_cond[key] = new_cond_item

                    # --- Level 4: anything else (pass through) ---
                    else:
                        resized_actual_cond[key] = cond_item
                finally:
                    del cond_item
            resized_cond.append(resized_actual_cond)
        return resized_cond


# =============================================================================
# WanSCAILToVideo — Enhanced multi-reference conditioning node
# =============================================================================
class WanSCAILToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSCAILToVideo",
            category="model/conditioning/wan/scail",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=512, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=896, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("pose_video", optional=True, tooltip="Video used for pose conditioning. Will be downscaled to half the resolution of the main video."),
                io.Image.Input("pose_video_mask", optional=True, tooltip="SCAIL-2 only. Colored per-identity SAM3 mask video at the same resolution as pose_video."),
                io.Boolean.Input("replacement_mode", default=False, optional=True, tooltip="SCAIL-2 only. False = Animation Mode (pose_video_mask should have black background). True = Replacement Mode (pose_video_mask should have white background)."),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Strength of the pose latent."),
                io.Float.Input("pose_start", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Start step of the pose conditioning."),
                io.Float.Input("pose_end", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="End step of the pose conditioning."),
                io.Combo.Input("ref_encoding_mode", options=["1+4n batch", "per-frame", "hybrid"], default="1+4n batch",
                               tooltip="Reference encoding mode. '1+4n batch' = 1st ref x1, rest x4 then batch encode (default, closer to training); "
                                       "'per-frame' = each ref independently encoded then concatenated along time (high fidelity); "
                                       "'hybrid' = first N-1 as '1+4n batch' + last frame per-frame encoded (balances quality and reduces inter-chunk discontinuity)."),
                io.Image.Input("reference_image", optional=True, tooltip="Reference image. The first image is the primary reference (composite all identities onto it). SCAIL-2: extra batch images are used as additional views, each needing a matching reference_image_mask."),
                io.Image.Input("reference_image_mask", optional=True, tooltip="SCAIL-2 only. Colored reference mask, batch matching reference_image (first = primary reference mask, rest = identity masks for the additional reference_image)."),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True, tooltip="CLIP vision features for conditioning. Model is trained with stretch resize to aspect ratio."),
                io.Int.Input("video_frame_offset", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Cumulative output frame this chunk begins at. Wire from the previous chunk's video_frame_offset output."),
                io.Int.Input("previous_frame_count", default=5, min=1, max=nodes.MAX_RESOLUTION, step=4, tooltip="Tail frames of previous_frames to anchor. SCAIL-2 trained at 5 (81-frame chunks, 76-frame step)."),
                io.Image.Input("previous_frames", optional=True, tooltip="SCAIL-2 only. Full decoded output of the previous chunk. Only the last previous_frame_count are used as the extension anchor."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Empty latent of the generation size."),
                io.Int.Output(display_name="video_frame_offset", tooltip="Adjusted offset + length. Wire into the next chunk."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, pose_strength, pose_start, pose_end,
                video_frame_offset, previous_frame_count, replacement_mode=False, ref_encoding_mode="1+4n batch",
                reference_image=None, clip_vision_output=None, pose_video=None,
                pose_video_mask=None, reference_image_mask=None, previous_frames=None) -> io.NodeOutput:
        # RoPE downsample patch: full-res RoPE first, then avg_pool2d downsample.
        # Matches WanAnimatePlus commit 22555324 (Original SCAIL-2 path).
        _apply_rope_downsample_patch()

        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        noise_mask = None

        ref_mask_flag = not replacement_mode
        positive = node_helpers.conditioning_set_values(positive, {"ref_mask_flag": ref_mask_flag})
        negative = node_helpers.conditioning_set_values(negative, {"ref_mask_flag": ref_mask_flag})

        prev_trimmed = None
        if previous_frames is not None and previous_frames.shape[0] > 0:
            prev_trimmed = previous_frames[-previous_frame_count:]
            video_frame_offset -= prev_trimmed.shape[0]
            video_frame_offset = max(0, video_frame_offset)

        # ----- Multi-reference image handling -----
        # Three encoding modes:
        #   "1+4n batch" : 1st ref x1, each additional x4 -> batch -> single VAE encode
        #   "per-frame"  : each ref independently upsample + VAE encode -> cat in time dim
        #   "hybrid"     : first N-1 as 1+4n batch + last frame independently
        concat_ref_latent = None
        if reference_image is not None:
            num_refs = reference_image.shape[0]
            if ref_encoding_mode == "1+4n batch":
                # 1+4n batch encoding
                ref_pixel_parts = [reference_image[0:1]]
                if num_refs > 1:
                    for i in range(1, num_refs):
                        ref_pixel_parts.append(reference_image[i:i+1].repeat(4, 1, 1, 1))
                ref_pixels = torch.cat(ref_pixel_parts, dim=0)  # (4N-3, H, W, 3)

                # Replacement Mode mask compositing
                if replacement_mode and reference_image_mask is not None:
                    mask_parts = [reference_image_mask[0:1]]
                    if reference_image_mask.shape[0] > 1:
                        for i in range(1, min(num_refs, reference_image_mask.shape[0])):
                            mask_parts.append(reference_image_mask[i:i+1].repeat(4, 1, 1, 1))
                    while len(mask_parts) < len(ref_pixel_parts):
                        mask_parts.append(mask_parts[-1])
                    ref_masks = torch.cat(mask_parts, dim=0)
                    rm = comfy.utils.common_upscale(
                        ref_masks.movedim(-1, 1), width, height, "nearest-exact", "center"
                    ).movedim(1, -1)
                    is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ref_pixels.dtype)
                    ref_pixels = ref_pixels * is_char

                # Batch upsample + VAE encode (bicubic for upsampling)
                ref_pixels = comfy.utils.common_upscale(
                    ref_pixels.movedim(-1, 1), width, height, "bicubic", "center"
                ).movedim(1, -1)
                concat_ref_latent = vae.encode(ref_pixels[:, :, :, :3])
                # concat_ref_latent: (1, 16, 4N-3, H/8, W/8)

            elif ref_encoding_mode == "per-frame":
                ref_latent_parts = []
                for i in range(num_refs):
                    single_ref = comfy.utils.common_upscale(
                        reference_image[i:i+1].movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    if replacement_mode and reference_image_mask is not None:
                        mask_idx = min(i, reference_image_mask.shape[0] - 1)
                        rm = comfy.utils.common_upscale(
                            reference_image_mask[mask_idx:mask_idx+1].movedim(-1, 1), width, height, "nearest-exact", "center"
                        ).movedim(1, -1)
                        is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(single_ref.dtype)
                        single_ref = single_ref * is_char
                    ref_latent = vae.encode(single_ref[:, :, :, :3])
                    ref_latent_parts.append(ref_latent)
                concat_ref_latent = torch.cat(ref_latent_parts, dim=2)
                # concat_ref_latent: (1, 16, N, H/8, W/8)

            else:  # "hybrid" — first N-1 as 1+4n batch + last frame independently
                if num_refs == 1:
                    # Single ref degenerates to per-frame encoding
                    single_ref = comfy.utils.common_upscale(
                        reference_image[0:1].movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    concat_ref_latent = vae.encode(single_ref[:, :, :, :3])
                else:
                    # First N-1: 1+4n batch
                    batch_parts = [reference_image[0:1]]
                    for i in range(1, num_refs - 1):
                        batch_parts.append(reference_image[i:i+1].repeat(4, 1, 1, 1))
                    ref_pixels_batch = torch.cat(batch_parts, dim=0)
                    if replacement_mode and reference_image_mask is not None:
                        mask_parts = [reference_image_mask[0:1]]
                        if reference_image_mask.shape[0] > 1:
                            for i in range(1, min(num_refs - 1, reference_image_mask.shape[0])):
                                mask_parts.append(reference_image_mask[i:i+1].repeat(4, 1, 1, 1))
                        while len(mask_parts) < len(batch_parts):
                            mask_parts.append(mask_parts[-1])
                        ref_masks = torch.cat(mask_parts, dim=0)
                        rm = comfy.utils.common_upscale(
                            ref_masks.movedim(-1, 1), width, height, "nearest-exact", "center"
                        ).movedim(1, -1)
                        is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ref_pixels_batch.dtype)
                        ref_pixels_batch = ref_pixels_batch * is_char
                    ref_pixels_batch = comfy.utils.common_upscale(
                        ref_pixels_batch.movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    batch_latent = vae.encode(ref_pixels_batch[:, :, :, :3])
                    # Last frame: per-frame encoding
                    last_ref = comfy.utils.common_upscale(
                        reference_image[-1:].movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    if replacement_mode and reference_image_mask is not None:
                        mask_idx = min(num_refs - 1, reference_image_mask.shape[0] - 1)
                        rm = comfy.utils.common_upscale(
                            reference_image_mask[mask_idx:mask_idx+1].movedim(-1, 1), width, height, "nearest-exact", "center"
                        ).movedim(1, -1)
                        is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(last_ref.dtype)
                        last_ref = last_ref * is_char
                    last_latent = vae.encode(last_ref[:, :, :, :3])
                    concat_ref_latent = torch.cat([batch_latent, last_latent], dim=2)

        if concat_ref_latent is not None:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [concat_ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [concat_ref_latent]}, append=True)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]
        if pose_video_mask is not None:
            if pose_video_mask.shape[0] <= video_frame_offset:
                pose_video_mask = None
            else:
                pose_video_mask = pose_video_mask[video_frame_offset:]

        # Truncate pose+mask jointly to the shorter of the two, capped at length.
        ts = [v.shape[0] for v in (pose_video, pose_video_mask) if v is not None]
        if ts:
            T_kept = ((min(min(ts), length) - 1) // 4) * 4 + 1
            if pose_video is not None:
                pose_video = pose_video[:T_kept]
            if pose_video_mask is not None:
                pose_video_mask = pose_video_mask[:T_kept]

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(pose_video[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            pose_video_latent = vae.encode(pose_video[:, :, :, :3]) * pose_strength
            positive = node_helpers.conditioning_set_values_with_timestep_range(positive, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)
            negative = node_helpers.conditioning_set_values_with_timestep_range(negative, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)

        if pose_video_mask is not None:
            mask_video_hw = comfy.utils.common_upscale(pose_video_mask[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            driving_mask_28ch = _extract_mask_to_28ch(mask_video_hw)
            positive = node_helpers.conditioning_set_values(positive, {"driving_mask_28ch": driving_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"driving_mask_28ch": driving_mask_28ch})

        # ----- Multi-reference mask handling (ref_mask_28ch) -----
        if reference_image_mask is not None:
            if ref_encoding_mode == "1+4n batch":
                # 1+4n pixel-level expansion: 1st mask x1, rest x4
                mask_parts = [reference_image_mask[0:1]]
                if reference_image_mask.shape[0] > 1:
                    for i in range(1, reference_image_mask.shape[0]):
                        mask_parts.append(reference_image_mask[i:i+1].repeat(4, 1, 1, 1))
                masks_expanded = torch.cat(mask_parts, dim=0)  # (4N-3, H, W, 3)
                ref_mask_hw = comfy.utils.common_upscale(
                    masks_expanded.movedim(-1, 1), width, height, "bicubic", "center"
                ).movedim(1, -1)
                ref_mask_concat = _extract_mask_to_28ch(ref_mask_hw)  # (1, N, 28, H_lat, W_lat)
            else:  # "per-frame" or "hybrid" — per-frame mask encode
                ref_mask_t_parts = []
                for i in range(reference_image_mask.shape[0]):
                    ref_mask_hw = comfy.utils.common_upscale(
                        reference_image_mask[i:i+1].movedim(-1, 1), width, height, "bicubic", "center"
                    ).movedim(1, -1)
                    ref_mask_1f = _extract_mask_to_28ch(ref_mask_hw)  # (1, 1, 28, H_lat, W_lat)
                    ref_mask_t_parts.append(ref_mask_1f)
                ref_mask_concat = torch.cat(ref_mask_t_parts, dim=1)

            # Pad with small value to avoid zero boundary in ref_mask_28ch
            T_lat = latent.shape[2]
            zeros = torch.full(
                (1, T_lat, 28, ref_mask_concat.shape[-2], ref_mask_concat.shape[-1]),
                fill_value=0.001,
                device=ref_mask_concat.device, dtype=ref_mask_concat.dtype
            )
            ref_mask_full = torch.cat([ref_mask_concat, zeros], dim=1)  # (1, N_ref_mask+T_lat, 28, H_lat, W_lat)

            positive = node_helpers.conditioning_set_values(positive, {"ref_mask_28ch": ref_mask_full})
            negative = node_helpers.conditioning_set_values(negative, {"ref_mask_28ch": ref_mask_full})

        if prev_trimmed is not None:
            pf = comfy.utils.common_upscale(prev_trimmed.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            prev_latent = vae.encode(pf[:, :, :, :3])
            prev_latent_frames = min(prev_latent.shape[2], latent.shape[2])
            latent[:, :, :prev_latent_frames] = prev_latent[:, :, :prev_latent_frames].to(latent.dtype)
            noise_mask = torch.ones((1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]),
                                    device=latent.device, dtype=latent.dtype)
            noise_mask[:, :, :prev_latent_frames] = 0.0

        out_latent = {"samples": latent}
        if noise_mask is not None:
            out_latent["noise_mask"] = noise_mask
        return io.NodeOutput(positive, negative, out_latent, video_frame_offset + length)


# =============================================================================
# SCAIL2ColoredMask — Render SAM3 tracks into colored masks
# =============================================================================
class SCAIL2ColoredMask(io.ComfyNode):
    """Render SAM3 tracks for the driving pose video and reference image(s) into the
    colored masks WanSCAILToVideo consumes. Shared `sort_by` keeps each identity on the
    same color across both outputs.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2ColoredMask",
            display_name="Create SCAIL-2 Colored Mask",
            category="model/conditioning/wan/scail",
            inputs=[
                SAM3TrackData.Input("driving_track_data", tooltip="SAM3 track of the driving pose video. Will be rendered into the pose_video_mask output."),
                io.MultiType.Input("ref_track_data", [SAM3TrackData, io.Mask], optional=True, display_name="reference_masks",
                                   tooltip="SAM3 track of the reference image(s) (one identity per object, colored in batch order), or a plain MASK of the reference subject (rendered as a single identity)."),
                io.String.Input("object_indices", default="",
                                tooltip="Comma-separated list of person indices to include (e.g. '0,2,3'). Applied to both reference and pose video masks. Empty = all."),
                io.Combo.Input("sort_by", options=["none", "left_to_right", "area"], default="left_to_right",
                               tooltip="Order in which palette colors are assigned to the tracked objects (applied to both reference and pose video so each identity keeps the same color). Objects that appear in earlier frames always come first; within a frame, left_to_right = leftmost object (by centroid at first appearance) gets the first color, area = biggest object (by mask area at first appearance) gets the first color; none = keep SAM3's order."),
                io.Boolean.Input("replacement_mode", default=False,
                    tooltip="False = Animation Mode (pose_video_mask has black background, reference_image_mask has white background). "
                    "True = Replacement Mode (pose_video_mask has white background, reference_image_mask has black background)."),
            ],
            outputs=[
                io.Image.Output("pose_video_mask"),
                io.Image.Output("reference_image_mask"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, driving_track_data, object_indices, sort_by, replacement_mode, ref_track_data=None):
        def _prep(td):
            masks_bool = _unpack(td)
            if sort_by != "none" and masks_bool is not None:
                first_t, cx, area = _first_appearance_cx_area(masks_bool)
                if sort_by == "left_to_right":
                    order = sorted(range(len(cx)), key=lambda i: (first_t[i], cx[i]))
                else:  # "area"
                    order = sorted(range(len(area)), key=lambda i: (first_t[i], -area[i]))
                td = _subset_track_data(td, order)
            if object_indices.strip():
                indices = [int(i.strip()) for i in object_indices.split(",") if i.strip().isdigit()]
                packed = td.get("packed_masks")
                n_obj = packed.shape[1] if packed is not None else 0
                indices = [i for i in indices if 0 <= i < n_obj]
                td = _subset_track_data(td, indices)
            return td

        drv = _prep(driving_track_data)
        # Animation: driving=black, ref=white. Replacement: driving=white, ref=black.
        mask_video = _render_colored_masks(drv, "white" if replacement_mode else "black")
        ref_bg = "black" if replacement_mode else "white"

        if ref_track_data is not None:
            if isinstance(ref_track_data, torch.Tensor):  # plain comfy MASK
                reference_image_mask = _render_mask_as_identity(ref_track_data, ref_bg)
            else:
                reference_image_mask = _render_colored_masks(_prep(ref_track_data), ref_bg)
        else:
            H, W = drv["orig_size"]
            fill_value = 1.0 if ref_bg == "white" else 0.0
            reference_image_mask = torch.full((1, H, W, 3), fill_value, device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())

        return io.NodeOutput(mask_video, reference_image_mask)


# =============================================================================
# WanSCAILContextWindows — SCAIL context window node
# =============================================================================
class WanSCAILContextWindows(io.ComfyNode):
    """SCAIL context window node. Automatically handles time-dimension differences
    across conditioning fields.

    SCAIL's reference_latents / ref_mask_28ch[N frames] are preserved as global
    conditions — no prefix mechanism needed.

    SCAIL conditioning fields:
    - reference_latents: keep all N frames (global reference, no slicing)
    - pose_video_latent: slice along dim=2 matching latent window
    - driving_mask_28ch: slice along dim=1 matching latent window
    - ref_mask_28ch: keep ref portion (N frames) + slice video portion by window
    """

    @classmethod
    def define_schema(cls):
        schedule_options = [
            ContextSchedules.UNIFORM_LOOPED,
            ContextSchedules.UNIFORM_STANDARD,
            ContextSchedules.STATIC_STANDARD,
            ContextSchedules.BATCHED,
        ]
        fuse_options = ContextFuseMethods.LIST_STATIC
        return io.Schema(
            node_id="WanSCAILContextWindows",
            display_name="Wan SCAIL Context Windows",
            category="model/patch/wan",
            description="SCAIL context windows. Automatically handles time-dimension differences in conditioning fields.",
            inputs=[
                io.Model.Input("model", tooltip="The model to apply context windows to (SCAIL model)."),
                io.Int.Input("context_length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4,
                             tooltip="Context window length (pixel frames). Recommended to match SCAIL's length parameter (e.g. 81)."),
                io.Int.Input("context_overlap", default=30, min=0, max=nodes.MAX_RESOLUTION,
                             tooltip="Overlap between windows (pixel frames)."),
                io.Combo.Input("context_schedule", options=schedule_options,
                               tooltip="Window generation schedule. static_standard=fixed windows (recommended)."),
                io.Int.Input("context_stride", default=1, min=1, max=10,
                             tooltip="Window stride (only effective for uniform schedules).", advanced=True),
                io.Boolean.Input("closed_loop", default=False,
                                 tooltip="Whether to close the window loop (only effective for looped schedules).", advanced=True),
                io.Combo.Input("fuse_method", options=fuse_options, default=ContextFuseMethods.PYRAMID,
                               tooltip="Window fusion method."),
                io.Boolean.Input("freenoise", default=False,
                                 tooltip="FreeNoise noise shuffling, improves inter-window blending."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with context windows applied during sampling."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, context_length, context_overlap,
                context_schedule, context_stride, closed_loop,
                fuse_method, freenoise) -> io.Model:
        context_length = max(((context_length - 1) // 4) + 1, 1)
        context_overlap = max(((context_overlap - 1) // 4) + 1, 0)

        _apply_scail_model_patch()

        model = model.clone()
        handler = SCAILContextHandler(
            context_schedule=get_matching_context_schedule(context_schedule),
            fuse_method=get_matching_fuse_method(fuse_method),
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=2,
            freenoise=freenoise,
            split_conds_to_windows=False,
            causal_window_fix=False,
            cond_retain_index_list=[],
        )
        model.model_options["context_handler"] = handler

        create_prepare_sampling_wrapper(model)
        if freenoise:
            create_sampler_sample_wrapper(model)

        logging.info(
            "[SCAILContext] Applied context windows: len=%d_latent overlap=%d_latent "
            "schedule=%s fuse=%s",
            context_length, context_overlap,
            context_schedule, fuse_method
        )
        return io.NodeOutput(model)


# =============================================================================
# Extension registration
# =============================================================================
class SCAILExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanSCAILToVideo,
            SCAIL2ColoredMask,
            WanSCAILContextWindows,
        ]


async def comfy_entrypoint() -> SCAILExtension:
    return SCAILExtension()