"""
SAM3 (Segment Anything 3) ComfyUI nodes for detection, segmentation, and video tracking.
"""

from typing_extensions import override

import json
import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io


def _refine_mask(sam3_model, orig_image_hwc, coarse_mask, box_xyxy, H, W, device, dtype, iterations):
    """Refine a coarse detector mask via SAM decoder, cropping to the detection box.

    Returns: [1, H, W] binary mask
    """
    if iterations <= 0:
        mask = F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
        return (mask[0] > 0).float()

    # Crop detection region from original image with padding
    pad_frac = 0.1
    x1, y1, x2, y2 = box_xyxy.tolist()
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * pad_frac))
    cy1 = max(0, int(y1 - bh * pad_frac))
    cx2 = min(W, int(x2 + bw * pad_frac))
    cy2 = min(H, int(y2 + bh * pad_frac))

    crop = orig_image_hwc[cy1:cy2, cx1:cx2]
    crop_1008 = comfy.utils.common_upscale(crop.unsqueeze(0).movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
    crop_frame = crop_1008.to(device=device, dtype=dtype)
    crop_h, crop_w = cy2 - cy1, cx2 - cx1

    # Crop coarse mask and refine via SAM on the cropped image
    mask_h, mask_w = coarse_mask.shape[-2:]
    mx1, my1 = int(cx1 / W * mask_w), int(cy1 / H * mask_h)
    mx2, my2 = int(cx2 / W * mask_w), int(cy2 / H * mask_h)
    mask_logit = coarse_mask[..., my1:my2, mx1:mx2].unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        coarse_input = F.interpolate(mask_logit, size=(1008, 1008), mode="bilinear", align_corners=False)
        mask_logit = sam3_model.forward_segment(crop_frame, mask_inputs=coarse_input)

    refined_crop = F.interpolate(mask_logit, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
    full_mask = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    full_mask[:, :, cy1:cy2, cx1:cx2] = refined_crop
    coarse_full = F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
    return ((full_mask[0] > 0) | (coarse_full[0] > 0)).float()



class SAM3_Detect(io.ComfyNode):
    """Open-vocabulary detection and segmentation using text, box, or point prompts."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_Detect",
            display_name="SAM3 Detect",
            category="detection/",
            search_aliases=["sam3", "segment anything", "open vocabulary", "text detection", "segment"],
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Conditioning.Input("conditioning", display_name="conditioning", optional=True, tooltip="Text conditioning from CLIPTextEncode"),
                io.BoundingBox.Input("bboxes", display_name="bboxes", force_input=True, optional=True, tooltip="Bounding boxes to segment within"),
                io.String.Input("positive_coords", display_name="positive_coords", force_input=True, optional=True, tooltip="Positive point prompts as JSON [{\"x\": int, \"y\": int}, ...] (pixel coords)"),
                io.String.Input("negative_coords", display_name="negative_coords", force_input=True, optional=True, tooltip="Negative point prompts as JSON [{\"x\": int, \"y\": int}, ...] (pixel coords)"),
                io.Float.Input("threshold", display_name="threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Int.Input("refine_iterations", display_name="refine_iterations", default=2, min=0, max=5, tooltip="SAM decoder refinement passes (0=use raw detector masks)"),
            ],
            outputs=[
                io.Mask.Output("masks"),
                io.BoundingBox.Output("bboxes"),
            ],
        )

    @classmethod
    def execute(cls, model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.5, refine_iterations=2) -> io.NodeOutput:
        B, H, W, C = image.shape

        image_in = comfy.utils.common_upscale(image.movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")

        # Convert bboxes to normalized cxcywh format [1, N, 4]
        # BoundingBox type can be: single dict, list of dicts, or list of lists of dicts (per-frame)
        boxes_tensor = None
        if bboxes is not None:
            # Flatten to list of dicts
            if isinstance(bboxes, dict):
                flat_boxes = [bboxes]
            elif isinstance(bboxes, list) and len(bboxes) > 0 and isinstance(bboxes[0], list):
                flat_boxes = [d for frame in bboxes for d in frame]  # per-frame list of lists
            elif isinstance(bboxes, list):
                flat_boxes = bboxes
            else:
                flat_boxes = []
            if flat_boxes:
                coords = []
                for d in flat_boxes:
                    cx = (d["x"] + d["width"] / 2) / W
                    cy = (d["y"] + d["height"] / 2) / H
                    coords.append([cx, cy, d["width"] / W, d["height"] / H])
                boxes_tensor = torch.tensor([coords], dtype=torch.float32)  # [1, N, 4]

        # Parse point prompts from JSON (KJNodes PointsEditor format: [{"x": int, "y": int}, ...])
        pos_pts = json.loads(positive_coords) if positive_coords else []
        neg_pts = json.loads(negative_coords) if negative_coords else []
        has_points = len(pos_pts) > 0 or len(neg_pts) > 0

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model

        # Build point inputs for tracker SAM decoder path
        point_inputs = None
        if has_points:
            all_coords = [[p["x"] / W * 1008, p["y"] / H * 1008] for p in pos_pts] + \
                         [[p["x"] / W * 1008, p["y"] / H * 1008] for p in neg_pts]
            all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)
            point_inputs = {
                "point_coords": torch.tensor([all_coords], dtype=dtype, device=device),
                "point_labels": torch.tensor([all_labels], dtype=torch.int32, device=device),
            }

        # Build per-prompt list: [(cond_tensor, attention_mask, max_detections), ...]
        # SAM3 CLIP packs comma-separated prompts with per-prompt max_detections in metadata
        cond_list = []
        if conditioning is not None and len(conditioning) > 0:
            cond_meta = conditioning[0][1]
            multi = cond_meta.get("sam3_multi_cond")
            if multi is not None:
                for entry in multi:
                    cond_list.append((
                        entry["cond"].to(device=device, dtype=dtype),
                        entry["attention_mask"].to(device) if entry["attention_mask"] is not None else None,
                        entry["max_detections"],
                    ))
            else:
                cond_tensor = conditioning[0][0].to(device=device, dtype=dtype)
                attn_mask = cond_meta.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)
                else:
                    attn_mask = torch.ones(cond_tensor.shape[0], cond_tensor.shape[1], dtype=torch.int64, device=device)
                cond_list.append((cond_tensor, attn_mask, 1))
        has_text = len(cond_list) > 0

        # Run per-image through detector (text/boxes) and/or tracker (points)
        all_bbox_dicts = []
        all_masks = []
        pbar = comfy.utils.ProgressBar(B)

        for b in range(B):
            frame = image_in[b:b+1].to(device=device, dtype=dtype)
            b_boxes_tensor = boxes_tensor.to(device=device, dtype=dtype) if boxes_tensor is not None else None

            frame_bbox_dicts = []
            frame_masks = []

            # Point prompts: tracker SAM decoder path with iterative refinement
            if point_inputs is not None:
                mask_logit = sam3_model.forward_segment(frame, point_inputs=point_inputs)
                for _ in range(max(0, refine_iterations - 1)):
                    mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
                mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
                frame_masks.append((mask[0] > 0).float())

            # Box prompts: SAM decoder path (segment inside each box)
            if b_boxes_tensor is not None and not has_text:
                for box_cxcywh in b_boxes_tensor[0]:
                    cx, cy, bw, bh = box_cxcywh.tolist()
                    # Convert cxcywh normalized → xyxy in 1008 space → [1, 2, 2] corners
                    sam_box = torch.tensor([[[(cx - bw/2) * 1008, (cy - bh/2) * 1008],
                                             [(cx + bw/2) * 1008, (cy + bh/2) * 1008]]],
                                           device=device, dtype=dtype)
                    mask_logit = sam3_model.forward_segment(frame, box_inputs=sam_box)
                    for _ in range(max(0, refine_iterations - 1)):
                        mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
                    mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
                    frame_masks.append((mask[0] > 0).float())

            # Text prompts: run detector per text prompt (each detects one category)
            prompts = cond_list if has_text else []
            for prompt in prompts:
                if prompt is not None:
                    text_embeddings, text_mask, max_det = prompt
                else:
                    text_embeddings, text_mask, max_det = None, None, 1

                results = sam3_model(
                    frame,
                    text_embeddings=text_embeddings,
                    text_mask=text_mask,
                    boxes=b_boxes_tensor,
                    threshold=threshold,
                    orig_size=(H, W),
                )

                pred_boxes = results["boxes"][0]
                scores = results["scores"][0]
                masks = results["masks"][0]

                if has_text:
                    probs = scores.sigmoid()
                    keep = probs > threshold
                    kept_boxes = pred_boxes[keep].cpu()
                    kept_scores = probs[keep].cpu()
                    kept_masks = masks[keep]

                    order = kept_scores.argsort(descending=True)[:max_det]
                    kept_boxes = kept_boxes[order]
                    kept_scores = kept_scores[order]
                    kept_masks = kept_masks[order]

                    for box, score in zip(kept_boxes, kept_scores):
                        frame_bbox_dicts.append({
                            "x": float(box[0]), "y": float(box[1]),
                            "width": float(box[2] - box[0]), "height": float(box[3] - box[1]),
                            "score": float(score),
                        })
                    if kept_masks.shape[0] > 0:
                        for m, box in zip(kept_masks, kept_boxes):
                            frame_masks.append(_refine_mask(
                                sam3_model, image[b], m, box, H, W, device, dtype, refine_iterations))
                else:
                    frame_masks.append(_refine_mask(
                        sam3_model, image[b], masks[0], pred_boxes[0].cpu(), H, W, device, dtype, refine_iterations))

            all_bbox_dicts.append(frame_bbox_dicts)
            if len(frame_masks) > 0:
                combined = torch.cat(frame_masks, dim=0)
                all_masks.append((combined > 0).any(dim=0).float())
            else:
                all_masks.append(torch.zeros(H, W, device=comfy.model_management.intermediate_device()))
            pbar.update(1)

        mask_out = torch.stack(all_masks)
        return io.NodeOutput(mask_out, all_bbox_dicts)


class SAM3_VideoTrack(io.ComfyNode):
    """Track objects across video frames using SAM3's memory-based tracker."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_VideoTrack",
            display_name="SAM3 Video Track",
            category="detection/",
            search_aliases=["sam3", "video", "track", "propagate"],
            inputs=[
                io.Image.Input("images", display_name="images", tooltip="Video frames as batched images"),
                io.Model.Input("model", display_name="model"),
                io.Mask.Input("initial_mask", display_name="initial_mask", tooltip="Mask(s) for the first frame to track (one per object)"),
            ],
            outputs=[
                io.Mask.Output("masks", display_name="masks"),
            ],
        )

    @classmethod
    def execute(cls, images, model, initial_mask) -> io.NodeOutput:
        N, H, W, C = images.shape

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model

        frames = images.movedim(-1, 1)
        frames_in = comfy.utils.common_upscale(frames, 1008, 1008, "bilinear", crop="disabled").to(device=device, dtype=dtype)
        # initial_mask: [N_obj, H, W] — one mask per object to track
        init_masks = initial_mask.unsqueeze(1).to(device=device, dtype=dtype)  # [N_obj, 1, H, W]

        pbar = comfy.utils.ProgressBar(N)
        mask_logits = sam3_model.forward_video(images=frames_in, initial_masks=init_masks, pbar=pbar)
        # mask_logits: [N, N_obj, image_size, image_size]

        mask_out = F.interpolate(mask_logits, size=(H, W), mode="bilinear", align_corners=False)
        # Apply non-overlapping constraints for multi-object (matching reference _postprocess_output)
        N_obj = mask_out.shape[1]
        if N_obj > 1:
            for t in range(mask_out.shape[0]):
                obj_masks = mask_out[t]  # [N_obj, H, W]
                # At each pixel, only the highest-scoring object keeps positive logits
                max_obj = torch.argmax(obj_masks, dim=0, keepdim=True)  # [1, H, W]
                batch_inds = torch.arange(N_obj, device=obj_masks.device)[:, None, None]
                keep = (max_obj == batch_inds)
                mask_out[t] = torch.where(keep, obj_masks, torch.clamp(obj_masks, max=-10.0))
        # Union all objects per frame → [N, H, W]
        mask_out = (mask_out.amax(dim=1) > -1.0).float()

        return io.NodeOutput(mask_out)


class SAM3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SAM3_Detect,
            SAM3_VideoTrack,
        ]


async def comfy_entrypoint() -> SAM3Extension:
    return SAM3Extension()
