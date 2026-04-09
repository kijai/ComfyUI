"""
SAM3 (Segment Anything 3) ComfyUI nodes for detection, segmentation, and video tracking.
"""

from typing_extensions import override

import json
import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
from comfy.ldm.sam3.tracker import fill_holes_in_mask_scores
from comfy_api.latest import ComfyExtension, io


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
                io.Float.Input("threshold", display_name="threshold", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Int.Input("max_detections", display_name="max_detections", default=1, min=1, max=200),
            ],
            outputs=[
                io.BoundingBox.Output("bboxes"),
                io.Mask.Output("masks"),
            ],
        )

    @classmethod
    def execute(cls, model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.3, max_detections=50) -> io.NodeOutput:
        B, H, W, C = image.shape

        image_in = comfy.utils.common_upscale(image.movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")

        # Extract text embeddings and attention mask from conditioning
        text_embeddings = None
        text_mask = None
        if conditioning is not None and len(conditioning) > 0:
            cond_tensor = conditioning[0][0]
            cond_meta = conditioning[0][1]
            text_embeddings = cond_tensor
            if "attention_mask" in cond_meta:
                text_mask = cond_meta["attention_mask"]
            else:
                text_mask = torch.ones(cond_tensor.shape[0], cond_tensor.shape[1], dtype=torch.int64, device=cond_tensor.device)

        # Convert bboxes to normalized cxcywh format
        boxes_tensor = None
        if bboxes is not None and len(bboxes) > 0:
            batch_boxes = []
            for frame_bboxes in bboxes:
                frame_tensors = []
                for d in frame_bboxes:
                    cx = (d["x"] + d["width"] / 2) / W
                    cy = (d["y"] + d["height"] / 2) / H
                    w = d["width"] / W
                    h = d["height"] / H
                    frame_tensors.append([cx, cy, w, h])
                if frame_tensors:
                    batch_boxes.append(torch.tensor(frame_tensors, dtype=torch.float32))
            if batch_boxes:
                boxes_tensor = torch.stack(batch_boxes)

        # Parse point prompts from JSON (KJNodes PointsEditor format: [{"x": int, "y": int}, ...])
        pos_pts = json.loads(positive_coords) if positive_coords else []
        neg_pts = json.loads(negative_coords) if negative_coords else []
        has_points = len(pos_pts) > 0 or len(neg_pts) > 0

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model

        if has_points:
            # Point prompts: use SAM interactive decoder (tracker path), per-image
            all_coords = [[p["x"] / W * 1008, p["y"] / H * 1008] for p in pos_pts] + \
                         [[p["x"] / W * 1008, p["y"] / H * 1008] for p in neg_pts]
            all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)
            point_inputs = {
                "point_coords": torch.tensor([all_coords], dtype=dtype, device=device),
                "point_labels": torch.tensor([all_labels], dtype=torch.int32, device=device),
            }
            all_masks = []
            pbar = comfy.utils.ProgressBar(B)
            for b in range(B):
                frame = image_in[b:b+1].to(device=device, dtype=dtype)
                mask_logits = sam3_model.forward_segment(frame, point_inputs=point_inputs)
                mask = F.interpolate(mask_logits, size=(H, W), mode="bilinear", align_corners=False)
                all_masks.append((mask[:, 0] > 0).float())
                pbar.update(1)
            mask_out = torch.cat(all_masks, dim=0)
            return io.NodeOutput([[] for _ in range(B)], mask_out)

        # Text / box detection: run per-image to avoid OOM
        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(device=device, dtype=dtype)
        if text_mask is not None:
            text_mask = text_mask.to(device)

        has_text = conditioning is not None and len(conditioning) > 0
        all_bbox_dicts = []
        all_masks = []
        pbar = comfy.utils.ProgressBar(B)

        for b in range(B):
            frame = image_in[b:b+1].to(device=device, dtype=dtype)
            b_boxes_tensor = None
            if boxes_tensor is not None:
                b_boxes_tensor = boxes_tensor[b:b+1].to(device=device, dtype=dtype)

            results = sam3_model(
                frame,
                text_embeddings=text_embeddings,
                text_mask=text_mask,
                boxes=b_boxes_tensor,
                threshold=threshold,
                orig_size=(H, W),
            )

            pred_boxes = results["boxes"][0]  # (Q, 4) xyxy
            frame_scores = results["scores"][0]  # (Q,)
            frame_masks = results["masks"][0]  # (Q, H, W)

            if has_text:
                keep = frame_scores > threshold
                b_boxes = pred_boxes[keep].cpu()
                b_scores = frame_scores[keep].cpu()
                b_masks = frame_masks[keep]

                order = b_scores.argsort(descending=True)[:max_detections]
                b_boxes = b_boxes[order]
                b_scores = b_scores[order]
                b_masks = b_masks[order]

                bbox_dicts = [
                    {
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "width": float(box[2] - box[0]),
                        "height": float(box[3] - box[1]),
                        "score": float(score),
                    }
                    for box, score in zip(b_boxes, b_scores)
                ]
                all_bbox_dicts.append(bbox_dicts)
                if b_masks.shape[0] > 0:
                    mask_logit = fill_holes_in_mask_scores(b_masks[0:1].unsqueeze(0), max_area=200)[0, 0]
                    all_masks.append((mask_logit > 0).float())
                else:
                    all_masks.append(torch.zeros(H, W, device=comfy.model_management.intermediate_device()))
            else:
                all_bbox_dicts.append([])
                mask_logit = fill_holes_in_mask_scores(frame_masks[0:1].unsqueeze(0), max_area=200)[0, 0]
                all_masks.append((mask_logit > 0).float())
            pbar.update(1)

        mask_out = torch.stack(all_masks)
        return io.NodeOutput(all_bbox_dicts, mask_out)


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

        N_obj = mask_logits.shape[1]
        # Clean masks at video resolution (fill holes, remove sprinkles) matching reference
        mask_logits = fill_holes_in_mask_scores(mask_logits.flatten(0, 1).unsqueeze(1), max_area=16)
        mask_logits = mask_logits.squeeze(1).view(N, N_obj, mask_logits.shape[-2], mask_logits.shape[-1])
        mask_out = F.interpolate(mask_logits, size=(H, W), mode="bilinear", align_corners=False)
        mask_out = (mask_out > 0).float()  # [N, N_obj, H, W]

        # Flatten to [N * N_obj, H, W] batch of masks
        mask_out = mask_out.view(N * N_obj, H, W)

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
