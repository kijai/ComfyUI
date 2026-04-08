"""
SAM3 (Segment Anything 3) ComfyUI nodes for detection, segmentation, and video tracking.
"""

from typing_extensions import override

import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io


class SAM3_Detect(io.ComfyNode):
    """Open-vocabulary detection + segmentation using text prompts."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_Detect",
            display_name="SAM3 Detect",
            category="detection/",
            search_aliases=["sam3", "segment anything", "open vocabulary", "text detection"],
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Conditioning.Input("conditioning", display_name="conditioning", tooltip="Text conditioning from CLIPTextEncode"),
                io.Float.Input("threshold", display_name="threshold", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Int.Input("max_detections", display_name="max_detections", default=50, min=1, max=200),
            ],
            outputs=[
                io.BoundingBox.Output("bboxes"),
                io.Mask.Output("masks"),
            ],
        )

    @classmethod
    def execute(cls, model, image, conditioning, threshold, max_detections) -> io.NodeOutput:
        B, H, W, C = image.shape

        # Preprocess image to model input size (1008x1008)
        image_in = comfy.utils.common_upscale(image.movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")

        # Extract text embeddings and attention mask from conditioning
        text_embeddings = None
        text_mask = None
        if conditioning is not None and len(conditioning) > 0:
            cond_tensor = conditioning[0][0]  # (1, T, C) conditioning tensor
            cond_meta = conditioning[0][1]
            text_embeddings = cond_tensor
            # Extract attention mask (1=valid, 0=padding) from CLIP
            if "attention_mask" in cond_meta:
                text_mask = cond_meta["attention_mask"]  # (1, T) int, 1=valid
            else:
                # Fallback: assume all valid
                text_mask = torch.ones(cond_tensor.shape[0], cond_tensor.shape[1], dtype=torch.int64, device=cond_tensor.device)

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        image_in = image_in.to(device=device, dtype=dtype)
        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(device=device, dtype=dtype)
        if text_mask is not None:
            text_mask = text_mask.to(device)
        results = model.model.diffusion_model(
            image_in,
            text_embeddings=text_embeddings,
            text_mask=text_mask,
            threshold=threshold,
            orig_size=(H, W),
        )

        boxes = results["boxes"]  # (B, Q, 4) xyxy
        scores = results["scores"]  # (B, Q)
        masks = results["masks"]  # (B, Q, H, W)

        all_bbox_dicts = []
        all_masks = []

        for b in range(B):
            keep = scores[b] > threshold
            b_boxes = boxes[b][keep].cpu()
            b_scores = scores[b][keep].cpu()
            b_masks = masks[b][keep]  # (K, H, W)

            # Sort by score and limit
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
            all_masks.append(b_masks)

        if all_masks and all_masks[0].shape[0] > 0:
            if B == 1:
                # Single image: return all detection masks
                mask_out = (all_masks[0] > 0).float()
            else:
                # Batch: return best detection mask per image
                mask_out = torch.stack([(m[0] > 0).float() if m.shape[0] > 0 else torch.zeros(H, W, device=m.device) for m in all_masks])
        else:
            mask_out = torch.zeros((B, H, W), device=comfy.model_management.intermediate_device())

        return io.NodeOutput(all_bbox_dicts, mask_out)


class SAM3_Segment(io.ComfyNode):
    """Interactive segmentation using point or box prompts."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SAM3_Segment",
            display_name="SAM3 Segment",
            category="detection/",
            search_aliases=["sam3", "segment", "mask", "interactive"],
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.BoundingBox.Input("bboxes", display_name="bboxes", optional=True, tooltip="Bounding boxes to segment within"),
            ],
            outputs=[
                io.Mask.Output("masks"),
            ],
        )

    @classmethod
    def execute(cls, model, image, bboxes=None) -> io.NodeOutput:
        B, H, W, C = image.shape

        image_in = comfy.utils.common_upscale(image.movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")

        # Convert bboxes to normalized cxcywh format for the model
        boxes_tensor = None
        if bboxes is not None and len(bboxes) > 0:
            # bboxes is list[list[dict]] with x, y, width, height
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
                boxes_tensor = torch.stack(batch_boxes).to(image_in.device)

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        image_in = image_in.to(device=device, dtype=dtype)
        if boxes_tensor is not None:
            boxes_tensor = boxes_tensor.to(device=device, dtype=dtype)
        results = model.model.diffusion_model(
            image_in,
            boxes=boxes_tensor,
            orig_size=(H, W),
        )

        masks = results["masks"]
        mask_out = masks[:, 0]
        mask_out = (mask_out > 0).float()

        return io.NodeOutput(mask_out)


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
                io.Mask.Input("initial_mask", display_name="initial_mask", tooltip="Mask for the first frame to track"),
            ],
            outputs=[
                io.Mask.Output("masks", display_name="masks"),
            ],
        )

    @classmethod
    def execute(cls, images, model, initial_mask) -> io.NodeOutput:
        N, H, W, C = images.shape  # N = number of frames

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model

        frames = images.movedim(-1, 1)
        frames_in = comfy.utils.common_upscale(frames, 1008, 1008, "bilinear", crop="disabled").to(device=device, dtype=dtype)
        # initial_mask: [N_obj, H, W] — one mask per object to track
        init_masks = initial_mask.unsqueeze(1).to(device=device, dtype=dtype)  # [N_obj, 1, H, W]

        pbar = comfy.utils.ProgressBar(N)
        mask_logits = sam3_model.forward_video(images=frames_in, initial_mask=init_masks, pbar=pbar)
        # mask_logits: [N, N_obj, image_size, image_size]

        # Resize masks back to original resolution and binarize
        N_obj = mask_logits.shape[1]
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
            SAM3_Segment,
            SAM3_VideoTrack,
        ]


async def comfy_entrypoint() -> SAM3Extension:
    return SAM3Extension()
