"""
SAM 2 Predict — runs on test images, outputs segmentation masks.
Compares predicted masks with GT masks for evaluation.
RUN: python method1_sam2/predict.py
"""
import os, sys
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_DIR = os.path.join(BASE, 'segment-anything-2')
sys.path.insert(0, BASE)
sys.path.insert(0, SAM2_DIR)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from utils.metrics import compute_metrics, print_metrics

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
SAM2_CFG = os.path.join(SAM2_DIR, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
CKPT_IN  = os.path.join(BASE, 'checkpoints', 'sam2', 'sam2_hiera_large.pt')
CKPT_FT  = os.path.join(BASE, 'checkpoints', 'sam2', 'best.pth')
TEST_IMG = os.path.join(BASE, 'dataset', 'test', 'images')
TEST_MSK = os.path.join(BASE, 'dataset', 'test', 'masks')
OUT_DIR  = os.path.join(BASE, 'outputs', 'sam2')
IMG_SIZE = 1024


def get_bbox_from_mask(mask_np):
    """Extract bbox from GT mask — used as SAM2 location prompt."""
    h, w   = mask_np.shape[:2]
    ys, xs = np.where(mask_np > 0)
    if len(ys) == 0:
        return np.array([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=np.float32)
    x1 = xs.min() * IMG_SIZE / w
    y1 = ys.min() * IMG_SIZE / h
    x2 = xs.max() * IMG_SIZE / w
    y2 = ys.max() * IMG_SIZE / h
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


def load_model():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(SAM2_CFG, CKPT_IN, device=DEVICE)
    model.load_state_dict(torch.load(CKPT_FT, map_location=DEVICE))
    model.eval()
    predictor = SAM2ImagePredictor(model)
    return model, predictor


def predict_one(model, predictor, img_np, gt_np):
    """
    Predict segmentation mask.
    img_np : HxWx3 uint8 RGB
    gt_np  : HxW uint8 — used ONLY to get bbox prompt location
              (same as training — bbox tells model WHERE to look,
               model still has to find exact burn boundary itself)
    """
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    box   = get_bbox_from_mask(gt_np)

    with torch.no_grad():
        predictor.set_image(img_r)
        feats    = predictor._features
        img_emb  = feats['image_embed']
        high_res = feats['high_res_feats']
        box_t    = torch.from_numpy(box).to(DEVICE)
        sparse, dense = model.sam_prompt_encoder(
            points=None, boxes=box_t, masks=None)
        out = model.sam_mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = F.interpolate(logits, size=(h, w),
                               mode='bilinear', align_corners=False)
        pred   = (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy().astype(np.uint8) * 255
    return pred


def overlay(img_bgr, pred_mask, gt_mask, alpha=0.4):
    """Red = predicted burn, cyan = boundary, green = GT boundary."""
    vis = img_bgr.copy()
    # predicted burn area — red fill
    color = np.zeros_like(img_bgr)
    color[pred_mask > 128] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 1 - alpha, color, alpha, 0)
    # predicted boundary — cyan
    cnts, _ = cv2.findContours((pred_mask > 128).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    # GT boundary — green
    cnts_gt, _ = cv2.findContours((gt_mask > 128).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts_gt, -1, (0, 255, 0), 2)
    return vis


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, predictor = load_model()
    results = []

    print(f"Running SAM 2 predictions on test set...")
    for fname in sorted(os.listdir(TEST_IMG)):
        if not fname.endswith(('.jpg', '.png')):
            continue
        stem     = os.path.splitext(fname)[0]
        img_path = os.path.join(TEST_IMG, fname)
        msk_path = os.path.join(TEST_MSK, stem + '.png')

        img_np = np.array(Image.open(img_path).convert('RGB'))
        gt_np  = np.array(Image.open(msk_path).convert('L')) \
                 if os.path.exists(msk_path) \
                 else np.zeros(img_np.shape[:2], dtype=np.uint8)

        # predict
        pred = predict_one(model, predictor, img_np, gt_np)

        # save predicted mask
        cv2.imwrite(os.path.join(OUT_DIR, stem + '_mask.png'), pred)

        # save overlay (pred=red, GT=green boundary)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gt_resized = cv2.resize(gt_np, (img_np.shape[1], img_np.shape[0]))
        cv2.imwrite(os.path.join(OUT_DIR, stem + '_overlay.png'),
                    overlay(img_bgr, pred, gt_resized))

        # compute metrics vs GT
        if os.path.exists(msk_path):
            m = compute_metrics(pred / 255., gt_np / 255.)
            results.append(m)
            print(f"  {fname}")
            print(f"    Dice: {m['dice']:.4f}  IoU: {m['iou']:.4f}  "
                  f"Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}")

    if results:
        print_metrics(results, 'SAM 2')
    print(f"\nOutputs saved to: {OUT_DIR}")
    print(f"  *_mask.png    = predicted burn segmentation mask")
    print(f"  *_overlay.png = prediction (red) vs GT (green) on original image")


if __name__ == '__main__':
    main()
