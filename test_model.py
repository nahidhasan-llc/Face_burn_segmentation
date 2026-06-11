"""
Test either model on a single image or entire folder of images.
No labels or masks needed — just raw images.

RUN on single image:
  python test_single_image.py --input path/to/image.jpg --model sam2

RUN on folder:
  python test_single_image.py --input path/to/folder --model sam2
  python test_single_image.py --input path/to/folder --model medsam
"""
import os, sys, argparse
BASE     = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR = os.path.join(BASE, 'segment-anything-2')
sys.path.insert(0, BASE)
sys.path.insert(0, SAM2_DIR)

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 1024
FULL_BOX = np.array([[0, 0, IMG_SIZE, IMG_SIZE]], dtype=np.float32)
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


# ── SAM2 ──────────────────────────────────────────────────────────
def load_sam2():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg   = os.path.join(SAM2_DIR, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')
    ckpt  = os.path.join(BASE, 'checkpoints', 'sam2', 'sam2_hiera_large.pt')
    ft    = os.path.join(BASE, 'checkpoints', 'sam2', 'best.pth')
    model = build_sam2(cfg, ckpt, device=DEVICE)
    model.load_state_dict(torch.load(ft, map_location=DEVICE))
    model.eval()
    predictor = SAM2ImagePredictor(model)
    return model, predictor


def predict_sam2(model, predictor, img_np):
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    box_t = torch.from_numpy(FULL_BOX).to(DEVICE)
    with torch.no_grad():
        predictor.set_image(img_r)
        feats    = predictor._features
        img_emb  = feats['image_embed']
        high_res = feats['high_res_feats']
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


# ── SAM ViT-B (method2) ───────────────────────────────────────────
def load_samvitb():
    from segment_anything import sam_model_registry
    ckpt  = os.path.join(BASE, 'checkpoints', 'medsam', 'medsam_vit_b.pth')
    ft    = os.path.join(BASE, 'checkpoints', 'medsam', 'best.pth')
    model = sam_model_registry['vit_b'](checkpoint=ckpt).to(DEVICE)
    model.load_state_dict(torch.load(ft, map_location=DEVICE))
    model.eval()
    return model


def predict_samvitb(model, img_np):
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy(img_r).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.
    box_t = torch.from_numpy(FULL_BOX).to(DEVICE)
    with torch.no_grad():
        img_emb       = model.image_encoder(img_t)
        sparse, dense = model.prompt_encoder(
            points=None, boxes=box_t.unsqueeze(1), masks=None)
        logits, _     = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = F.interpolate(logits, size=(h, w),
                               mode='bilinear', align_corners=False)
        pred   = (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy().astype(np.uint8) * 255
    return pred


# ── Shared ────────────────────────────────────────────────────────
def make_overlay(img_np, pred):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    color   = np.zeros_like(img_bgr)
    color[pred > 128] = (0, 0, 255)
    overlay = cv2.addWeighted(img_bgr, 0.6, color, 0.4, 0)
    cnts, _ = cv2.findContours((pred > 128).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2)
    return overlay


def process_image(img_path, model_name, model, predictor, out_dir):
    img_np   = np.array(Image.open(img_path).convert('RGB'))
    stem     = os.path.splitext(os.path.basename(img_path))[0]

    if model_name == 'sam2':
        pred = predict_sam2(model, predictor, img_np)
    else:
        pred = predict_samvitb(model, img_np)

    # save outputs
    cv2.imwrite(os.path.join(out_dir, f'{stem}_mask.png'), pred)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_overlay.png'),
                make_overlay(img_np, pred))

    burn_pct = 100 * (pred > 128).sum() / pred.size
    print(f"  {os.path.basename(img_path):60s} burn: {burn_pct:.2f}%")


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True,
                        help='Path to single image OR folder of images')
    parser.add_argument('--model',  default='sam2',
                        choices=['sam2', 'medsam'],
                        help='Which model to use (default: sam2)')
    parser.add_argument('--output', default=None,
                        help='Output folder (default: outputs/single_test)')
    args = parser.parse_args()

    out_dir = args.output or os.path.join(BASE, 'outputs', 'single_test')
    os.makedirs(out_dir, exist_ok=True)

    # collect images — only direct files in folder, no subfolders
    if os.path.isfile(args.input):
        images = [args.input]
    elif os.path.isdir(args.input):
        images = [
            os.path.join(args.input, f)
            for f in sorted(os.listdir(args.input))
            if f.lower().endswith(IMG_EXTS)
            and os.path.isfile(os.path.join(args.input, f))  # direct files only
        ]
    else:
        print(f"ERROR: {args.input} is not a valid file or folder")
        return

    if not images:
        print(f"No images found in {args.input}")
        return

    print(f"Device  : {DEVICE}")
    print(f"Model   : {args.model}")
    print(f"Images  : {len(images)}")
    print(f"Output  : {out_dir}")
    print()

    # load model once
    if args.model == 'sam2':
        model, predictor = load_sam2()
    else:
        model    = load_samvitb()
        predictor = None

    # process all images
    for img_path in images:
        process_image(img_path, args.model, model, predictor, out_dir)

    print(f"\nDone! Results saved to: {out_dir}")
    print(f"  *_mask.png    = binary burn mask")
    print(f"  *_overlay.png = burn area (red) on original image")


if __name__ == '__main__':
    main()



# Single image:
#     python test_model.py --input "C:\path\to\image.jpg" --model sam2
# Entire folder:
#     python test_model.py --input "C:\path\to\folder" --model sam2
# Custom output folder:
#     python test_model.py --input "C:\path\to\folder" --model sam2 --output "C:\path\to\results"

