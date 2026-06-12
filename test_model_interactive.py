"""
Test any model on a single image or folder.

For SAM2/MedSAM:
  - LEFT CLICK  = add polygon point
  - RIGHT CLICK = undo last point
  - ENTER       = confirm polygon and run model
  - R           = reset polygon

For UNet++/SegFormer: fully automatic, no interaction needed.

RUN:
  python test_single_image.py --input path/to/image.jpg --model sam2
  python test_single_image.py --input path/to/folder   --model sam2
  python test_single_image.py --input path/to/folder   --model unetpp
  python test_single_image.py --input path/to/folder   --model segformer
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
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# ── Polygon drawing state ──────────────────────────────────────────
points   = []
img_disp = None
scale    = 1.0


def mouse_callback(event, x, y, flags, param):
    global points, img_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            redraw()


def redraw():
    global img_disp
    tmp = img_disp.copy()
    # draw filled polygon preview
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        overlay = tmp.copy()
        cv2.fillPoly(overlay, [pts], (0, 120, 255))
        cv2.addWeighted(overlay, 0.25, tmp, 0.75, 0, tmp)
        cv2.polylines(tmp, [pts], isClosed=True,
                      color=(0, 200, 255), thickness=2)
    # draw points and connecting lines
    for i, pt in enumerate(points):
        cv2.circle(tmp, pt, 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(tmp, points[i-1], pt, (0, 255, 0), 1)
    # instructions
    cv2.putText(tmp, 'LEFT=add point  RIGHT=undo  ENTER=confirm  R=reset',
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(tmp, 'LEFT=add point  RIGHT=undo  ENTER=confirm  R=reset',
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
    cv2.putText(tmp, f'Points: {len(points)}  (need at least 3)',
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 2)
    cv2.putText(tmp, f'Points: {len(points)}  (need at least 3)',
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,150,0), 1)
    cv2.imshow('Draw polygon around burn area', tmp)


def get_user_polygon(img_np, img_size):
    """
    Opens a window. User clicks multiple points to form a rough polygon.
    Returns bounding box of that polygon in img_size coords,
    AND the polygon mask (so we can use it as SAM2 mask prompt).
    """
    global points, img_disp, scale
    points   = []
    h, w     = img_np.shape[:2]
    scale    = min(900/w, 750/h, 1.0)
    disp_w   = int(w * scale)
    disp_h   = int(h * scale)
    img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_disp = cv2.resize(img_bgr, (disp_w, disp_h))

    win = 'Draw polygon around burn area'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_callback)
    redraw()

    print("  LEFT CLICK = add point | RIGHT CLICK = undo | ENTER = confirm | R = reset")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(points) >= 3:   # ENTER
            break
        elif key == ord('r') or key == ord('R'):
            points = []
            redraw()
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

    if len(points) < 3:
        print("  No polygon drawn — using full image")
        return (np.array([[0, 0, img_size, img_size]], dtype=np.float32), None)

    # scale points back to original image coords
    orig_pts = np.array([(int(x / scale), int(y / scale))
                         for x, y in points], dtype=np.int32)

    # bounding box of the polygon
    x1, y1 = orig_pts[:,0].min(), orig_pts[:,1].min()
    x2, y2 = orig_pts[:,0].max(), orig_pts[:,1].max()

    # scale bbox to img_size
    box = np.array([[
        x1 * img_size / w,
        y1 * img_size / h,
        x2 * img_size / w,
        y2 * img_size / h,
    ]], dtype=np.float32)

    # create polygon mask at img_size (for optional mask prompt)
    MASK_SIZE = 256
    mask_pts = np.array([[
        int(px * MASK_SIZE / w),
        int(py * MASK_SIZE / h),
    ] for px, py in orig_pts], dtype=np.int32)
    poly_mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
    cv2.fillPoly(poly_mask, [mask_pts], 1.0)
    poly_mask_t = torch.from_numpy(poly_mask).unsqueeze(0).unsqueeze(0)  # 1x1xHxW

    print(f"  Polygon: {len(points)} points  BBox: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    return box, poly_mask_t


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
    return model, SAM2ImagePredictor(model)


def predict_sam2(model, predictor, img_np, prompt):
    """prompt = (box, poly_mask_t) from get_user_polygon"""
    box, poly_mask = prompt
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    box_t = torch.from_numpy(box).to(DEVICE)

    with torch.no_grad():
        predictor.set_image(img_r)
        feats    = predictor._features
        img_emb  = feats['image_embed']
        high_res = feats['high_res_feats']

        # use polygon mask as additional prompt if available
        mask_input = poly_mask.to(DEVICE) if poly_mask is not None else None

        sparse, dense = model.sam_prompt_encoder(
            points=None, boxes=box_t,
            masks=mask_input)
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
        logits = F.interpolate(logits, size=(h,w),
                               mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits[0,0]) > 0.5).cpu().numpy().astype(np.uint8) * 255


# ── MedSAM ────────────────────────────────────────────────────────
def load_medsam():
    from segment_anything import sam_model_registry
    ckpt_in = os.path.join(BASE, 'checkpoints', 'medsam', 'medsam_vit_b.pth')
    ckpt_ft = os.path.join(BASE, 'checkpoints', 'medsam', 'best.pth')
    model = sam_model_registry['vit_b'](checkpoint=ckpt_in).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_ft, map_location=DEVICE))
    model.eval()
    return model, None


def predict_medsam(model, aux, img_np, prompt):
    box, _ = prompt
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy(img_r).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.
    box_t = torch.from_numpy(box).to(DEVICE)
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
        logits = F.interpolate(logits, size=(h,w),
                               mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits[0,0]) > 0.5).cpu().numpy().astype(np.uint8) * 255


# ── UNet++ ────────────────────────────────────────────────────────
def load_unetpp():
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b4', encoder_weights=None,
        in_channels=3, classes=1, activation=None).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(BASE, 'checkpoints', 'unetpp', 'best.pth'), map_location=DEVICE))
    model.eval()
    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    return model, transform


def predict_unetpp(model, transform, img_np, prompt=None):
    h, w = img_np.shape[:2]
    inp  = transform(image=img_np)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(inp)[0,0]).cpu().numpy()
    prob = cv2.resize(prob, (w, h))
    return (prob > 0.5).astype(np.uint8) * 255


# ── SegFormer ─────────────────────────────────────────────────────
def load_segformer():
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    model = SegformerForSemanticSegmentation.from_pretrained(
        os.path.join(BASE, 'checkpoints', 'segformer')).to(DEVICE)
    model.eval()
    return model, SegformerImageProcessor()


def predict_segformer(model, processor, img_np, prompt=None):
    h, w   = img_np.shape[:2]
    inputs = processor(images=Image.fromarray(img_np), return_tensors='pt')
    pv     = inputs['pixel_values'].to(DEVICE)
    with torch.no_grad():
        out    = model(pixel_values=pv)
        logits = F.interpolate(out.logits, size=(h,w),
                               mode='bilinear', align_corners=False)
        return logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8) * 255


# ── Config ────────────────────────────────────────────────────────
MODELS = {
    'sam2':      (load_sam2,      predict_sam2,      True),
    'medsam':    (load_medsam,    predict_medsam,    True),
    'unetpp':    (load_unetpp,    predict_unetpp,    False),
    'segformer': (load_segformer, predict_segformer, False),
}


def make_overlay(img_np, pred):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    color   = np.zeros_like(img_bgr)
    color[pred > 128] = (0, 0, 255)
    overlay = cv2.addWeighted(img_bgr, 0.6, color, 0.4, 0)
    cnts, _ = cv2.findContours((pred > 128).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2)
    return overlay


def process_image(img_path, model_name, model, aux, needs_prompt, out_dir):
    img_np = np.array(Image.open(img_path).convert('RGB'))
    stem   = os.path.splitext(os.path.basename(img_path))[0]
    _, predict_fn, _ = MODELS[model_name]

    prompt = None
    if needs_prompt:
        print(f"\n{os.path.basename(img_path)}")
        prompt = get_user_polygon(img_np, IMG_SIZE)

    pred = predict_fn(model, aux, img_np, prompt)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_mask.png'), pred)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_overlay.png'),
                make_overlay(img_np, pred))

    burn_pct = 100 * (pred > 128).sum() / pred.size
    print(f"  Burn area: {burn_pct:.2f}%  ->  {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--model',  default='sam2',
                        choices=['sam2','medsam','unetpp','segformer'])
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    out_dir = args.output or os.path.join(BASE, 'outputs', 'single_test')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(args.input):
        images = [args.input]
    elif os.path.isdir(args.input):
        images = [os.path.join(args.input, f)
                  for f in sorted(os.listdir(args.input))
                  if f.lower().endswith(IMG_EXTS)
                  and os.path.isfile(os.path.join(args.input, f))]
    else:
        print(f"ERROR: {args.input} not found"); return

    if not images:
        print("No images found"); return

    _, _, needs_prompt = MODELS[args.model]
    print(f"Device  : {DEVICE}")
    print(f"Model   : {args.model}")
    print(f"Images  : {len(images)}")
    if needs_prompt:
        print("NOTE: Draw a polygon around the burn area for each image")
        print("      LEFT=add point | RIGHT=undo | ENTER=confirm | R=reset")
    print(f"Output  : {out_dir}\n")

    load_fn, _, _ = MODELS[args.model]
    model, aux    = load_fn()

    for img_path in images:
        process_image(img_path, args.model, model, aux, needs_prompt, out_dir)

    print(f"\nDone! Results in: {out_dir}")
    print("  *_mask.png    = binary burn mask")
    print("  *_overlay.png = burn area (red) on original image")


if __name__ == '__main__':
    main()