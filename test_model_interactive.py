"""
Test any model on a single image or folder.

For SAM2/MedSAM: you will be asked to draw a bounding box
around the burn region (click and drag). This is the location
prompt the model needs — same as training.

For UNet++/SegFormer: no prompt needed, fully automatic.

RUN:
  python test_single_image.py --input path/to/image.jpg --model sam2
  python test_single_image.py --input path/to/folder   --model unetpp
  python test_single_image.py --input path/to/folder   --model segformer
  python test_single_image.py --input path/to/folder   --model medsam

For SAM2/MedSAM on a folder, you will draw a box for each image.
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

# ── Interactive bbox drawing ───────────────────────────────────────
drawing  = False
ix, iy   = -1, -1
rect     = [0, 0, 0, 0]
img_disp = None

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, rect, img_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy  = x, y
        rect    = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rect[2], rect[3] = x, y
        tmp = img_disp.copy()
        cv2.rectangle(tmp, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0), 2)
        cv2.imshow('Draw box around burn — press ENTER when done', tmp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect[2], rect[3] = x, y
        tmp = img_disp.copy()
        cv2.rectangle(tmp, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0), 2)
        cv2.imshow('Draw box around burn — press ENTER when done', tmp)


def get_user_bbox(img_np, img_size):
    """Opens a window for user to draw bbox. Returns box in img_size coords."""
    global img_disp, rect
    h, w     = img_np.shape[:2]
    img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # resize for display if too large
    scale    = min(800/w, 700/h, 1.0)
    disp_w, disp_h = int(w*scale), int(h*scale)
    img_disp = cv2.resize(img_bgr, (disp_w, disp_h))
    rect     = [0, 0, disp_w, disp_h]

    win = 'Draw box around burn — press ENTER when done'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_callback)
    print("  -> Draw a box around the burn area, then press ENTER")
    cv2.imshow(win, img_disp)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord('q'):  # ENTER or q
            break
    cv2.destroyAllWindows()

    # scale box back to original image coords, then to img_size
    x1 = min(rect[0], rect[2]) / scale * img_size / w
    y1 = min(rect[1], rect[3]) / scale * img_size / h
    x2 = max(rect[0], rect[2]) / scale * img_size / w
    y2 = max(rect[1], rect[3]) / scale * img_size / h
    print(f"  -> Box: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


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


def predict_sam2(model, predictor, img_np, box):
    h, w  = img_np.shape[:2]
    img_r = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    box_t = torch.from_numpy(box).to(DEVICE)
    with torch.no_grad():
        predictor.set_image(img_r)
        feats    = predictor._features
        sparse, dense = model.sam_prompt_encoder(
            points=None, boxes=box_t, masks=None)
        out = model.sam_mask_decoder(
            image_embeddings=feats['image_embed'],
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=feats['high_res_feats'],
        )
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = F.interpolate(logits, size=(h,w), mode='bilinear', align_corners=False)
        return (torch.sigmoid(logits[0,0]) > 0.5).cpu().numpy().astype(np.uint8) * 255


# ── MedSAM ────────────────────────────────────────────────────────
def load_medsam():
    from segment_anything import sam_model_registry
    model = sam_model_registry['vit_b'](
        checkpoint=os.path.join(BASE, 'checkpoints', 'medsam', 'medsam_vit_b.pth')).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(BASE, 'checkpoints', 'medsam', 'best.pth'), map_location=DEVICE))
    model.eval()
    return model, None


def predict_medsam(model, aux, img_np, box):
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
        logits = F.interpolate(logits, size=(h,w), mode='bilinear', align_corners=False)
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


def predict_unetpp(model, transform, img_np, box=None):
    h, w = img_np.shape[:2]
    inp  = transform(image=img_np)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(inp)[0,0]).cpu().numpy()
    prob = cv2.resize(prob, (w, h))
    return (prob > 0.5).astype(np.uint8) * 255


# ── SegFormer ─────────────────────────────────────────────────────
def load_segformer():
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    ckpt  = os.path.join(BASE, 'checkpoints', 'segformer')
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt).to(DEVICE)
    model.eval()
    return model, SegformerImageProcessor()


def predict_segformer(model, processor, img_np, box=None):
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
    'sam2':      (load_sam2,      predict_sam2,      True),   # needs_box
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


def process_image(img_path, model_name, model, aux, needs_box, out_dir):
    img_np = np.array(Image.open(img_path).convert('RGB'))
    stem   = os.path.splitext(os.path.basename(img_path))[0]
    _, predict_fn, _ = MODELS[model_name]

    box = None
    if needs_box:
        print(f"\n{os.path.basename(img_path)}")
        box = get_user_bbox(img_np, IMG_SIZE)

    pred = predict_fn(model, aux, img_np, box)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_mask.png'), pred)
    cv2.imwrite(os.path.join(out_dir, f'{stem}_overlay.png'),
                make_overlay(img_np, pred))

    burn_pct = 100 * (pred > 128).sum() / pred.size
    print(f"  Burn area: {burn_pct:.2f}%  ->  saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True,
                        help='Path to image or folder')
    parser.add_argument('--model',  default='sam2',
                        choices=['sam2', 'medsam', 'unetpp', 'segformer'])
    parser.add_argument('--output', default=None,
                        help='Output folder (default: outputs/single_test)')
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
        print(f"No images found"); return

    _, _, needs_box = MODELS[args.model]

    print(f"Device  : {DEVICE}")
    print(f"Model   : {args.model}")
    print(f"Images  : {len(images)}")
    if needs_box:
        print(f"NOTE: You will draw a bounding box for each image")
    print(f"Output  : {out_dir}\n")

    load_fn, _, _ = MODELS[args.model]
    model, aux    = load_fn()

    for img_path in images:
        process_image(img_path, args.model, model, aux, needs_box, out_dir)

    print(f"\nDone! Results in: {out_dir}")


if __name__ == '__main__':
    main()