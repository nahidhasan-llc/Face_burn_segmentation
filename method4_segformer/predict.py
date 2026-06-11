"""
SegFormer Inference
Run: python method4_segformer/predict.py --input dataset/test/images --output outputs/segformer
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from utils.metrics import compute_metrics, print_metrics

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(ckpt_dir):
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt_dir).to(DEVICE)
    model.eval()
    return model, SegformerImageProcessor()


def predict_one(model, processor, img_path):
    img  = Image.open(img_path).convert('RGB')
    orig = np.array(img)
    h, w = orig.shape[:2]
    inputs = processor(images=img, return_tensors='pt')
    pv     = inputs['pixel_values'].to(DEVICE)
    with torch.no_grad():
        out = model(pixel_values=pv)
    logits = F.interpolate(out.logits, size=(h, w), mode='bilinear', align_corners=False)
    pred   = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return pred * 255


def overlay(img_path, mask, alpha=0.4):
    img    = cv2.imread(img_path)
    color  = np.zeros_like(img)
    color[mask > 128] = (0, 0, 255)
    blended = cv2.addWeighted(img, 1 - alpha, color, alpha, 0)
    cnts, _ = cv2.findContours((mask > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, cnts, -1, (0, 255, 255), 2)
    return blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default=os.path.join(BASE, 'dataset', 'test', 'images'))
    parser.add_argument('--output', default=os.path.join(BASE, 'outputs', 'segformer'))
    parser.add_argument('--ckpt',   default=os.path.join(BASE, 'checkpoints', 'segformer'))
    parser.add_argument('--gt',     default=os.path.join(BASE, 'dataset', 'test', 'masks'))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model, processor = load_model(args.ckpt)
    results = []

    for fname in [f for f in os.listdir(args.input) if f.endswith(('.jpg', '.png'))]:
        img_path = os.path.join(args.input, fname)
        mask     = predict_one(model, processor, img_path)
        stem     = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(args.output, stem + '_mask.png'),    mask)
        cv2.imwrite(os.path.join(args.output, stem + '_overlay.png'), overlay(img_path, mask))
        gt_path = os.path.join(args.gt, stem + '.png')
        if os.path.exists(gt_path):
            gt = np.array(Image.open(gt_path).convert('L'))
            results.append(compute_metrics(mask / 255., gt / 255.))
        print(f"  Processed: {fname}")

    if results:
        print_metrics(results, 'SegFormer-B5')


if __name__ == '__main__':
    main()
