"""
UNet++ Inference — run on any image or folder.
Run: python method3_unetpp/predict.py --input dataset/test/images --output outputs/unetpp
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from utils.metrics import compute_metrics, print_metrics

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def load_model(ckpt_path):
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b4', encoder_weights=None,
        in_channels=3, classes=1, activation=None,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def predict_one(model, img_path):
    img = np.array(Image.open(img_path).convert('RGB'))
    h, w = img.shape[:2]
    inp  = transform(image=img)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(inp)
        prob  = torch.sigmoid(logit[0, 0]).cpu().numpy()
    prob = cv2.resize(prob, (w, h))
    return (prob > 0.5).astype(np.uint8) * 255


def overlay(img_path, mask, alpha=0.4):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask_r = cv2.resize(mask, (w, h))
    color  = np.zeros_like(img)
    color[mask_r > 128] = (0, 0, 255)
    blended = cv2.addWeighted(img, 1 - alpha, color, alpha, 0)
    cnts, _ = cv2.findContours((mask_r > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, cnts, -1, (0, 255, 255), 2)
    return blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default=os.path.join(BASE, 'dataset', 'test', 'images'))
    parser.add_argument('--output', default=os.path.join(BASE, 'outputs', 'unetpp'))
    parser.add_argument('--ckpt',   default=os.path.join(BASE, 'checkpoints', 'unetpp', 'best.pth'))
    parser.add_argument('--gt',     default=os.path.join(BASE, 'dataset', 'test', 'masks'), help='GT mask dir for metrics')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model   = load_model(args.ckpt)
    results = []

    imgs = [f for f in os.listdir(args.input) if f.endswith(('.jpg', '.png'))]
    for fname in imgs:
        img_path  = os.path.join(args.input, fname)
        mask      = predict_one(model, img_path)
        stem      = os.path.splitext(fname)[0]

        # save mask
        cv2.imwrite(os.path.join(args.output, stem + '_mask.png'), mask)
        # save overlay
        cv2.imwrite(os.path.join(args.output, stem + '_overlay.png'), overlay(img_path, mask))

        # metrics if GT available
        gt_path = os.path.join(args.gt, stem + '.png')
        if os.path.exists(gt_path):
            gt = np.array(Image.open(gt_path).convert('L'))
            results.append(compute_metrics(mask / 255., gt / 255.))

        print(f"  Processed: {fname}")

    if results:
        print_metrics(results, 'UNet++')


if __name__ == '__main__':
    main()
