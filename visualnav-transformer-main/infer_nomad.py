# -*- coding: utf-8 -*-
"""
usage:
    python infer_nomad.py \
        --cfg train/config/nomad.yaml \
        --ckpt train/logs/nomad/nomad_2025_05_19_17_59_56/latest.pth \
        --obs test_imgs/obs.jpg \
        --goal test_imgs/goal.jpg
"""
import matplotlib.pyplot as plt, numpy as np
import argparse, torch, torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image

# 모델 factory 는 train/vint_train/models/__init__.py 에 정의
from train.vint_train.models.nomad.nomad import NoMaD   # pip install -e train/ 했기 때문에 import 가능

def load_image(path, size):
    tr = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    return tr(Image.open(path).convert("RGB")).unsqueeze(0)  # (1,3,H,W)

def main(args):
    cfg = OmegaConf.load(args.cfg)
    model = NoMaD(cfg)
    sd = torch.load(args.ckpt, map_location="cpu")

    # ----- NEW: checkpoint 형태에 따라 분기 -----------------
    if isinstance(sd, dict) and "model" in sd:              # (옛 버전 체크포인트)
        state_dict = sd["model"]
    elif isinstance(sd, dict) and "model_state_dict" in sd: # Lightning 등
        state_dict = sd["model_state_dict"]
    else:                                                   # 파라미터만 저장된 경우
        state_dict = sd
    model.load_state_dict(state_dict, strict=False)  # strict=False: 누락 키 무시
    # --------------------------------------------------------

    model.eval()

    obs  = load_image(args.obs,  cfg.image_size[::-1])      # [H,W] → [W,H]
    goal = load_image(args.goal, cfg.image_size[::-1])

    with torch.no_grad():
        # forward: (B,3,H,W) 두 장 → (B, len_traj_pred, 3)   (x, y, θ) 방식
        waypoints = model(obs, goal)        # shape: [1,8,3]  (예시 cfg 기준)
    print("predicted way-points:\n", waypoints.squeeze(0).numpy())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",  required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--obs",  required=True)
    p.add_argument("--goal", required=True)
    main(p.parse_args())
    
wp = np.loadtxt("out.txt")   # or use waypoints.numpy() directly
plt.plot(wp[:,0], wp[:,1], marker="o")      # XY 평면 경로
plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Predicted path")
plt.gca().invert_yaxis(); plt.axis("equal"); plt.show()