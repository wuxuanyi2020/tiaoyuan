"""HomoFormer 去阴影模块封装（MOG2 前置去阴影）。"""

import os
import sys
import numpy as np
import cv2
import torch


class HomoFormerShadowRemover:
    """封装 HomoFormer 模型用于图像阴影去除。

    加载 HomoFormer-CVPR2024 预训练权重，对输入 BGR ROI 图像进行去阴影处理。
    输入: BGR 图像 (H, W, 3), uint8
    输出: 去阴影后的 BGR 图像 (H, W, 3), uint8
    """

    def __init__(self, weights_path=None):
        self.model = None
        self.device = torch.device("cpu")
        self._loaded = False

        if weights_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            weights_path = os.path.join(base, "HomoFormer-master", "SRD.pth")
            if not os.path.exists(weights_path):
                # 尝试当前目录
                weights_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "HomoFormer-master", "SRD.pth",
                )
        self._weights_path = weights_path

        if os.path.exists(weights_path):
            self._load_model(weights_path)
        else:
            print(f"[HomoFormer] 权重文件未找到: {weights_path}")

    def _load_model(self, weights_path):
        """加载 HomoFormer 模型。"""
        try:
            # 临时将 HomoFormer-master 加入 sys.path
            homo_dir = os.path.dirname(weights_path)
            if homo_dir not in sys.path:
                sys.path.insert(0, homo_dir)

            import argparse
            import utils as homo_utils

            parser = argparse.ArgumentParser()
            parser.add_argument("--arch", default="HomoFormer", type=str)
            parser.add_argument("--embed_dim", type=int, default=32)
            parser.add_argument("--win_size", type=int, default=8)
            parser.add_argument("--token_projection", type=str, default="linear")
            parser.add_argument("--token_mlp", type=str, default="leff")
            parser.add_argument("--vit_dim", type=int, default=256)
            parser.add_argument("--vit_depth", type=int, default=12)
            parser.add_argument("--vit_nheads", type=int, default=8)
            parser.add_argument("--vit_mlp_dim", type=int, default=512)
            parser.add_argument("--vit_patch_size", type=int, default=16)
            parser.add_argument("--global_skip", action="store_true", default=False)
            parser.add_argument("--local_skip", action="store_true", default=False)
            parser.add_argument("--vit_share", action="store_true", default=False)
            parser.add_argument("--train_ps", type=int, default=320)
            parser.add_argument("--plus", action="store_true", default=False)
            args = parser.parse_args([])

            model = homo_utils.get_arch(args)
            # 单卡 CPU，不包装 DataParallel
            checkpoint = torch.load(
                weights_path,
                map_location=self.device,
                weights_only=False,
            )
            state_dict = checkpoint["state_dict"]
            # 去掉 DataParallel 的 module. 前缀
            from collections import OrderedDict
            new_sd = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if "module." in k else k
                new_sd[name] = v
            model.load_state_dict(new_sd, strict=True)
            model.eval()

            self.model = model
            self._loaded = True
            print(f"[HomoFormer] 模型已加载: {os.path.basename(weights_path)}")
        except Exception as e:
            print(f"[HomoFormer] 加载失败: {e}")
            self._loaded = False

    @property
    def ready(self):
        return self._loaded and self.model is not None

    def remove_shadow(self, img_bgr, mask=None):
        """对输入 BGR 图像进行去阴影处理。

        Args:
            img_bgr: BGR 图像 (H, W, 3) uint8
            mask:   可选阴影区域 mask (H, W) uint8，0=阴影区域，255=非阴影
                    若为 None 则使用全 1 mask（整体去阴影）

        Returns:
            去阴影后的 BGR 图像 (H, W, 3) uint8，若失败返回原图
        """
        if not self.ready:
            return img_bgr

        try:
            h, w = img_bgr.shape[:2]
            # BGR -> RGB -> float32 [0,1]
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # 构造 mask
            if mask is None:
                mask_np = np.ones((h, w), dtype=np.float32)
            else:
                mask_np = mask.astype(np.float32) / 255.0

            # pad 到 img_multiple_of (64) 的倍数
            mul = 64
            pad_h = (mul - h % mul) % mul
            pad_w = (mul - w % mul) % mul
            if pad_h > 0 or pad_w > 0:
                rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w)), mode="reflect")

            # 转 tensor: (C,H,W) 并 batch
            inp_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

            # 分块推理（原图可能很大，按 tile 处理）
            tile = 384
            overlap = 30
            if min(rgb.shape[0], rgb.shape[1]) >= tile:
                result = self._tile_infer(inp_t, mask_t, tile, overlap)
            else:
                result = self.model(inp_t, mask_t)

            # 转回 numpy
            out = result.squeeze().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            # 裁回原始尺寸
            out = out[:h, :w]
            out = (out * 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            return out

        except Exception as e:
            print(f"[HomoFormer] 推理失败: {e}")
            return img_bgr

    def _tile_infer(self, inp, mask, tile_size, overlap):
        """分块推理大图。"""
        from utils.image_utils import splitimage, mergeimage
        B, C, H, W = inp.shape
        split_data, starts = splitimage(inp, crop_size=tile_size, overlap_size=overlap)
        mask_data, _ = splitimage(mask, crop_size=tile_size, overlap_size=overlap)
        restored_list = []
        for i, (data, m_) in enumerate(zip(split_data, mask_data)):
            with torch.no_grad():
                out = self.model(data, m_)
            restored_list.append(out)
        merged = mergeimage(restored_list, starts, crop_size=tile_size, resolution=(B, C, H, W))
        return merged
