import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from ode_pc import QUANTIZER_CLASSES
from quant_helper import QUANT_HELPER_CLS, QUANT_SCHEME_PC, replace_with_quant_layers


MISMATCH_LEVELS_5b = {
    0: 0.0, 15: 0.158, 14: 0.165, 13: 0.171, 12: 0.172, 11: 0.166, 10: 0.166, 9: 0.178, 8: 0.182,
    7: 0.207, 6: 0.261, 5: 0.278, 4: 0.237, 3: 0.352, 2: 0.361, 1: 0.290,
}


def get_parametrized_weight_mods(model):
    """
    Only applies to model with modules that have ONLY ONE parametrization.
    """
    out = {}
    for name, mod in model.named_modules():
        plist = getattr(getattr(mod, "parametrizations", None), "weight", None)
        if plist is not None:
            plist = list(plist)
            out[name] = plist[0].__class__.__name__ if len(plist) > 0 else None
    return out


def load_and_register_buffer(model: nn.Module, sd, device, parametrized_map=None, load_weight_only=False):
    if load_weight_only:
        # Load original un-parametrized weights only for the full_param checkpoints
        # Works for:
        # 1. Keep finetuning previously fine-tuned models;
        # 2. Loading fine-tuned model and test with a different setting as used in finetuning.
        _sd = {}
        for k in model.state_dict().keys():
            if not (k == "weight" or k.endswith(".weight")):
                continue

            if k in sd:
                _sd[k] = sd[k]
                continue

            parent_name, sep, child = k.rpartition(".")  # child == "weight"
            _k = f"{parent_name}.parametrizations.weight.original" if parent_name else "parametrizations.weight.original"
            if _k in sd:
                _sd[k] = sd[_k]

        return model.load_state_dict(_sd, strict=False)

    inc = model.load_state_dict(sd, strict=False)
    unexpected = list(getattr(inc, "unexpected_keys", []))
    if len(unexpected) == 0:
        return inc

    mod_map = dict(model.named_modules())
    # parametrize first
    parametrized_map = parametrized_map if parametrized_map is not None else {}
    param_str = ".parametrizations.weight"
    p_list = set([_.rpartition(param_str)[0] for _ in unexpected if param_str in _])
    for _p in p_list:
        parent = mod_map.get(_p, None)
        if parent is not None:
            P.register_parametrization(
                parent, "weight", QUANTIZER_CLASSES[parametrized_map[_p]]().to(device))

    for k in unexpected:
        parent_name, sep, child = k.rpartition(".")
        parent = mod_map.get(parent_name, None)
        if parent is None or hasattr(parent, child):
            continue
        parent.register_buffer(child, torch.empty_like(sd[k], device=device))
    return model.load_state_dict(sd, strict=True)


def get_quant_model(net, quant_cls, act_quant_cls, sigma_lsb, calib_loader, pvt_level=None, agg_bits=8,
                    w_quant_type="per_tensor", act_perc=0.9999, w_bits=4, act_bits=4, max_inp=None, device="cpu"):
    pc_conv_cls = net.PcConvs[0].__class__.__name__
    quant_scheme = QUANT_SCHEME_PC.get(pc_conv_cls, QUANT_SCHEME_PC["default"])
    for _k, _vd in quant_scheme.items():
        if "w_" in _k:
            # use max calibration for weights
            _vd.update({"quant_type": w_quant_type, "n_bits": w_bits})
        elif "act_" in _k:
            _vd.update({"quant_type": "per_tensor", "n_bits": act_bits, "calib_perc": act_perc})
    replace_with_quant_layers(net,
                              w_conv_quant=quant_scheme["w_conv"], act_conv_quant=quant_scheme["act_conv"],
                              w_conv_trans_quant=quant_scheme["w_conv_trans"],
                              act_conv_trans_quant=quant_scheme["act_conv_trans"],
                              w_linear_quant=quant_scheme["w_linear"], act_linear_quant=quant_scheme["act_linear"],
                              w_quant_cls=QUANT_HELPER_CLS[quant_cls], act_quant_cls=QUANT_HELPER_CLS[act_quant_cls],
                              adc_quant_cls=QUANT_HELPER_CLS[act_quant_cls], agg_bits=agg_bits,
                              sigma_lsb=sigma_lsb, pvt_level=pvt_level, max_inp=max_inp)
    # Run one forward batch for calibration
    if calib_loader is not None:
        calib_batch = next(iter(calib_loader))[0].to(device)
        _ = net(calib_batch)
    return quant_scheme


def format_df_col_name(col, col_fmt, data_type="R_vs_SpinV"):
    if data_type == "R_vs_SpinV":
        var_name = "v_spin" if col.endswith("X") else "R"
        v_ctrl = col.split("Vctrl=")[1].split(",")[0].replace(".", "p")
        temp = col.split("temperature=")[1].split(")")[0].replace(".", "p")
        return col_fmt.format(v_ctrl, temp, var_name)
    return col


def load_and_prepare_df(dir_path=os.path.dirname(os.path.abspath(__file__)), data_type="R_vs_SpinV", **kwargs):
    data_dir = os.path.join(dir_path, "hardware_data")
    if data_type == "R_vs_SpinV":
        file_name = "CU_Resis_vs_Spin_V_Finer.csv"
        col_fmt = "Vctrl{}Temp{}_{}"
        df = pd.read_csv(os.path.join(data_dir, file_name))
        df = df.rename(columns={_c: format_df_col_name(_c, col_fmt, data_type) for _c in df.columns})


# color space
def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / (1 + 0.055)) ** 2.4)


def linear_to_srgb(x):
    return torch.where(x <= 0.0031308, 12.92 * x, (1 + 0.055) * torch.pow(x, 1 / 2.4) - 0.055)


# mosaic, sRGB -> raw
def mosaic_rggb(rgb_lin):  # [3,H,W] linear [0,1]
    _, H, W = rgb_lin.shape
    raw = torch.empty((H, W), dtype=rgb_lin.dtype)
    raw[0::2, 0::2] = rgb_lin[0, 0::2, 0::2]  # R
    raw[0::2, 1::2] = rgb_lin[1, 0::2, 1::2]  # G1
    raw[1::2, 0::2] = rgb_lin[1, 1::2, 0::2]  # G2
    raw[1::2, 1::2] = rgb_lin[2, 1::2, 1::2]  # B
    return raw


def pack_rggb(raw):  # [H,W] -> [4,H/2,W/2]
    R = raw[0::2, 0::2]
    G1 = raw[0::2, 1::2]
    G2 = raw[1::2, 0::2]
    B = raw[1::2, 1::2]
    return torch.stack([R, G1, G2, B], dim=0)


def unpack_rggb(packed):  # [4,H/2,W/2] -> [H,W] mosaic
    R, G1, G2, B = packed
    H2, W2 = R.shape
    H, W = 2 * H2, 2 * W2
    raw = torch.zeros((H, W), dtype=packed.dtype)
    raw[0::2, 0::2] = R
    raw[0::2, 1::2] = G1
    raw[1::2, 0::2] = G2
    raw[1::2, 1::2] = B
    return raw


# demosaic (simple), raw -> sRGB
def _norm_blur(img, mask):
    k = torch.ones((1, 1, 3, 3), dtype=img.dtype)
    num = F.conv2d(img[None, None], k, padding=1)
    den = F.conv2d(mask[None, None], k, padding=1).clamp_min(1e-8)
    return (num / den)[0, 0]


def simple_demosaic_rggb(raw):  # raw: [H,W], float
    H, W = raw.shape
    MR = torch.zeros_like(raw)
    MR[0::2, 0::2] = 1
    MG1 = torch.zeros_like(raw)
    MG1[0::2, 1::2] = 1
    MG2 = torch.zeros_like(raw)
    MG2[1::2, 0::2] = 1
    MB = torch.zeros_like(raw)
    MB[1::2, 1::2] = 1

    R = _norm_blur(raw * MR, MR)
    G1 = _norm_blur(raw * MG1, MG1)
    G2 = _norm_blur(raw * MG2, MG2)
    B = _norm_blur(raw * MB, MB)
    G = 0.5 * (G1 + G2)
    return torch.stack([R, G, B], dim=0)  # [3,H,W], linear


# demosaic (OpenCV), raw -> sRGB
def cv2_demosaic_rggb(raw):  # raw: [H,W] float [0,1]
    raw16 = (raw.clamp(0, 1).numpy() * 4095.0).astype(np.uint16)  # 12-bit
    rgb = cv2.cvtColor(raw16, cv2.COLOR_BayerRG2BGR_EA)  # (H,W,3)
    rgb = rgb.astype(np.float32) / 4095.0
    return torch.from_numpy(rgb).permute(2, 0, 1)  # [3,H,W] linear


# transform for PyTorch dataset
class ToPackedRGGB:
    """3x32x32 sRGB -> 4x16x16 packed RGGB (float32 [0,1])"""

    def __init__(self, linearize=True, return_orig=False):
        self.linearize = linearize
        self.return_orig = return_orig

    def __call__(self, img):  # img: [3,32,32] sRGB
        orig = img.clone()
        x = img.clamp(0, 1)
        if self.linearize:
            x = srgb_to_linear(x)
        raw = mosaic_rggb(x)
        packed = pack_rggb(raw)
        return (packed, orig) if self.return_orig else packed


class RawImgDataset(Dataset):
    def __init__(self, root, train, transform=None):
        super().__init__()
        data_dir_path = os.path.join(root, "train" if train else "test")
        data_list = sorted(glob.glob(os.path.join(data_dir_path, "*.pt"), recursive=True))
        self.data, self.target = [], []
        for _dp in data_list:
            _data_dict = torch.load(_dp, map_location="cpu")
            self.data.append(_data_dict["data"].permute([0, 2, 3, 1]).contiguous().numpy().astype(np.uint8))
            self.target.append(_data_dict["targets"])
        self.data = np.concatenate(self.data)
        self.target = torch.cat(self.target).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx]), self.target[idx]
        return self.data[idx], self.target[idx]


# show previews
def make_previews(packed, orig_srgb):
    raw = unpack_rggb(packed)  # [32,32]

    # Demosaic with OpenCV and simple filter
    rgb_cv2_lin = cv2_demosaic_rggb(raw)
    rgb_simp_lin = simple_demosaic_rggb(raw)

    # Convert both back to sRGB for display
    rgb_cv2 = linear_to_srgb(rgb_cv2_lin.clamp(0, 1)).permute(1, 2, 0).numpy()
    rgb_simp = linear_to_srgb(rgb_simp_lin.clamp(0, 1)).permute(1, 2, 0).numpy()
    orig = orig_srgb.permute(1, 2, 0).numpy()

    return (np.clip(orig, 0, 1), np.clip(rgb_cv2, 0, 1), np.clip(rgb_simp, 0, 1))


# example usage
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), ToPackedRGGB(linearize=True, return_orig=True)])
    ds = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    (packed, orig_srgb), label = ds[0]  # packed: [4,16,16], orig_srgb: [3,32,32]

    # preview
    orig, recon_cv2, recon_simp = make_previews(packed, orig_srgb)
    plt.figure(figsize=(9, 3), dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(orig)
    plt.title("Original CIFAR-10")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(recon_cv2)
    plt.title("Reconstructed (cv2)")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(recon_simp)
    plt.title("Reconstructed (simple)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("preview.png")