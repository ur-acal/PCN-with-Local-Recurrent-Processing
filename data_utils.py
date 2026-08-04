import os
import glob
import csv
import re
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


MISMATCH_LEVELS_5b = {0: 0.0, 15: 0.181, 14: 0.179, 13: 0.177, 12: 0.175, 11: 0.172, 10: 0.169, 9: 0.165, 8: 0.16,
                      7: 0.153, 6: 0.142, 5: 0.125, 4: 0.1, 3: 0.072, 2: 0.105, 1: 0.342}

_CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

_PROCESS_ORDER = ("tt", "ff", "ss", "fs", "sf")
_PROCESS_ALIASES = {
    "tt": "tt", "ttg": "tt",
    "ff": "ff", "ffg": "ff", "ffag": "ff",
    "ss": "ss", "ssg": "ss", "ssag": "ss",
    "fs": "fs", "fsg": "fs",
    "sf": "sf", "sfg": "sf",
}


def _canonical_process(value):
    token = str(value).lower().strip()
    if token.startswith("top_"):
        token = token[4:]
    token = token.split("_mismatch", 1)[0]
    if token not in _PROCESS_ALIASES:
        raise ValueError("Unknown process-corner label: {}".format(value))
    return _PROCESS_ALIASES[token]


def _summary_corner_rows(path):
    frame = pd.read_csv(path, encoding="latin1")
    process_column = next(
        name for name in frame.columns if name.lower().startswith("process"))
    voltage_column = next(
        name for name in frame.columns if name.lower().startswith("voltage"))
    temperature_column = next(
        name for name in frame.columns if name.lower().startswith("temperature"))
    voltages = sorted(float(value) for value in frame[voltage_column].unique())
    temperatures = sorted(
        float(value) for value in frame[temperature_column].unique())
    output = {}
    for _, row in frame.iterrows():
        key = (
            _canonical_process(row[process_column]),
            voltages.index(float(row[voltage_column])),
            temperatures.index(float(row[temperature_column])),
        )
        output[key] = {
            "mean": float(row["Mean"]),
            "std": float(row["Std. Dev."]),
            "voltage": float(row[voltage_column]),
            "temperature": float(row[temperature_column]),
        }
    return output


def _paired_curve_count(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] % 2 != 0:
        raise ValueError("Expected paired X/Y columns in {}.".format(path))
    return data.shape[1] // 2


class MC45CornerData:
    """Align the repository MC data by process and ordered V/T levels."""

    def __init__(self, root,
                 spin_variation_source="PVT_Monte_Carlo_Results_SPIN.csv",
                 dtc_pulse_width_variation_source=
                 "PVT_45corner_DTC_pulse_width.csv",
                 relu_monte_carlo_source="relu_monteCarlo",
                 coupler_nonlinear_variation_source="coupler_monte",
                 coupler_nonlinear_variation_quantity=None,
                 coupler_nominal_R=67e3):
        self.root = os.path.abspath(os.fspath(root))
        spin_source = os.fspath(spin_variation_source)
        dtc_source = os.fspath(dtc_pulse_width_variation_source)
        relu_source = os.fspath(relu_monte_carlo_source)
        self.spin_source_path = (
            spin_source if os.path.isabs(spin_source)
            else os.path.join(self.root, spin_source))
        self.dtc_source_path = (
            dtc_source if os.path.isabs(dtc_source)
            else os.path.join(self.root, dtc_source))
        self.relu_source_path = (
            relu_source if os.path.isabs(relu_source)
            else os.path.join(self.root, relu_source))
        source = os.fspath(coupler_nonlinear_variation_source)
        self.coupler_source_path = (
            source if os.path.isabs(source)
            else os.path.join(self.root, source))
        self.coupler_nominal_R = float(coupler_nominal_R)
        quantity = (
            None if coupler_nonlinear_variation_quantity is None
            else str(coupler_nonlinear_variation_quantity).lower())
        if quantity not in {None, "conductance", "resistance"}:
            raise ValueError(
                "coupler_nonlinear_variation_quantity must be conductance "
                "or resistance.")
        self.coupler_quantity = quantity

        self.spin = _summary_corner_rows(self.spin_source_path)
        self.dtc = _summary_corner_rows(self.dtc_source_path)
        self.relu = self._load_relu_files()
        if os.path.isdir(self.coupler_source_path):
            self.coupler = self._load_coupler_folder()
        elif os.path.isfile(self.coupler_source_path):
            self.coupler = self._load_coupler_file()
        else:
            raise FileNotFoundError(
                "Coupler nonlinear-variation source not found: {}".format(
                    self.coupler_source_path))

        expected = set(self.spin)
        for name, values in (
                ("spin", self.spin), ("DTC", self.dtc),
                ("ReLU", self.relu), ("coupler", self.coupler)):
            if set(values) != expected:
                missing = sorted(expected - set(values))
                extra = sorted(set(values) - expected)
                raise ValueError(
                    "{} corner mismatch: missing={}, extra={}.".format(
                        name, missing, extra))

        self.corners = []
        for process in _PROCESS_ORDER:
            for voltage_level in range(3):
                for temperature_level in range(3):
                    key = (process, voltage_level, temperature_level)
                    self.corners.append({
                        "id": "{}_V{}_T{}".format(
                            process.upper(), voltage_level, temperature_level),
                        "process": process,
                        "voltage_level": voltage_level,
                        "temperature_level": temperature_level,
                        "spin": self.spin[key],
                        "dtc": self.dtc[key],
                        "relu": self.relu[key],
                        "coupler": self.coupler[key],
                    })

    def _load_coupler_folder(self):
        paths = sorted(glob.glob(os.path.join(
            self.coupler_source_path, "*.csv")))
        if not paths:
            raise ValueError(
                "Coupler nonlinear-variation folder contains no CSV files: "
                "{}".format(self.coupler_source_path))
        temperatures = sorted({
            float(re.match(r"[a-z]+_(-?[0-9.]+)_[0-2]\.csv\Z",
                           os.path.basename(path)).group(1))
            for path in paths
        })
        output = {}
        for path in paths:
            match = re.match(
                r"([a-z]+)_(-?[0-9.]+)_([0-2])\.csv\Z",
                os.path.basename(path))
            process = _canonical_process(match.group(1))
            temperature = float(match.group(2))
            voltage_level = int(match.group(3))
            key = (process, voltage_level, temperatures.index(temperature))
            count = _paired_curve_count(path)
            output[key] = {
                "path": path, "curve_indices": list(range(count)),
                "quantity": self.coupler_quantity or "conductance",
                "nominal_R": self.coupler_nominal_R,
                "temperature": temperature,
            }
        return output

    def _load_relu_files(self):
        output = {}
        paths = sorted(glob.glob(os.path.join(
            self.relu_source_path, "*.csv")))
        for path in paths:
            match = re.match(
                r"relu_([a-z]+)([0-8])\.csv\Z", os.path.basename(path))
            process = _canonical_process(match.group(1))
            voltage_level, temperature_level = divmod(int(match.group(2)), 3)
            with open(path) as handle:
                header = handle.readline()
            vdd_match = re.search(r"VDD_VALUE=([-+0-9.eE]+)", header)
            temp_match = re.search(r"temperature=([-+0-9.eE]+)", header)
            count = _paired_curve_count(path)
            output[(process, voltage_level, temperature_level)] = {
                "path": path, "curve_indices": list(range(count)),
                "voltage": float(vdd_match.group(1)),
                "temperature": float(temp_match.group(1)),
            }
        return output

    def _load_coupler_file(self):
        path = self.coupler_source_path
        with open(path, newline="") as handle:
            header = next(csv.reader(handle))
        records = []
        pattern = re.compile(
            r"top_([a-z]+)_mismatch,.*?VDD=([-+0-9.eE]+),"
            r"temperature=([-+0-9.eE]+),mcparamset=([0-9]+)\) X\Z")
        for column_index in range(0, len(header), 2):
            match = pattern.search(header[column_index])
            if match is None:
                raise ValueError(
                    "Cannot parse alternative CU column: {}".format(
                        header[column_index]))
            records.append((
                column_index // 2, _canonical_process(match.group(1)),
                float(match.group(2)), float(match.group(3))))
        voltages = sorted({record[2] for record in records})
        temperatures = sorted({record[3] for record in records})
        output = {}
        for curve_index, process, voltage, temperature in records:
            key = (process, voltages.index(voltage), temperatures.index(temperature))
            entry = output.setdefault(key, {
                "path": path, "curve_indices": [],
                "quantity": self.coupler_quantity or "resistance",
                "nominal_R": self.coupler_nominal_R,
                "temperature": temperature,
            })
            entry["curve_indices"].append(curve_index)
        return output


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

class PackedRGGBToRGB(nn.Module):
    """
    Convert packed RGGB tensor [4, H, W] to approximate RGB tensor [3, 2H, 2W].
    Channel order assumed: [R, G1, G2, B].
    """

    def forward(self, x):
        if x.ndim != 3 or x.size(0) != 4:
            raise ValueError(f"Expected [4,H,W], got {tuple(x.shape)}")

        r = x[0:1]
        g = 0.5 * (x[1:2] + x[2:3])
        b = x[3:4]

        rgb = torch.cat([r, g, b], dim=0)
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return rgb

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