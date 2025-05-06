import torch.nn as nn
import torch
from pc_model import PCNet

model_path = 'checkpoint/PredNetBpD_5_30CLS_FalseNes_0.001WD_FalseTIED_4REP_best_ckpt.t7'
checkpoint_weight = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ics = [3, 32, 64, 64, 128]
ocs = [32, 64, 64, 128, 128]
max_p = [False, True, False, True, False]
net_ = PCNet(inp_channels=ics, out_channels=ocs, max_pool=max_p, cls=30)
total_params = sum(p.numel() for p in net_.parameters())


net_ = net_.to(device)
net_ = nn.DataParallel(net_)
net_.load_state_dict(checkpoint_weight['net'])
