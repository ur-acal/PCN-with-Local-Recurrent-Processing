import torch.nn as nn
import torch
import torchvision

from torchvision import transforms
from pc_model import PCNet
from pc_conv import PCConv, PCConvNoisy

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


def load_model(model_path_):
    ckpt = torch.load(model_path_, map_location=device)
    model_ = PCNet(pc_conv_layer=PCConv,
                   **ckpt["init_args"]["model_args"],
                   **ckpt["init_args"]["kwargs"])
    model_.load_state_dict(ckpt["net"])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), ])
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    # Create a DataLoader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    model_.eval()
    model_.to(device)
    test_res_ = model_(next(iter(test_loader))[0].to(device))
    print(test_res_.shape)

model_path_list = [
    "/home/rongzeng/_workspce_old/repos/pcn/PCN-with-Local-Recurrent-Processing/saved_ckpt/PPCN_5CLS_1.0LRPC_0.001WD_withTied_withBPtied_withBP_noRelu_9Layers_1REP/PPCN_5CLS_1.0LRPC_0.001WD_withTied_withBPtied_withBP_noRelu_9Layers_1REP_best_ckpt.pth",
    "/home/rongzeng/_workspce_old/repos/pcn/PCN-with-Local-Recurrent-Processing/saved_ckpt/PPCN_5CLS_1.0LRPC_0.001WD_noTied_noBPtied_noBP_noRelu_7Layers_1REP/PPCN_5CLS_1.0LRPC_0.001WD_noTied_noBPtied_noBP_noRelu_7Layers_1REP_best_ckpt.pth"
]

for mp_ in model_path_list:
    load_model(mp_)
