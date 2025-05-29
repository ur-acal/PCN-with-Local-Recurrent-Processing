from pc_model import PCNetWithMiddleConv
import torch.nn as nn

import logging
log = logging.getLogger(__name__)


def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        # fuse both the weight and the bias
        return nn.utils.fuse_conv_bn_eval(block2, block1)
    else:
        return False


def fuse_bn_recursively(model) -> nn.Module:
    if not hasattr(model, "mid_convs"):
        log.warning("----- Not able to fuse -----")
        return model
    n_bn = len(model.BNs)
    for i in range(n_bn):
        fused = fuse_single_conv_bn_pair(model.BNs[i], model.mid_convs[i])
        model.mid_convs[i] = fused
        model.BNs[i] = nn.Identity()

    fused = fuse_single_conv_bn_pair(model.BNend, model.mid_convs[-1])
    model.mid_convs[-1] = fused
    model.BNend = nn.Identity()
    log.warning("----- BN fused to conv -----")
    return model