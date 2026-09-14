"""RGB teacher views: native CIFAR augmentation, then teacher-only resizing."""
import torch.nn.functional as F
from torchvision import transforms


def resize_rgb_teacher_input(inputs, size):
    """Inputs are already normalized with the student's CIFAR statistics."""
    if inputs.shape[-2:] == (size, size):
        return inputs
    return F.interpolate(inputs, size=(size, size), mode='bilinear', align_corners=False)


class RGBTeacherResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return resize_rgb_teacher_input(image.unsqueeze(0), self.size).squeeze(0)


def rgb_teacher_metadata(dataset, size=224):
    from trainer import _CIFAR_STATS
    mean, std = _CIFAR_STATS[dataset]
    return dict(kind='rgb_cifar_normalize_then_resize', dataset=dataset,
                size=size, mean=mean, std=std, interpolation='bilinear',
                align_corners=False, antialias=False)


def rgb_teacher_transforms(dataset, train_size=224, test_size=224):
    meta = rgb_teacher_metadata(dataset)
    def final(size):
        return [transforms.ToTensor(), transforms.Normalize(meta['mean'], meta['std']),
                RGBTeacherResize(size)]
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(), *final(train_size)])
    test = transforms.Compose(final(test_size))
    return train, test
