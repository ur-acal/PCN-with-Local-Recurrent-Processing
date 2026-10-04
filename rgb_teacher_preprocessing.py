"""RGB teacher views: native CIFAR augmentation, then teacher-only resizing."""
import torch.nn.functional as F
from torchvision import transforms


def normalization_bool(value):
    import argparse
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def resize_rgb_teacher_input(inputs, size):
    """Resize inputs already expressed in the teacher's input coordinates."""
    if inputs.shape[-2:] == (size, size):
        return inputs
    return F.interpolate(inputs, size=(size, size), mode='bilinear', align_corners=False)


class RGBTeacherResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return resize_rgb_teacher_input(image.unsqueeze(0), self.size).squeeze(0)


def rgb_teacher_metadata(dataset, size=224, normalize=True):
    from trainer import _CIFAR_STATS
    mean, std = _CIFAR_STATS[dataset]
    return dict(kind=('rgb_cifar_normalize_then_resize' if normalize else
                      'rgb_cifar_resize'), dataset=dataset, normalize=bool(normalize),
                size=size, mean=mean, std=std, interpolation='bilinear',
                align_corners=False, antialias=False)


def rgb_teacher_transforms(dataset, train_size=224, test_size=224, normalize=True):
    meta = rgb_teacher_metadata(dataset, normalize=normalize)
    def final(size):
        return [transforms.ToTensor(),
                *([transforms.Normalize(meta['mean'], meta['std'])] if normalize else []),
                RGBTeacherResize(size)]
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(), *final(train_size)])
    test = transforms.Compose(final(test_size))
    return train, test


def prepare_rgb_teacher_input(inputs, teacher, dataset, student_normalized=True):
    """Convert the shared RGB view to the checkpoint's teacher coordinates.

    No additional augmentation is sampled. Legacy RGB checkpoints without
    metadata retain CIFAR normalization; non-RGB callers do not use this helper.
    """
    from data_utils import _CIFAR_STATS
    meta = getattr(teacher, 'rgb_teacher_preprocessing', None)
    normalize = True if meta is None else meta.get('normalize', True)
    student_mean, student_std = _CIFAR_STATS[dataset]
    teacher_mean = student_mean if meta is None else meta['mean']
    teacher_std = student_std if meta is None else meta['std']
    same_stats = (tuple(student_mean) == tuple(teacher_mean) and
                  tuple(student_std) == tuple(teacher_std))
    # Keep the legacy path bit-identical (no normalize/denormalize round trip).
    if student_normalized != normalize or (normalize and not same_stats):
        def channel(values):
            return inputs.new_tensor(values).view(1, -1, 1, 1)
        if student_normalized:
            inputs = inputs * channel(student_std) + channel(student_mean)
        if normalize:
            inputs = (inputs - channel(teacher_mean)) / channel(teacher_std)
    size = getattr(teacher, 'rgb_teacher_input_size', None)
    return resize_rgb_teacher_input(inputs, size) if size is not None else inputs


def resolve_teacher_training_normalization(args):
    """Teacher test-only runs inherit saved preprocessing; training defaults on."""
    requested = getattr(args, 'normalize_input', None)
    if getattr(args, 'test_only', False) and args.img_type.lower() == 'rgb':
        import torch
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        meta = checkpoint.get('teacher_preprocessing', {})
        if meta and meta.get('dataset') != args.dataset:
            raise ValueError('Teacher checkpoint preprocessing dataset mismatch')
        saved = meta.get('normalize', True)
        if requested is not None and requested != saved:
            raise ValueError('Teacher evaluation normalization must match its checkpoint')
        return saved
    return True if requested is None else requested
