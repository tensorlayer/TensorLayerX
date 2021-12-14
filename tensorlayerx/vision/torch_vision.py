#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numbers
import numpy as np
from PIL import Image
from torch import Tensor
import torch
from . import functional_cv2 as F_cv2
from . import functional_pil as F_pil
from torch.nn.functional import pad as torch_pad
from torchvision.transforms.functional_tensor import _pad_symmetric, _cast_squeeze_in, _cast_squeeze_out
from torchvision.transforms.functional_tensor import _blend, rgb_to_grayscale, _rgb2hsv, _hsv2rgb
from torch.nn.functional import interpolate, grid_sample
import random
import math
__all__ = [
    'central_crop',
    'to_tensor',
    'crop',
    'pad',
    'resize',
    'transpose',
    'hwc_to_chw',
    'chw_to_hwc',
    'rgb_to_hsv',
    'hsv_to_rgb',
    'rgb_to_gray',
    'adjust_brightness',
    'adjust_contrast',
    'adjust_hue',
    'adjust_saturation',
    'normalize',
    'hflip',
    'vflip',
    'padtoboundingbox',
    'standardize',
    'random_brightness',
    'random_contrast',
    'random_saturation',
    'random_hue',
    'random_crop',
    'random_resized_crop',
    'random_vflip',
    'random_hflip',
    'random_rotation',
    'random_shear',
    'random_shift',
    'random_zoom',
    'random_affine',
]


def _is_pil_image(image):
    return isinstance(image, Image.Image)


def _is_numpy_image(image):
    return isinstance(image, np.ndarray) and (image.ndim in {2, 3})


def _is_tensor_image(image):
    return isinstance(image, torch.Tensor) and image.ndim >= 2


def _get_image_size(image):
    if _is_tensor_image(image):
        return image.shape[-2], image.shape[-1]
    elif _is_numpy_image(image):
        return image.shape[-3], image.shape[-2]
    elif _is_pil_image(image):
        return image.size[::-1]


def _get_image_num_channels(image):

    if _is_tensor_image(image):
        if image.ndim == 2:
            return 1
        elif image.ndim > 2:
            return image.shape[-3]
    elif _is_numpy_image(image):
        return image.shape[-1]
    elif _is_pil_image(image):
        return 1 if image.mode == 'L' else 3


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]
    cx, cy = center
    tx, ty = translate
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def _compute_output_size(matrix, w, h):

    pts = torch.tensor(
        [
            [-0.5 * w, -0.5 * h, 1.0],
            [-0.5 * w, 0.5 * h, 1.0],
            [0.5 * w, 0.5 * h, 1.0],
            [0.5 * w, -0.5 * h, 1.0],
        ]
    )
    theta = torch.tensor(matrix, dtype=torch.float).reshape(1, 2, 3)
    new_pts = pts.view(1, 4, 3).bmm(theta.transpose(1, 2)).view(4, 2)
    min_vals, _ = new_pts.min(dim=0)
    max_vals, _ = new_pts.max(dim=0)

    tol = 1e-4
    cmax = torch.ceil((max_vals / tol).trunc_() * tol)
    cmin = torch.floor((min_vals / tol).trunc_() * tol)
    size = cmax - cmin
    return int(size[0]), int(size[1])


def _gen_affine_grid(theta, w, h, ow, oh):
    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)
    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)


def _apply_grid_transform(img, grid, mode, fill):

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [
        grid.dtype,
    ])

    if img.shape[0] > 1:
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    if fill is not None:
        dummy = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)
        img = torch.cat((img, dummy), dim=1)

    img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = torch.tensor(fill, dtype=img.dtype, device=img.device).view(1, len_fill, 1, 1).expand_as(img)
        if mode == 'nearest':
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:
            img = img * mask + (1.0 - mask) * fill_img

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def random_factor(factor, name, center=1, bound=(0, float('inf')), non_negative=True):
    if isinstance(factor, numbers.Number):
        if factor < 0:
            raise ValueError('The input value of {} cannot be negative.'.format(name))
        factor = [center - factor, center + factor]
        if non_negative:
            factor[0] = max(0, factor[0])
    elif isinstance(factor, (tuple, list)) and len(factor) == 2:
        if not bound[0] <= factor[0] <= factor[1] <= bound[1]:
            raise ValueError(
                "Please check your value range of {} is valid and "
                "within the bound {}.".format(name, bound)
            )
    else:
        raise TypeError("Input of {} should be either a single value, or a list/tuple of " "length 2.".format(name))
    factor = np.random.uniform(factor[0], factor[1])
    return factor


def central_crop(image, size=None, central_fraction=None):
    if size is None and central_fraction is None:
        raise ValueError('central_fraction and size can not be both None')
    image_height, image_width = _get_image_size(image)
    if size is not None:
        if not isinstance(size, (int, list, tuple)) or (isinstance(size, (list, tuple)) and len(size) != 2):
            raise ValueError(
                "Size should be a single integer or a list/tuple (h, w) of length 2.But"
                "got {}.".format(type(size))
            )
        if isinstance(size, int):
            target_height = size
            target_width = size
        else:
            target_height = size[0]
            target_width = size[1]
    elif central_fraction is not None:
        if central_fraction <= 0.0 or central_fraction > 1.0:
            raise ValueError('central_fraction must be within (0, 1]')

        target_height = int(central_fraction * image_height)
        target_width = int(central_fraction * image_width)

    crop_top = int(round((image_height - target_height) / 2.))
    crop_left = int(round((image_width - target_width) / 2.))

    return crop(image, crop_top, crop_left, target_height, target_width)


def to_tensor(image, data_format):
    if not (_is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be PIL Image or ndarray. Got {}'.format(type(image)))

    default_float_dtype = torch.get_default_dtype()

    if _is_numpy_image(image):
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image).contiguous()
        if isinstance(image, torch.ByteTensor):
            image = image.to(dtype=default_float_dtype).div(255)
        if data_format == 'CHW':
            image = image.permute((2, 0, 1)).contiguous()
        return image

    if _is_pil_image(image):
        mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
        img = torch.from_numpy(np.array(image, mode_to_nptype.get(image.mode, np.uint8), copy=True))

    if image.mode == '1':
        img = 255 * img
    img = img.view(image.size[1], image.size[0], len(image.getbands()))

    if isinstance(img, torch.ByteTensor):
        img = img.to(dtype=default_float_dtype).div(255)

    if data_format == 'CHW':
        img = img.permute((2, 0, 1)).contiguous()

    return img


def crop(image, top, left, height, width):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))

    if _is_tensor_image(image):
        right = left + width
        bottom = top + height
        return image[:, top:bottom, left:right]
    elif _is_pil_image(image):
        return F_pil.crop(image, top, left, height, width)
    elif _is_numpy_image(image):
        return F_cv2.crop(image, top, left, height, width)


def pad(image, padding, padding_value, mode):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))

    if _is_tensor_image(image):
        if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
            raise ValueError(
                "Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                "{} element tuple".format(len(padding))
            )
        if not isinstance(padding_value, (int, float)):
            raise TypeError("Padding_value should be int or float, but got {}.".format(type(padding_value)))
        if isinstance(padding, int):
            if torch.jit.is_scripting():
                raise ValueError("padding can't be an int while torchscripting, set it as a list [value, ]")
            top = bottom = left = right = padding
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 1:
                top = bottom = left = right = padding[0]
            elif len(padding) == 2:
                left = right = padding[0]
                top = bottom = padding[1]
            elif len(padding) == 4:
                left = padding[0]
                top = padding[1]
                right = padding[2]
                bottom = padding[3]

        p = [left, right, top, bottom]
        if mode == "edge":
            mode = "replicate"
        elif mode == "symmetric":
            return _pad_symmetric(image, p)

        need_squeeze = False
        if image.ndim < 4:
            image = image.unsqueeze(dim=0)
            need_squeeze = True
        out_dtype = image.dtype
        need_cast = False
        if (mode != "constant") and image.dtype not in (torch.float32, torch.float64):
            # Here we temporary cast input tensor to float
            # until pytorch issue is resolved :
            # https://github.com/pytorch/pytorch/issues/40763
            need_cast = True
            image = image.to(torch.float32)
        image = torch_pad(image, p, mode=mode, value=float(padding_value))
        if need_squeeze:
            image = image.squeeze(dim=0)
        if need_cast:
            image = image.to(out_dtype)
        return image

    elif _is_pil_image(image):
        return F_pil.pad(image, padding, padding_value, mode)
    else:
        return F_cv2.pad(image, padding, padding_value, mode)


def resize(image, size, method):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))

    if _is_tensor_image(image):
        if not (isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) == 2)):
            raise TypeError('Size should be a single number or a list/tuple (h, w) of length 2.' 'Got {}.'.format(size))

        h, w = _get_image_size(image)
        if isinstance(size, int):
            if (w <= h and w == size) or (h <= w and h == size):
                size = [h, w]
            if w < h:
                new_w = size
                new_h = int(size * h / w)
                size = [new_h, new_w]
            else:
                new_h = size
                new_w = int(size * w / h)
                size = [new_h, new_w]
        image, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(image, [torch.float32, torch.float64])
        align_corners = False if method in ["bilinear", "bicubic"] else None
        image = interpolate(image, size=size, mode=method, align_corners=align_corners)
        if method == 'bicubic' and out_dtype == torch.uint8:
            image = image.clamp(min=0, max=255)
        image = _cast_squeeze_out(image, need_cast=need_cast, need_squeeze=need_squeeze, out_dtype=out_dtype)
        return image
    elif _is_pil_image(image):
        return F_pil.resize(image, size, method)
    elif _is_numpy_image(image):
        return F_cv2.resize(image, size, method)


def transpose(image, order):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        image = image.permute(order).contiguous()
        return image
    elif _is_pil_image(image):
        return F_pil.transpose(image, order)
    elif _is_numpy_image(image):
        return F_cv2.transpose(image, order)


def hwc_to_chw(image):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        if image.ndim == 3:
            return image.permute((2, 0, 1)).contiguous()
        if image.ndim == 4:
            return image.permute((0, 3, 1, 2)).contiguous()
    elif _is_pil_image(image):
        image = np.asarray(image)
        image_shape = image.shape
        if len(image_shape) == 2:
            image = image[..., np.newaxis]
        image = image.transpose((2, 0, 1))
        image = Image.fromarray(image)
        return image
    elif _is_numpy_image(image):
        image_shape = image.shape
        if len(image_shape) == 2:
            image = image[..., np.newaxis]
        return image.transpose((2, 0, 1))


def chw_to_hwc(image):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        if image.ndim == 3:
            return image.permute((1, 2, 0)).contiguous()
        if image.ndim == 4:
            return image.permute((0, 2, 3, 1)).contiguous()
    elif _is_pil_image(image):
        image = np.asarray(image)
        image_shape = image.shape
        if len(image_shape) == 2:
            image = image[..., np.newaxis]
        image = image.transpose((1, 2, 0))
        image = Image.fromarray(image)
        return image
    elif _is_numpy_image(image):
        image_shape = image.shape
        if len(image_shape) == 2:
            image = image[..., np.newaxis]
        return image.transpose((1, 2, 0))


def rgb_to_hsv(image):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        orig_dtype = image.dtype
        if orig_dtype == torch.uint8:
            image = image.to(dtype=torch.float32) / 255.0
        r, g, b = image.unbind(dim=-3)
        maxc = torch.max(image, dim=-3).values
        minc = torch.min(image, dim=-3).values
        eqc = maxc == minc
        cr = maxc - minc
        ones = torch.ones_like(maxc)
        s = cr / torch.where(eqc, ones, maxc)
        cr_divisor = torch.where(eqc, ones, cr)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor
        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = (hr + hg + hb)
        h = torch.fmod((h / 6.0 + 1.0), 1.0)
        image = torch.stack((h, s, maxc), dim=-3)
        if orig_dtype == torch.uint8:
            image = (image * 255.0).to(dtype=orig_dtype)
        return image
    elif _is_pil_image(image):
        return F_pil.rgb_to_hsv(image)
    elif _is_numpy_image(image):
        return F_cv2.rgb_to_hsv(image)


def hsv_to_rgb(image):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        orig_dtype = image.dtype
        if orig_dtype == torch.uint8:
            image = image.to(dtype=torch.float32) / 255.0
        h, s, v = image.unbind(dim=-3)
        i = torch.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.to(dtype=torch.int32)
        p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
        q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
        t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)
        a1 = torch.stack((v, q, p, p, t, v), dim=-3)
        a2 = torch.stack((t, v, v, q, p, p), dim=-3)
        a3 = torch.stack((p, p, t, v, v, q), dim=-3)
        a4 = torch.stack((a1, a2, a3), dim=-4)

        image = torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=image.dtype), a4)
        if orig_dtype == torch.uint8:
            image = (image * 255.0).to(dtype=orig_dtype)
        return image
    elif _is_pil_image(image):
        return F_pil.hsv_to_rgb(image)
    elif _is_numpy_image(image):
        return F_cv2.hsv_to_rgb(image)


def rgb_to_gray(image, num_output_channels):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        if image.ndim < 3:
            raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(image.ndim))
        if num_output_channels not in (1, 3):
            raise ValueError('num_output_channels should be either 1 or 3')
        r, g, b = image.unbind(dim=-3)
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(image.dtype)
        l_img = l_img.unsqueeze(dim=-3)
        if num_output_channels == 3:
            return l_img.expand(image.shape)

        return l_img
    elif _is_pil_image(image):
        return F_pil.rgb_to_gray(image, num_output_channels)
    elif _is_numpy_image(image):
        return F_cv2.rgb_to_gray(image, num_output_channels)


def adjust_brightness(image, brightness_factor):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        if brightness_factor < 0:
            raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))
        if _get_image_num_channels(image) not in [1, 3]:
            raise TypeError(
                "Input image tensor permitted channel values are {}, but found {}".format(
                    [1, 3], _get_image_num_channels(image)
                )
            )
        return _blend(image, torch.zeros_like(image), brightness_factor)
    elif _is_pil_image(image):
        return F_pil.adjust_brightness(image, brightness_factor)
    elif _is_numpy_image(image):
        return F_cv2.adjust_brightness(image, brightness_factor)


def adjust_contrast(image, contrast_factor):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        if contrast_factor < 0:
            raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))
        if _get_image_num_channels(image) not in [3]:
            raise TypeError(
                "Input image tensor permitted channel values are {}, but found {}".format(
                    [3], _get_image_num_channels(image)
                )
            )
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        mean = torch.mean(rgb_to_grayscale(image).to(dtype), dim=(-3, -2, -1), keepdim=True)

        return _blend(image, mean, contrast_factor)
    elif _is_pil_image(image):
        return F_pil.adjust_contrast(image, contrast_factor)
    elif _is_numpy_image(image):
        return F_cv2.adjust_contrast(image, contrast_factor)


def adjust_hue(image, hue_factor):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))
        if _get_image_num_channels(image) not in [1, 3]:
            raise TypeError(
                "Input image tensor permitted channel values are {}, but found {}".format(
                    [1, 3], _get_image_num_channels(image)
                )
            )
        if _get_image_num_channels(image) == 1:
            return image
        orig_dtype = image.dtype
        if image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32) / 255.0

        image = _rgb2hsv(image)
        h, s, v = image.unbind(dim=-3)
        h = (h + hue_factor) % 1.0
        image = torch.stack((h, s, v), dim=-3)
        img_hue_adj = _hsv2rgb(image)

        if orig_dtype == torch.uint8:
            img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)
        return img_hue_adj
    elif _is_pil_image(image):
        return F_pil.adjust_hue(image, hue_factor)
    elif _is_numpy_image(image):
        return F_cv2.adjust_hue(image, hue_factor)


def adjust_saturation(image, saturation_factor):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        if saturation_factor < 0:
            raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))
        if _get_image_num_channels(image) not in [3]:
            raise TypeError(
                "Input image tensor permitted channel values are {}, but found {}".format(
                    [3], _get_image_num_channels(image)
                )
            )
        return _blend(image, rgb_to_grayscale(image), saturation_factor)
    elif _is_pil_image(image):
        return F_pil.adjust_saturation(image, saturation_factor)
    elif _is_numpy_image(image):
        return F_cv2.adjust_saturation(image, saturation_factor)


def hflip(image):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        return image.flip(-1)
    elif _is_pil_image(image):
        return F_pil.hflip(image)
    elif _is_numpy_image(image):
        return F_cv2.hflip(image)


def vflip(image):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        return image.flip(-2)
    elif _is_pil_image(image):
        return F_pil.vflip(image)
    elif _is_numpy_image(image):
        return F_cv2.vflip(image)


def padtoboundingbox(image, offset_height, offset_width, target_height, target_width, padding_value):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))

    if _is_tensor_image(image):
        if offset_height < 0:
            raise ValueError("offset_height must be >= 0.")
        if offset_width < 0:
            raise ValueError("offset_width must be >= 0.")

        h, w = _get_image_size(image)
        left = offset_width
        top = offset_height
        bottom = target_height - h - offset_height
        right = target_width - w - offset_width
        if bottom < 0:
            raise ValueError("image height must be <= target - offset.")
        if right < 0:
            raise ValueError("image width must be <= target - offset.")
        p = [left, right, top, bottom]

        need_squeeze = False
        if image.ndim < 4:
            image = image.unsqueeze(dim=0)
            need_squeeze = True
        image = torch_pad(image, p, mode="constant", value=float(padding_value))
        if need_squeeze:
            image = image.squeeze(dim=0)
        return image
    elif _is_pil_image(image):
        return F_pil.padtoboundingbox(image, offset_height, offset_width, target_height, target_width, padding_value)
    elif _is_numpy_image(image):
        return F_cv2.padtoboundingbox(image, offset_height, offset_width, target_height, target_width, padding_value)


def normalize(image, mean, std, data_format):
    if not _is_tensor_image(image):
        raise TypeError('image should be a Tensor. Got {}'.format(type(image)))
    if image.ndim < 3:
        raise ValueError("Expected dimensions of image should be at least 3, but got {}".format(image.size()))
    image = image.to(torch.float32)
    dtype = image.dtype
    if isinstance(mean, numbers.Number):
        mean = [mean, mean, mean]
    if isinstance(std, numbers.Number):
        std = [std, std, std]
    mean = torch.as_tensor(mean, dtype=dtype, device=image.device)
    std = torch.as_tensor(std, dtype=dtype, device=image.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if data_format == 'CHW':
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif data_format == 'HWC':
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)
    image.sub_(mean).div_(std)
    return image


def standardize(image):
    if not _is_tensor_image(image):
        raise TypeError('image should be a Tensor. Got {}'.format(type(image)))
    if image.ndim < 3:
        raise ValueError("Expected dimensions of image should be at least 3, but got {}".format(image.size()))
    image = image.to(torch.float32)
    dtype = image.dtype
    num_pixels = torch.tensor(torch.numel(image), dtype=dtype, device=image.device)
    mean = torch.mean(image, dtype=dtype)
    stddev = torch.std(image, unbiased=False)
    min_stddev = 1.0 / torch.sqrt(num_pixels)
    adjusted_stddev = torch.maximum(stddev, min_stddev)
    return image.sub_(mean).div_(adjusted_stddev)


def random_brightness(image, brightness_factor):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    brightness_factor = random_factor(brightness_factor, name='brightness')
    if _is_tensor_image(image):
        if _get_image_num_channels(image) not in [1, 3]:
            raise TypeError(
                "Input image tensor permitted channel values are {}, but found {}".format(
                    [1, 3], _get_image_num_channels(image)
                )
            )
        return _blend(image, torch.zeros_like(image), brightness_factor)
    elif _is_pil_image(image):
        return F_pil.adjust_brightness(image, brightness_factor)
    elif _is_numpy_image(image):
        return F_cv2.adjust_brightness(image, brightness_factor)


def random_contrast(image, contrast_factor):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    contrast_factor = random_factor(contrast_factor, name='contrast')
    if _is_tensor_image(image):
        if _get_image_num_channels(image) not in [3]:
            raise TypeError(
                "Input image tensor permitted channel values are {}, but found {}".format(
                    [3], _get_image_num_channels(image)
                )
            )
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        mean = torch.mean(rgb_to_grayscale(image).to(dtype), dim=(-3, -2, -1), keepdim=True)
        return _blend(image, mean, contrast_factor)
    elif _is_pil_image(image):
        return F_pil.adjust_contrast(image, contrast_factor)
    elif _is_numpy_image(image):
        return F_cv2.adjust_contrast(image, contrast_factor)


def random_saturation(image, saturation_factor):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    saturation_factor = random_factor(saturation_factor, name='saturation')
    if _is_tensor_image(image):
        if _get_image_num_channels(image) not in [3]:
            raise TypeError(
                "Input image tensor permitted channel values are {}, but found {}".format(
                    [3], _get_image_num_channels(image)
                )
            )
        return _blend(image, rgb_to_grayscale(image), saturation_factor)
    elif _is_pil_image(image):
        return F_pil.adjust_saturation(image, saturation_factor)
    elif _is_numpy_image(image):
        return F_cv2.adjust_saturation(image, saturation_factor)


def random_hue(image, hue_factor):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    hue_factor = random_factor(hue_factor, name='hue', center=0, bound=(-0.5, 0.5), non_negative=False)
    if _is_tensor_image(image):
        if _get_image_num_channels(image) not in [1, 3]:
            raise TypeError(
                "Input image tensor permitted channel values are {}, but found {}".format(
                    [1, 3], _get_image_num_channels(image)
                )
            )
        if _get_image_num_channels(image) == 1:
            return image
        orig_dtype = image.dtype
        if image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32) / 255.0
        image = _rgb2hsv(image)
        h, s, v = image.unbind(dim=-3)
        h = (h + hue_factor) % 1.0
        image = torch.stack((h, s, v), dim=-3)
        img_hue_adj = _hsv2rgb(image)
        if orig_dtype == torch.uint8:
            img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)
        return img_hue_adj
    elif _is_pil_image(image):
        return F_pil.adjust_hue(image, hue_factor)
    elif _is_numpy_image(image):
        return F_cv2.adjust_hue(image, hue_factor)


def random_crop(image, size, padding, pad_if_needed, fill, padding_mode):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        size = tuple(size)
    else:
        raise ValueError('Size should be a int or a list/tuple with length of 2. But got {}'.format(size))

    if padding is not None:
        image = pad(image, padding, fill, padding_mode)

    h, w = _get_image_size(image)

    if pad_if_needed and w < size[1]:
        image = pad(image, (size[1] - w, 0), fill, padding_mode)
    if pad_if_needed and h < size[0]:
        image = pad(image, (0, size[0] - h), fill, padding_mode)

    h, w = _get_image_size(image)
    target_height, target_width = size

    if h < target_height or w < target_width:
        raise ValueError(
            'Crop size {} should be smaller than input image size {}. '.format((target_height, target_width), (h, w))
        )

    offset_height = random.randint(0, h - target_height)
    offset_width = random.randint(0, w - target_width)

    return crop(image, offset_height, offset_width, target_height, target_width)


def random_resized_crop(image, size, scale, ratio, interpolation):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))

    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (list, tuple)) and len(size) == 2:
        size = tuple(size)
    else:
        raise TypeError('Size should be a int or a list/tuple with length of 2.' 'But got {}.'.format(size))
    if not (isinstance(scale, (list, tuple)) and len(scale) == 2):
        raise TypeError('Scale should be a list/tuple with length of 2.' 'But got {}.'.format(scale))
    if not (isinstance(ratio, (list, tuple)) and len(ratio) == 2):
        raise TypeError('Scale should be a list/tuple with length of 2.' 'But got {}.'.format(ratio))

    if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
        raise ValueError("Scale and ratio should be of kind (min, max)")

    def get_params(img, scale, ratio):
        height, width = _get_image_size(img)
        area = height * width
        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1, )).item()
                j = torch.randint(0, width - w + 1, size=(1, )).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    i, j, h, w = get_params(image, scale, ratio)
    image = crop(image, i, j, h, w)
    image = resize(image, size, interpolation)
    return image


def random_vflip(image, prob):
    if random.random() < prob:
        return vflip(image)
    return image


def random_hflip(image, prob):
    if random.random() < prob:
        return hflip(image)
    return image


def random_rotation(image, degrees, interpolation, expand, center, fill):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError('If degrees is a single number, it must be positive.' 'But got {}'.format(degrees))
        degrees = (-degrees, degrees)
    elif not (isinstance(degrees, (list, tuple)) and len(degrees) == 2):
        raise ValueError('If degrees is a list/tuple, it must be length of 2.' 'But got {}'.format(degrees))
    else:
        if degrees[0] > degrees[1]:
            raise ValueError('if degrees is a list/tuple, it should be (min, max).')

    angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
    h, w = _get_image_size(image)
    if _is_tensor_image(image):
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * _get_image_num_channels(image)
        else:
            fill = [float(f) for f in fill]
        center_f = [0.0, 0.0]
        if center is not None:
            center_f = [1.0 * (center[0] - 0.5 * w), 1.0 * (center[1] - 0.5 * h)]
        matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        ow, oh = _compute_output_size(matrix, w, h) if expand else (w, h)
        theta = torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)
        grid = _gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)
        return _apply_grid_transform(image, grid, interpolation, fill=fill)
    elif _is_pil_image(image):
        return F_pil.rotate(image, angle, interpolation, expand, center, fill)
    elif _is_numpy_image(image):
        return F_cv2.rotate(image, angle, interpolation, expand, center, fill)


def random_shear(image, degrees, interpolation, fill):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it must be positive.")
        degrees = (-degrees, degrees, 0, 0)
    elif isinstance(degrees, (list, tuple)) and (len(degrees) == 2 or len(degrees) == 4):
        if len(degrees) == 2:
            degrees = (degrees[0], degrees[1], 0, 0)
    else:
        raise ValueError(
            'degrees should be a single number or a list/tuple with length in (2 ,4).'
            'But got {}'.format(degrees)
        )
    if _is_tensor_image(image):
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * _get_image_num_channels(image)
        else:
            fill = [float(f) for f in fill]
        h, w = _get_image_size(image)
        angle = 0.0
        translation = (0.0, 0.0)
        scale = 1.0
        shear_x = float(torch.empty(1).uniform_(degrees[0], degrees[1]).item())
        shear_y = float(torch.empty(1).uniform_(degrees[2], degrees[3]).item())
        shear = (shear_x, shear_y)
        matrix = _get_inverse_affine_matrix([0.0, 0.0], angle, translation, scale, shear)
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        theta = torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)
        grid = _gen_affine_grid(theta, w=w, h=h, ow=w, oh=h)
        return _apply_grid_transform(image, grid, interpolation, fill=fill)
    elif _is_pil_image(image):
        return F_pil.random_shear(image, degrees, interpolation, fill)
    elif _is_numpy_image(image):
        return F_cv2.random_shear(image, degrees, interpolation, fill)


def random_shift(image, shift, interpolation, fill):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if not (isinstance(shift, (tuple, list)) and len(shift) == 2):
        raise ValueError('Shift should be a list/tuple with length of 2.' 'But got {}'.format(shift))
    for s in shift:
        if not (0.0 <= s <= 1.0):
            raise ValueError("shift values should be between 0 and 1.")
    if _is_tensor_image(image):
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * _get_image_num_channels(image)
        else:
            fill = [float(f) for f in fill]
        h, w = _get_image_size(image)
        angle = 0.0
        max_dx = float(shift[0] * w)
        max_dy = float(shift[1] * h)
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        translation = (float(tx), float(ty))
        scale = 1.0
        shear = (0.0, 0.0)
        matrix = _get_inverse_affine_matrix([0.0, 0.0], angle, translation, scale, shear)
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        theta = torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)
        grid = _gen_affine_grid(theta, w=w, h=h, ow=w, oh=h)
        return _apply_grid_transform(image, grid, interpolation, fill=fill)
    elif _is_pil_image(image):
        return F_pil.random_shift(image, shift, interpolation, fill)
    elif _is_numpy_image(image):
        return F_cv2.random_shift(image, shift, interpolation, fill)


def random_zoom(image, zoom, interpolation, fill):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if not (isinstance(zoom, (tuple, list)) and len(zoom) == 2):
        raise ValueError('Zoom should be a list/tuple with length of 2.' 'But got {}'.format(zoom))
    if not (0 <= zoom[0] <= zoom[1]):
        raise ValueError('Zoom values should be positive, and zoom[1] should be greater than zoom[0].')
    if _is_tensor_image(image):
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * _get_image_num_channels(image)
        else:
            fill = [float(f) for f in fill]
        h, w = _get_image_size(image)
        angle = 0.0
        translation = (0, 0)
        scale = float(torch.empty(1).uniform_(zoom[0], zoom[1]).item())
        shear = (0.0, 0.0)
        matrix = _get_inverse_affine_matrix([0.0, 0.0], angle, translation, scale, shear)
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        theta = torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)
        grid = _gen_affine_grid(theta, w=w, h=h, ow=w, oh=h)
        return _apply_grid_transform(image, grid, interpolation, fill=fill)
    elif _is_pil_image(image):
        return F_pil.random_zoom(image, zoom, interpolation, fill)
    elif _is_numpy_image(image):
        return F_cv2.random_zoom(image, zoom, interpolation, fill)


def random_affine(image, degrees, shift, zoom, shear, interpolation, fill):
    if not (_is_tensor_image(image) or _is_pil_image(image) or _is_numpy_image(image)):
        raise TypeError('image should be Tensor Image or PIL Image or ndarray. Got {}'.format(type(image)))
    if _is_tensor_image(image):
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * _get_image_num_channels(image)
        else:
            fill = [float(f) for f in fill]
        h, w = _get_image_size(image)
        translation = (0.0, 0.0)
        if shift is not None:
            angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
            max_dx = float(shift[0] * w)
            max_dy = float(shift[1] * h)
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translation = (float(tx), float(ty))
        scale = 1.0
        if zoom is not None:
            scale = float(torch.empty(1).uniform_(zoom[0], zoom[1]).item())
        shear_x = shear_y = 0.0
        if shear is not None:
            shear_x = float(torch.empty(1).uniform_(shear[0], shear[1]).item())
            shear_y = float(torch.empty(1).uniform_(shear[2], shear[3]).item())
        shear = (shear_x, shear_y)

        matrix = _get_inverse_affine_matrix([0.0, 0.0], angle, translation, scale, shear)
        dtype = image.dtype if torch.is_floating_point(image) else torch.float32
        theta = torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)
        grid = _gen_affine_grid(theta, w=w, h=h, ow=w, oh=h)
        return _apply_grid_transform(image, grid, interpolation, fill=fill)
    elif _is_pil_image(image):
        return F_pil.random_affine(image, degrees, shift, zoom, shear, interpolation, fill)
    elif _is_numpy_image(image):
        return F_cv2.random_affine(image, degrees, shift, zoom, shear, interpolation, fill)
