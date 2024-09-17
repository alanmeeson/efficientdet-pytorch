""" COCO transforms (quick and dirty)

Hacked together by Ross Wightman
"""
import random
import math
from copy import deepcopy

from PIL import Image
import numpy as np
import torch

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


class ImageToNumpy:

    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img, annotations


class ImageToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype), annotations


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def clip_boxes_(boxes, img_size):
    height, width = img_size
    clip_upper = np.array([height, width] * 2, dtype=boxes.dtype)
    np.clip(boxes, 0, clip_upper, out=boxes)


def clip_boxes(boxes, img_size):
    clipped_boxes = boxes.copy()
    clip_boxes_(clipped_boxes, img_size)
    return clipped_boxes


def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size


class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear', fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img, anno: dict):
        width, height = img.size

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        interp_method = _pil_interp(self.interpolation)
        img = img.resize((scaled_w, scaled_h), interp_method)
        new_img.paste(img)  # pastes at 0,0 (upper-left corner)

        if 'bbox' in anno:
            bbox = anno['bbox']
            bbox[:, :4] *= img_scale
            bbox_bound = (min(scaled_h, self.target_size[0]), min(scaled_w, self.target_size[1]))
            clip_boxes_(bbox, bbox_bound)  # crop to bounds of target image or letter-box, whichever is smaller
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        anno['img_scale'] = 1. / img_scale  # back to original

        return new_img, anno


class RandomResizePad:

    def __init__(self, target_size: int, scale: tuple = (0.1, 2.0), interpolation: str = 'random',
                 fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.scale = scale
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.fill_color = fill_color

    def _get_params(self, img):
        # Select a random scale factor.
        scale_factor = random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]

        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.size
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * random.uniform(0, 1))
        return scaled_h, scaled_w, offset_y, offset_x, img_scale

    def __call__(self, img, anno: dict):
        scaled_h, scaled_w, offset_y, offset_x, img_scale = self._get_params(img)

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        img = img.resize((scaled_w, scaled_h), interpolation)
        right, lower = min(scaled_w, offset_x + self.target_size[1]), min(scaled_h, offset_y + self.target_size[0])
        img = img.crop((offset_x, offset_y, right, lower))
        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        new_img.paste(img)  # pastes at 0,0 (upper-left corner)

        if 'bbox' in anno:
            bbox = anno['bbox']  # for convenience, modifies in-place
            bbox[:, :4] *= img_scale
            box_offset = np.stack([offset_y, offset_x] * 2)
            bbox -= box_offset
            bbox_bound = (min(scaled_h, self.target_size[0]), min(scaled_w, self.target_size[1]))
            clip_boxes_(bbox, bbox_bound)  # crop to bounds of target image or letter-box, whichever is smaller
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        anno['img_scale'] = 1. / img_scale  # back to original

        return new_img, anno


class RandomFlip:

    def __init__(self, horizontal=True, vertical=False, prob=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.prob = prob

    def _get_params(self):
        do_horizontal = random.random() < self.prob if self.horizontal else False
        do_vertical = random.random() < self.prob if self.vertical else False
        return do_horizontal, do_vertical

    def __call__(self, img, annotations: dict):
        do_horizontal, do_vertical = self._get_params()
        width, height = img.size

        def _fliph(bbox):
            x_max = width - bbox[:, 1]
            x_min = width - bbox[:, 3]
            bbox[:, 1] = x_min
            bbox[:, 3] = x_max

        def _flipv(bbox):
            y_max = height - bbox[:, 0]
            y_min = height - bbox[:, 2]
            bbox[:, 0] = y_min
            bbox[:, 2] = y_max

        if do_horizontal and do_vertical:
            img = img.transpose(Image.ROTATE_180)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
                _flipv(annotations['bbox'])
        elif do_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
        elif do_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if 'bbox' in annotations:
                _flipv(annotations['bbox'])

        return img, annotations


class RandomRotate:

    def __init__(self, degrees: float = 2, prob: float = 0.5, fill_color: tuple = (0, 0, 0)):
        self.degrees = degrees
        self.fill_color = fill_color
        self.prob = prob

    def _get_params(self):
        if (self.degrees > 0.0001) and (self.prob > 0.0001):
            do_rotate = random.random() < self.prob
            degrees = self.degrees * ((random.random() - 0.5) * 2) if do_rotate else 0.0
            return do_rotate, degrees
        else:
            return False, 0

    def __call__(self, img, annotations: dict):
        do_rotate, degrees = self._get_params()
        orig_image_size = img.size

        def _rotate(boxes, image_size, degrees):
            """
            Rotate bounding boxes by a given angle around the center of the image.

            Args:
                boxes (numpy.ndarray): A numpy array of shape (N, 4), where each row is [y_min, x_min, y_max, x_max].
                image_size (tuple): A tuple of (height, width) representing the size of the image.
                degrees (float): The angle by which to rotate the bounding boxes (clockwise).

            Returns:
                numpy.ndarray: A numpy array of shape (N, 4), where each row is the rotated bounding box [new_y_min, new_x_min, new_y_max, new_x_max].
            """
            # Convert degrees to radians
            radians = np.deg2rad(degrees)

            # Rotation matrix
            cos_angle = np.cos(radians)
            sin_angle = np.sin(radians)
            rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

            # Image center (pivot point)
            image_center = np.array([image_size[1] / 2, image_size[0] / 2])  # (x_center, y_center)

            # Extract corners of all bounding boxes
            top_left = boxes[:, [1, 0]]  # (x_min, y_min)
            top_right = boxes[:, [3, 0]]  # (x_max, y_min)
            bottom_left = boxes[:, [1, 2]]  # (x_min, y_max)
            bottom_right = boxes[:, [3, 2]]  # (x_max, y_max)

            # Stack all corners together: shape (N, 4, 2)
            corners = np.stack([top_left, top_right, bottom_left, bottom_right], axis=1)

            # Translate corners so the center of the image is the origin
            translated_corners = corners - image_center

            # Rotate all corners using the rotation matrix
            rotated_corners = np.dot(translated_corners, rotation_matrix.T)

            # Translate corners back to the original position
            rotated_corners += image_center

            # Get the new bounding boxes by finding min/max coordinates of the rotated corners
            new_x_min = np.min(rotated_corners[:, :, 0], axis=1)
            new_y_min = np.min(rotated_corners[:, :, 1], axis=1)
            new_x_max = np.max(rotated_corners[:, :, 0], axis=1)
            new_y_max = np.max(rotated_corners[:, :, 1], axis=1)

            # Stack the new bounding boxes
            rotated_boxes = np.stack([new_y_min, new_x_min, new_y_max, new_x_max], axis=1)

            return rotated_boxes

        def _resize(rotated_boxes, original_size, new_size):
            """
            Rescale bounding boxes after the image is rotated and resized to fit the entire original image.

            Args:
                rotated_boxes (numpy.ndarray): A numpy array of shape (N, 4) with bounding boxes after rotation,
                                               where each row is [y_min, x_min, y_max, x_max].
                original_size (tuple): A tuple (width, height) representing the original size of the image.
                new_size (tuple): A tuple (width, height) representing the size of the rotated image.

            Returns:
                numpy.ndarray: A numpy array of shape (N, 4), where each row is the rescaled bounding box
                               [scaled_y_min, scaled_x_min, scaled_y_max, scaled_x_max].
            """
            # Original image dimensions
            w, h = original_size

            # Calculate the new width and height after rotation
            new_w, new_h = new_size

            # Image centers (before and after rotation)
            original_center = np.array([w / 2, h / 2])
            new_center = np.array([new_w / 2, new_h / 2])

            # Compute scaling factors for width and height
            scale_x = new_w / w
            scale_y = new_h / h

            # Get box centers and dimensions
            x_min = rotated_boxes[:, 1]
            y_min = rotated_boxes[:, 0]
            x_max = rotated_boxes[:, 3]
            y_max = rotated_boxes[:, 2]

            box_centers = np.stack([(x_min + x_max) / 2, (y_min + y_max) / 2], axis=1)
            box_dims = np.stack([x_max - x_min, y_max - y_min], axis=1)

            # Translate the box centers relative to the original image center
            translated_centers = box_centers - original_center

            # Scale the box centers and dimensions
            scaled_centers = translated_centers * [scale_x, scale_y] + new_center
            scaled_dims = box_dims * [scale_x, scale_y]

            # Compute new bounding boxes
            new_x_min = scaled_centers[:, 0] - scaled_dims[:, 0] / 2
            new_x_max = scaled_centers[:, 0] + scaled_dims[:, 0] / 2
            new_y_min = scaled_centers[:, 1] - scaled_dims[:, 1] / 2
            new_y_max = scaled_centers[:, 1] + scaled_dims[:, 1] / 2

            # Stack new bounding boxes
            scaled_boxes = np.stack([new_y_min, new_x_min, new_y_max, new_x_max], axis=1)

            return scaled_boxes

        if do_rotate:
            img = img.rotate(degrees, expand=True, fillcolor=self.fill_color, resample=Image.BILINEAR)

            if 'bbox' in annotations:
                new_image_size = img.size
                new_bboxs = _rotate(annotations['bbox'], orig_image_size, degrees)
                new_bboxs = _resize(new_bboxs, orig_image_size, new_image_size)
                annotations['bbox'] = new_bboxs

        return img, annotations


def resolve_fill_color(fill_color, img_mean=IMAGENET_DEFAULT_MEAN):
    if isinstance(fill_color, tuple):
        assert len(fill_color) == 3
        fill_color = fill_color
    else:
        try:
            int_color = int(fill_color)
            fill_color = (int_color,) * 3
        except ValueError:
            assert fill_color == 'mean'
            fill_color = tuple([int(round(255 * x)) for x in img_mean])
    return fill_color


class Compose:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, annotations: dict):
        for t in self.transforms:
            img, annotations = t(img, annotations)
        return img, annotations


def transforms_coco_eval(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    fill_color = resolve_fill_color(fill_color, mean)

    image_tfl = [
        ResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=fill_color),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf


def transforms_coco_train(
        img_size=224,
        interpolation='random',
        use_prefetcher=False,
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    fill_color = resolve_fill_color(fill_color, mean)

    image_tfl = [
        RandomFlip(horizontal=True, prob=0.5),
        RandomResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=fill_color),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf


def transforms_documents_train(
        img_size=224,
        interpolation='random',
        use_prefetcher=False,
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        rotation_degrees=2
):

    fill_color = resolve_fill_color(fill_color, mean)

    image_tfl = [
        RandomRotate(degrees=rotation_degrees, prob=0.5, fill_color=fill_color),
        RandomResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=fill_color),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf