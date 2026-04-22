import warnings
warnings.filterwarnings("ignore")

import multiprocessing as mp
import torch.multiprocessing as tmp

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    CenterSpatialCropd,
    ToTensord,
    MapTransform,
    ResizeWithPadOrCropd,
    CropForegroundd,
    LoadImaged,
    EnsureTyped,
    Orientationd,
    Spacingd,
    SpatialPadd,
    CenterSpatialCropd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged
)

from monai.transforms import DeleteItemsd
import numpy as np
import json
import argparse
import h5py

from numbers import Number
import torch.distributed as dist
from tqdm import tqdm

from lighter_zoo import SegResEncoder


class MaskCenterCropd(MapTransform):
    """Custom MONAI transform to crop around mask center with padding."""

    def __init__(self, keys, mask_key="mask", roi_size=(224, 224, 160), fg_labels=None):
        super().__init__(keys)
        self.mask_key = mask_key
        self.roi_size = roi_size
        self.fg_labels = fg_labels
        self.img_key = 'image'

    def __call__(self, data):
        d = dict(data)

        # Get mask center
        mask_arr = d[self.mask_key]
        if len(mask_arr.shape) == 4:  # Remove channel dimension if present
            mask_arr = mask_arr[0]

            # make binary mask for specified foreground labels
        if self.fg_labels is not None:
            mask_arr = np.isin(mask_arr, self.fg_labels).astype(np.uint8)

            coords = np.argwhere(mask_arr == 1)
            if coords.size == 0:
                print(f"Mask unique values: {np.unique(mask_arr)}")
                print(f'Original unique values in mask: {np.unique(d[self.mask_key])}')
                print(f'shapes of image and mask: image {d["image"].shape=}, mask {d[self.mask_key].shape=}')
                mask_arr_orig = d['mask_original']
                if len(mask_arr_orig.shape) == 4:  # Remove channel dimension if present
                    mask_arr_orig = mask_arr_orig[0]
                mask_arr_orig = np.isin(mask_arr_orig, self.fg_labels).astype(np.uint8)
                shape_ori = mask_arr_orig.shape
                shape_resampled = mask_arr.shape
                coords = np.argwhere(mask_arr_orig == 1)
                if coords.size == 0:
                    print(f"================== No coordinates found in original mask, using image center.")
                    coords = np.array([[shape_resampled[0] // 2, shape_resampled[1] // 2, shape_resampled[2] // 2]])
                else:
                    scale_z = shape_resampled[0] / shape_ori[0]
                    scale_y = shape_resampled[1] / shape_ori[1]
                    scale_x = shape_resampled[2] / shape_ori[2]
                    coords = np.array([[int(c[0] * scale_z), int(c[1] * scale_y), int(c[2] * scale_x)] for c in coords])
            center = tuple(coords.mean(axis=0).astype(int))

        else:
            img_arr = d[self.img_key]
            shape_img = img_arr.shape[1:] if len(img_arr.shape) == 4 else img_arr.shape
            center = (shape_img[0] // 2, shape_img[1] // 2, shape_img[2] // 2)

        for key in self.keys:
            arr = d[key]
            has_channel = len(arr.shape) == 4
            if has_channel:
                arr_data = arr[0]
            else:
                arr_data = arr
            cropped = self._crop_with_padding(arr_data, center, self.roi_size)

            if has_channel:
                d[key] = cropped[np.newaxis, ...]
            else:
                d[key] = cropped

        return d

    def _crop_with_padding(self, arr, center, size):
        """Crop 3D array with zero padding around center (z, y, x)."""
        zc, yc, xc = center
        dz, dy, dx = size[0] // 2, size[1] // 2, size[2] // 2

        z_start, z_end = zc - dz, zc + dz
        y_start, y_end = yc - dy, yc + dy
        x_start, x_end = xc - dx, xc + dx

        if torch.is_tensor(arr):
            cropped = torch.zeros(size, dtype=arr.dtype, device=arr.device)
        else:
            cropped = np.zeros(size, dtype=arr.dtype)

        z_start_valid = max(z_start, 0)
        y_start_valid = max(y_start, 0)
        x_start_valid = max(x_start, 0)

        z_end_valid = min(z_end, arr.shape[0])
        y_end_valid = min(y_end, arr.shape[1])
        x_end_valid = min(x_end, arr.shape[2])

        z_off = z_start_valid - z_start
        y_off = y_start_valid - y_start
        x_off = x_start_valid - x_start

        cropped[
        z_off:z_off + (z_end_valid - z_start_valid),
        y_off:y_off + (y_end_valid - y_start_valid),
        x_off:x_off + (x_end_valid - x_start_valid)
        ] = arr[
            z_start_valid:z_end_valid,
            y_start_valid:y_end_valid,
            x_start_valid:x_end_valid
            ]

        return cropped


def get_first_valid_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of the specified keys found: {keys}")


def torch_resample_to_spacing(
        data: np.ndarray,
        current_spacing: tuple,
        new_spacing: tuple,
        is_seg: bool = False,
        order: int = 3,
        device: str = 'cuda',
) -> np.ndarray:
    assert data.ndim == 4, "data must be (c, x, y, z)"

    old_shape = np.array(data.shape[1:])
    current_spacing = np.array(current_spacing)
    new_spacing = np.array(new_spacing)

    new_shape = np.round(old_shape * current_spacing / new_spacing).astype(int)

    data_torch = torch.from_numpy(data).float().to(device)
    data_torch = data_torch.unsqueeze(0)

    if is_seg or order == 0:
        mode = 'nearest'
        align_corners = None
    elif order == 1:
        mode = 'trilinear'
        align_corners = False
    else:
        mode = 'trilinear'
        align_corners = False

    resampled = F.interpolate(
        data_torch,
        size=tuple(new_shape),
        mode=mode,
        align_corners=align_corners
    )

    resampled = resampled.squeeze(0).cpu().numpy()

    return resampled


class CopyMaskd(MapTransform):
    def __init__(self, keys, mask_key, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        for key in self.mask_key:
            mask = d[key]
            d[f'{key}_original'] = mask.clone()
        return d


class ResampleToSpacingd(MapTransform):
    def __init__(self, keys, target_spacing=(1.0, 1.0, 1.0), order=3, allow_missing_keys=False, is_seg_list=None,
                 device='cuda'):
        super().__init__(keys, allow_missing_keys)
        self.target_spacing = target_spacing
        self.order = order
        self.is_seg_list = is_seg_list
        self.device = device if torch.cuda.is_available() else 'cpu'
        assert len(keys) == len(is_seg_list), "Length of keys and is_seg_list must match"

    def __call__(self, data):
        d = dict(data)
        for key, is_seg in zip(self.key_iterator(d), self.is_seg_list):
            image = d[key]
            properties = d.get(f'{key}_properties', None)
            if properties is None:
                raise ValueError(f"Properties not found for {key}. Make sure SimpleITKLoadImaged was run first.")

            current_spacing = properties['spacing']

            resampled = torch_resample_to_spacing(
                data=image,
                current_spacing=current_spacing,
                new_spacing=self.target_spacing,
                is_seg=is_seg,
                order=self.order,
                device=self.device
            )
            d[key] = resampled
        return d


def extract_on_gpu(rank, args, datalist):
    """Feature extraction worker — runs on a single GPU identified by rank."""
    device = torch.device(f"cuda:{rank}")

    # Each rank processes its own slice of the datalist
    world_size = torch.cuda.device_count()
    worker_datalist = datalist[rank::world_size]

    model = SegResEncoder.from_pretrained(args.pretrained_weights)
    model.eval()
    model.to(device)

    image_size = 320 if args.masks_path is None else 160

    if args.masks_path is None:
        ImageTransforms = Compose([
            LoadImaged(keys=["image"], ensure_channel_first=True),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=['image'], axcodes="SPL"),
            Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            ScaleIntensityRanged(keys=["image"], a_min=args.hu_min, a_max=args.hu_max, b_min=0, b_max=1, clip=True),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=(image_size, image_size, image_size)),
            ToTensord(keys=["image"]),
        ])
    else:
        ImageTransforms = Compose([
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            EnsureTyped(keys=["image"]),
            CopyMaskd(keys=["mask"], mask_key=["mask"]),
            Orientationd(keys=['image', 'mask'], axcodes="SPL"),
            Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            Spacingd(keys=['mask'], pixdim=(1.0, 1.0, 1.0), mode='nearest'),
            MaskCenterCropd(keys=["image", "mask"], mask_key="mask", roi_size=(image_size, image_size, image_size), fg_labels=[1]),
            DeleteItemsd(keys=["mask_original"]),
            DeleteItemsd(keys=['mask']),
            ScaleIntensityRanged(keys=["image"], a_min=args.hu_min, a_max=args.hu_max, b_min=0, b_max=1, clip=True),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=(image_size, image_size, image_size)),
            ToTensord(keys=["image"]),
        ])

    from monai.data import ThreadDataLoader, Dataset

    dataset = Dataset(data=worker_datalist, transform=ImageTransforms)
    dataloader = ThreadDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    os.makedirs(args.dest, exist_ok=True)

    prefix = f"[GPU {rank}]"
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{prefix} Extracting features", position=rank)):
            if isinstance(batch, dict):
                images = batch["image"]
                filenames = batch.get('filename', [f'batch_{batch_idx}_sample_{i}' for i in range(images.shape[0])])
            else:
                raise ValueError(f"Expected dict batch, got {type(batch)}")

            if not isinstance(filenames, (list, tuple)):
                filenames = [filenames]

            images = images.to(device, non_blocking=True)

            output = model(images)[-1]
            image_embeddings = F.adaptive_avg_pool3d(output, 1).view(images.shape[0], -1)
            image_embeddings = image_embeddings.detach().cpu()

            for i, filename in enumerate(filenames):
                single_out_path = os.path.join(args.dest, f'{filename}.h5')
                with h5py.File(single_out_path, 'w') as hf:
                    hf.create_dataset('y_hat', data=image_embeddings[i].numpy())

            del output, images, image_embeddings, batch
            torch.cuda.empty_cache()


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", "--imgs_path", dest="imgs_path", type=str,
                    default='/workspace/inputs',
                    help='Path to input images directory')
    ap.add_argument("-o", "--output", "--dest", dest="dest", type=str,
                    default='/workspace/outputs',
                    help='Destination folder to save features')
    ap.add_argument("--masks_path", type=str, default=None,
                    help='Path to foreground masks for roi-disease (set to None for non-roi diseases)')
    ap.add_argument("--batch_size", type=int, default=1, help='Batch size for feature extraction')
    ap.add_argument("--num_workers", type=int, default=0, help='Number of workers for data loading')
    ap.add_argument("--num_classes", type=int, default=2, help='Number of classes for classification')
    ap.add_argument("--cache-dir", type=str, default=None, help="path to cache directory for monai persistent dataset")
    ap.add_argument("--hu_min", type=float, default=-1024, help="HU window lower bound for intensity normalization")
    ap.add_argument("--hu_max", type=float, default=2048, help="HU window upper bound for intensity normalization")
    ap.add_argument(
        "--pretrained-weights",
        type=str,
        default="/opt/app/ct_fm_weights",
        help="Path to CT-FM pretrained weights directory",
    )

    args = ap.parse_args()

    imgs_path = args.imgs_path
    os.makedirs(args.dest, exist_ok=True)

    datalist = []
    imgs_files = sorted([f for f in os.listdir(imgs_path) if f.endswith('.nii.gz')])
    if args.masks_path:
        imgs_files = [f for f in imgs_files if os.path.exists(os.path.join(args.masks_path, f))]

    for img_file in imgs_files:
        img_id = img_file.split('.nii.gz')[0]
        img_full_path = os.path.join(imgs_path, img_file)
        mask_full_path = os.path.join(args.masks_path, img_file) if args.masks_path is not None else None
        assert os.path.exists(img_full_path), f'Image file not found: {img_full_path}'
        if mask_full_path is not None:
            assert os.path.exists(mask_full_path), f'Mask file not found: {mask_full_path}'
            datalist.append({"image": img_full_path, "mask": mask_full_path, 'filename': img_id})
        else:
            datalist.append({"image": img_full_path, 'filename': img_id})

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        extract_on_gpu(0, args, datalist)
    else:
        print(f"Using {num_gpus} GPUs")
        tmp.spawn(extract_on_gpu, args=(args, datalist), nprocs=num_gpus, join=True)
