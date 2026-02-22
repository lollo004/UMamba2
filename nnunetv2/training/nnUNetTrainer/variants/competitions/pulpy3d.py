from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import numpy as np
from typing import Union, Tuple, List

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

class nnUNetTrainer_Pulpy_TF3Transforms_NoMirror(nnUNetTrainer):
    """
    TF3-like augmentations + NO mirroring (train + inference).
    """
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        mirror_axes = None
        self.inference_allowed_mirroring_axes = None

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
        patch_size,
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,  # ignored
        do_dummy_2d_data_aug,
        use_mask_for_norm=None,
        is_cascaded=False,
        foreground_labels=None,
        regions=None,
        ignore_label=None,
    ):

        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.85, 1.25),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))

        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.25, 0.75),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
                benchmark=True
            ), apply_probability=0.1
        ))

        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))

        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))

        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.8, 1.0),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.15
        ))

        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))

        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))


        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i, u in enumerate(use_mask_for_norm) if u],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded:
            from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
            from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
                RemoveRandomConnectedComponentFromOneHotEncodingTransform
            from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform

            assert foreground_labels is not None
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(RandomTransform(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    strel_size=(1, 8),
                    p_per_label=1
                ), apply_probability=0.4
            ))
            transforms.append(RandomTransform(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                    p_per_label=1
                ), apply_probability=0.2
            ))

        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

class nnUNetTrainer_Pulpy_TF3_NoMirror_Weighted(nnUNetTrainer_Pulpy_TF3Transforms_NoMirror):
    """
    TF3 aug, NO mirror, + DC+CE with weighted CE.
    """
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300
        self.oversample_foreground_percent = 0.95
        self.initial_lr = 1e-3

    def _build_loss(self):
        weights = torch.ones(18, device=self.device, dtype=torch.float32)
        weights[0] = 0.05
        weights[1] = 5.0
        weights[2:] = 20.0

        ce_kwargs = {"weight": weights}

        soft_dice_kwargs = {
            "batch_dice": True,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": self.is_ddp
        }

        loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights_ds = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float32)

            if self.is_ddp and not self._do_i_compile():
                weights_ds[-1] = 1e-6
            else:
                weights_ds[-1] = 0.0

            weights_ds = weights_ds / weights_ds.sum()
            loss = DeepSupervisionWrapper(loss, weights_ds)

        return loss

class nnUNetTrainer_Pulpy_TF3_NoMirror_Weighted_accum2(nnUNetTrainer_Pulpy_TF3_NoMirror_Weighted):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.accumulate_steps = 2
        self.num_iterations_per_epoch *= 2
        self.num_val_iterations_per_epoch = 10


class nnUNetTrainer_Pulpy_debug(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300
        self.oversample_foreground_percent = 1
        self.initial_lr = 1e-3

        self.dbg_every = 1
        self.dbg_max_prints = 50
        self._dbg_prints_done = 0

    def _build_loss(self):
        weights = torch.ones(18, device=self.device, dtype=torch.float32)
        weights[0] = 0.05
        weights[1] = 5.0 
        weights[2:] = 20.0

        ce_kwargs = {"weight": weights}

        soft_dice_kwargs = {
            "batch_dice": True,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": self.is_ddp
        }

        loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights_ds = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float32)
            if self.is_ddp and not self._do_i_compile():
                weights_ds[-1] = 1e-6
            else:
                weights_ds[-1] = 0.0
            weights_ds = weights_ds / weights_ds.sum()
            loss = DeepSupervisionWrapper(loss, weights_ds)

        return loss

    def train_step(self, batch: dict, step_optimizer=True) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if step_optimizer:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.no_grad():
            output = self.network(data)
            l = self.loss(output, target)
            if self.accumulate_steps > 1:
                l = l / self.accumulate_steps

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            if step_optimizer:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grap_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
        else:
            l.backward()
            if step_optimizer:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grap_norm)
                self.optimizer.step()

        # ---------------- DEBUG ----------------
        if self._dbg_prints_done < self.dbg_max_prints:
            if (self._dbg_prints_done % self.dbg_every) == 0:
                with torch.no_grad():
                    t0 = target[0] if isinstance(target, list) else target
                    if t0.ndim == 5 and t0.shape[1] == 1:
                        t0 = t0[:, 0]

                    uniq = torch.unique(t0).detach().cpu().tolist()
                    fg = int((t0 > 0).sum().item())
                    tot = int(t0.numel())
                    fg_ratio = fg / max(tot, 1)

                    o0 = output[0] if isinstance(output, (tuple, list)) else output
                    out_C = int(o0.shape[1])

                    pred = torch.argmax(o0, dim=1)
                    pred_uniq = torch.unique(pred).detach().cpu().tolist()
                    pred_fg = int((pred > 0).sum().item())
                    pred_tot = int(pred.numel())
                    pred_fg_ratio = pred_fg / max(pred_tot, 1)

                    p = next(self.network.parameters())
                    gnorm = float(p.grad.detach().norm().item()) if p.grad is not None else 0.0

                    self.print_to_log_file(
                        f"[DBG] epoch={self.current_epoch} "
                        f"loss={float(l.detach().cpu().item()):.6f} "
                        f"target_unique={uniq} fg_ratio={fg_ratio:.6e} "
                        f"pred_unique={pred_uniq} pred_fg_ratio={pred_fg_ratio:.6e} "
                        f"out_C={out_C} grad_norm={gnorm:.3e}",
                        also_print_to_console=True
                    )
                self._dbg_prints_done += 1
        # ---------------------------------------

        l_np = l.detach().cpu().numpy()
        if self.accumulate_steps > 1:
            l_np = l_np * self.accumulate_steps
        return {'loss': l_np}