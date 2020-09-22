# -*- coding: utf-8 -*-

import os
import copy
import glob
import json
import argparse
import datetime


class ConfigBase(dict):
    def __init__(self, args: (dict, argparse.Namespace), **kwargs):
        super(ConfigBase, self).__init__()
        if isinstance(args, dict):
            d = args
        elif isinstance(args, argparse.Namespace):
            d = copy.deepcopy(vars(args))
        else:
            raise AttributeError
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

        self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
            try:
                del value['hash']  # Removing hash from nested dictionary
            except KeyError:
                pass
        super(ConfigBase, self).__setattr__(name, value)
        super(ConfigBase, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'configs.json')
        with open(path, 'w') as fp:
            out = {k: v for k, v in self.items()}
            out['task'] = self.task
            out['model_name'] = self.model_name
            out['checkpoint_dir'] = self.checkpoint_dir
            json.dump(out, fp, indent=2)


class ClassificationConfig(ConfigBase):
    """Configurations for Classification."""
    def __init__(self, args, **kwargs):
        super(ClassificationConfig, self).__init__(args, **kwargs)

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,  # 'wm811k', 'cifar10', 'stl10', 'imagenet', ...
            self.task,  # 'scratch', 'denoising', 'pirl', 'simclr', ...
            self.model_name,
            self.augmentation,
            f'SEED_{self.seed:03}',
            f'LABEL_{self.label_proportion:.3f}',
            self.hash
            )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def model_name(self):
        return f'{self.backbone_type}.{self.backbone_config}'

    @property
    def task(self):
        suffix = '_scratch' if self.pretext is None else f'_{self.pretext}'
        return 'classification' + suffix

    def find_pretrained_model(self, root: str = None, name: str = None):
        """Find pretrained model under `root` directory. Duplicates should be manually avoided."""

        root = f"./checkpoints/{self.data}/{self.pretext}/" if root is None else root
        name = f"last_model.pt" if name is None else name  # FIXME: best vs. last

        config_files = glob.glob(os.path.join(root, "**/configs.json"), recursive=True)

        if self.pretext == 'denoising':
            candidates = self._find_denoising_models(config_files, name)
        elif self.pretext == 'inpainting':
            candidates = self._find_inpainting_models(config_files, name)
        elif self.pretext == 'jigsaw':
            candidates = self._find_jigsaw_models(config_files, name)
        elif self.pretext == 'rotation':
            candidates = self._find_rotation_models(config_files, name)
        elif self.pretext == 'bigan':
            candidates = self._find_bigan_models(config_files, name)
        elif self.pretext == 'pirl':
            candidates = self._find_pirl_models(config_files, name)
        elif self.pretext == 'simclr':
            candidates = self._find_simclr_models(config_files, name)
        elif self.pretext == 'semiclr':
            candidates =self._find_semiclr_models(config_files, name)
        elif self.pretext == 'attnclr':
            candidates = self._find_attnclr_models(config_files, name)
        else:
            raise NotImplementedError

        if len(candidates) == 0:
            raise IndexError(f"No available {self.pretext} model.")
        elif len(candidates) > 1:
            raise IndexError(f"{len(candidates)} multiple candidates available.")
        else:
            return candidates[0]

    def _find_denoising_models(self, config_files: list, name: str):
        candidates = []
        for config_file in config_files:
            with open(config_file, 'r') as fp:
                cfg = json.load(fp)
            conds = [
                cfg.get('backbone_type') == self.backbone_type,
                cfg.get('backbone_config') == self.backbone_config,
                cfg.get('augmentation') == self.augmentation,
                cfg.get('noise') == self.noise,
            ]
            if all(conds):
                pt_file = os.path.join(os.path.dirname(config_file), name)
                if os.path.isfile(pt_file):
                    candidates += [(pt_file, config_file)]

        return candidates

    def _find_inpainting_models(self, config_files: list, name: str):
        raise NotImplementedError

    def _find_jigsaw_models(self, config_files: list, name: str):
        candidates = []
        for config_file in config_files:
            with open(config_file, 'r') as fp:
                cfg = json.load(fp)
            conds =[
                cfg.get('backbone_type') == self.backbone_type,
                cfg.get('backbone_config') == self.backbone_config,
                # cfg.get('in_channels') == self.in_channels,
                cfg.get('num_patches') == self.num_patches,
                cfg.get('num_permutations') == self.num_permutations,
            ]
            if all(conds):
                pt_file = os.path.join(os.path.dirname(config_file), name)
                if os.path.isfile(pt_file):
                    candidates += [(pt_file, config_file)]

        return candidates

    def _find_rotation_models(self, config_files: list, name: str):
        candidates = []
        for config_file in config_files:
            with open(config_file, 'r') as fp:
                cfg = json.load(fp)
            conds = [
                cfg.get('backbone_type') == self.backbone_type,
                cfg.get('backbone_config') == self.backbone_config,
                # cfg.get('in_channels') == self.in_channels,
            ]
            if all(conds):
                pt_file = os.path.join(os.path.dirname(config_file), name)
                if os.path.isfile(pt_file):
                    candidates += [(pt_file, config_file)]

        return candidates

    def _find_bigan_models(self, config_files: list, name: str):
        candidates = []
        for config_file in config_files:
            with open(config_file, 'r') as fp:
                cfg = json.load(fp)
            conds =[
                cfg.get('backbone_type') == self.backbone_type,
                cfg.get('backbone_config') == self.backbone_config,
                # cfg.get('in_channels') == self.in_channels,
                cfg.get('gan_type') == self.gan_type,
                cfg.get('gan_config') == self.gan_config,
            ]
            if all(conds):
                pt_file = os.path.join(os.path.dirname(config_file), name)
                if os.path.isfile(pt_file):
                    candidates += [(pt_file, config_file)]

        return candidates

    def _find_pirl_models(self, config_files: list, name: str):
        candidates = []
        for config_file in config_files:
            with open(config_file, 'r') as fp:
                cfg = json.load(fp)
            conds = [
                cfg.get('backbone_type') == self.backbone_type,
                cfg.get('backbone_config') == self.backbone_config,
                cfg.get('projector_type') == self.projector_type,
                cfg.get('projector_size') == self.projector_size,
                cfg.get('num_negatives') == self.num_negatives,
                cfg.get('augmentation') == self.augmentation,
            ]
            if all(conds):
                pt_file = os.path.join(os.path.dirname(config_file), name)
                if os.path.isfile(pt_file):
                    candidates += [(pt_file, config_file)]

        return candidates

    def _find_simclr_models(self, config_files: list, name: str):
        candidates = []
        for config_file in config_files:
            with open(config_file, 'r') as fp:
                cfg = json.load(fp)
            conds = [
                cfg.get('data') == self.data,
                cfg.get('input_size') == self.input_size,
                cfg.get('backbone_type') == self.backbone_type,
                cfg.get('backbone_config') == self.backbone_config,
                cfg.get('projector_type') == self.projector_type,
                cfg.get('projector_size') == self.projector_size,
            ]
            if all(conds):
                pt_file = os.path.join(os.path.dirname(config_file), name)
                if os.path.isfile(pt_file):
                    candidates += [(pt_file, config_file)]

        return candidates

    def _find_semiclr_models(self, config_files: list, name: str):
        return self._find_simclr_models(config_files, name)

    def _find_attnclr_models(self, config_files: list, name: str):
        return self._find_simclr_models(config_files, name)

class DenoisingConfig(ConfigBase):
    """Configurations for Denoising."""
    def __init__(self, args, **kwargs):
        super(DenoisingConfig, self).__init__(args, **kwargs)

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,
            self.task,
            self.model_name,
            self.hash,
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def task(self):
        return 'denoising'

    @property
    def model_name(self):
        return f'{self.backbone_type}.{self.backbone_config}'


class InpaintingConfig(ConfigBase):
    """Configurations for Inpainting."""
    def __init__(self, args, **kwargs):
        super(InpaintingConfig, self).__init__(args, **kwargs)

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,
            self.task,
            self.model_name,
            self.hash,
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def task(self):
        return 'inpainting'

    @property
    def model_name(self):
        return f'{self.backbone_type}.{self.backbone_config}'


class JigsawConfig(ConfigBase):
    """Configurations for Jigsaw."""
    def __init__(self, args, **kwargs):
        super(JigsawConfig, self).__init__(args, **kwargs)

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,
            self.task,
            self.model_name,
            self.hash,
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def task(self):
        return 'jigsaw'

    @property
    def model_name(self):
        return f'{self.backbone_type}.{self.backbone_config}'


class RotationConfig(ConfigBase):
    """Configurations for Rotation."""
    def __init__(self, args, **kwargs):
        super(RotationConfig, self).__init__(args, **kwargs)

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,
            self.task,
            self.model_name,
            self.hash,
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def task(self):
        return 'rotation'

    @property
    def model_name(self):
        return f"{self.backbone_type}.{self.backbone_config}"


class BiGANConfig(ConfigBase):
    """Configurations for BiGAN."""
    def __init__(self, args, **kwargs):
        super(BiGANConfig, self).__init__(args, **kwargs)

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,
            self.task,
            self.model_name,
            self.hash,
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def task(self):
        return 'bigan'

    @property
    def model_name(self):
        return f'{self.backbone_type}.{self.backbone_config}'


class PIRLConfig(ConfigBase):
    """Configurations for PIRL."""
    def __init__(self, args, **kwargs):
        super(PIRLConfig, self).__init__(args, **kwargs)

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,
            self.task,
            self.model_name,
            self.augmentation,
            self.hash,
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def task(self):
        return 'pirl'

    @property
    def model_name(self):
        return f'{self.backbone_type}.{self.backbone_config}'


class SimCLRConfig(ConfigBase):
    """Configurations for SimCLR."""
    def __init__(self, args, **kwargs):
        super(SimCLRConfig, self).__init__(args, **kwargs)

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,
            self.task,
            self.model_name,
            self.hash,
        )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def task(self):
        return 'simclr'

    @property
    def model_name(self):
        return f'{self.backbone_type}.{self.backbone_config}'


class SemiCLRConfig(SimCLRConfig):
    """Configurations for SemiCLR."""
    def __init__(self, args, **kwargs):
        super(SemiCLRConfig, self).__init__(args, **kwargs)

    @property
    def task(self):
        return 'semiclr'


class AttnCLRConfig(SimCLRConfig):
    """Configurations for AttnCLR."""
    def __init__(self, args, **kwargs):
        super(AttnCLRConfig, self).__init__(args, **kwargs)

    @property
    def task(self):
        return 'attnclr'


class MixupConfig(ClassificationConfig):
    def __init__(self, args, **kwargs):
        super(MixupConfig, self).__init__(args, **kwargs)

    @property
    def task(self):
        return 'mixup-disabled' if self.disable_mixup else 'mixup'
