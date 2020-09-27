# -*- coding: utf-8 -*-

import os
import copy
import json
import glob
import datetime
import argparse


class ConfigBaseLegacy(dict):
    def __init__(self, args: (dict, argparse.Namespace), **kwargs):
        super(ConfigBaseLegacy, self).__init__()
        if isinstance(args, dict):
            d = args
        elif isinstance(args, argparse.Namespace):
            d = copy.deepcopy(vars(args))
        else:
            raise AttributeError
        if kwargs:
            d.update(kwargs)
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
        super(ConfigBaseLegacy, self).__setattr__(name, value)
        super(ConfigBaseLegacy, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'configs.json')
        with open(path, 'w') as fp:
            out = {k: v for k, v in self.items()}
            out['task'] = self.task
            out['model_name'] = self.model_name
            out['checkpoint_dir'] = self.checkpoint_dir
            json.dump(out, fp, indent=2)


class ClassificationConfigLegacy(ConfigBaseLegacy):
    def __init__(self, args, **kwargs):
        super(ClassificationConfigLegacy, self).__init__()

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
                cfg.get('temperature') == self.temperature,
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
