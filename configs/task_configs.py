# -*- coding: utf-8 -*-

import os
import copy
import json
import argparse
import datetime


class ConfigBase(object):
    def __init__(self, args: (dict, argparse.Namespace) = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    @classmethod
    def from_command_line(cls):
        """Create a configuration object from command line arguments."""
        parents = [
            cls.data_parser(),           # task-agnostic
            cls.model_parser(),          # task-agnostic
            cls.train_parser(),          # task-agnostic
            cls.logging_parser(),        # task-agnostic
            cls.task_specific_parser(),  # task-specific
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(namespace=config)  # sets parsed arguments as attributes of namespace

        return config

    @classmethod
    def from_json(cls, json_path: str):
        """Create a configuration object from a .json file."""
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, path: str = None):
        """Save configurations to a .json file."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['model_name'] = self.model_name
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def task(self):
        raise NotImplementedError

    @property
    def model_name(self):
        return f'{self.backbone_type}.{self.backbone_config}'

    @property
    def checkpoint_dir(self):
        ckpt = os.path.join(
            self.checkpoint_root,
            self.data,          # 'wm811k', 'cifar10', 'stl10', 'imagenet', ...
            self.task,          # 'scratch', 'denoising', 'pirl', 'simclr', ...
            self.model_name,    # 'resnet.50.original', ...
            self.hash           # ...
            )
        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        raise NotImplementedError

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data', type=str, choices=('wm811k', 'cifar10', 'stl10', 'imagenet'), required=True)
        parser.add_argument('--input_size', type=int, choices=(32, 64, 96, 112, 224), required=True)
        parser.add_argument('--augmentation', type=str, default='test', required=True)
        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("CNN Backbone", add_help=False)
        parser.add_argument('--backbone_type', type=str, default='resnet', choices=('resnet', 'vggnet', 'alexnet'), required=True)
        parser.add_argument('--backbone_config', type=str, default='50.tiny', required=True)
        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size.')
        parser.add_argument('--num_workers', type=int, default=0, help='Number of CPU threads.')
        parser.add_argument('--device', type=str, default='cuda:0', help='Device to train model on.')
        parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adamw', 'lars'), help='Optimization algorithm.')
        parser.add_argument('--learning_rate', type=float, default=1e-2, help='Base learning rate to start from.')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay factor.')
        parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor, for SGD only.')
        parser.add_argument('--scheduler', type=str, default='cosine', choices=('cosine', 'restart', 'step', 'none'))
        parser.add_argument('--milestone', type=int, default=None, help='Number of epochs for which after learning rate is decayed.')
        parser.add_argument('--warmup_steps', type=int, default=5, help='Number of epochs for linear warmups up to base learning rate.')
        parser.add_argument('--cycles', type=int, default=1, help='Total number of cycles for hard restarts.')
        return parser

    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/', help='Top-level directory of checkpoints.')
        parser.add_argument('--write_summary', action='store_true', help='Write summaries with TensorBoard.')
        parser.add_argument('--save_every', type=int, default=None, help='Save model checkpoint every `save_every` epochs.')
        return parser


class PretrainConfigBase(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(PretrainConfigBase, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        raise NotImplementedError


class DenoisingConfig(PretrainConfigBase):
    """Configurations for Denoising."""
    def __init__(self, args=None, **kwargs):
        super(DenoisingConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        parser = argparse.ArgumentParser("Denoising", add_help=False)
        parser.add_argument('--noise', type=float, default=0.05)
        return parser

    @property
    def task(self):
        return 'denoising'


class InpaintingConfig(PretrainConfigBase):
    """Configurations for Inpainting."""
    def __init__(self, args=None, **kwargs):
        super(InpaintingConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        return 'inpainting'


class JigsawConfig(PretrainConfigBase):
    """Configurations for Jigsaw."""
    def __init__(self, args=None, **kwargs):
        super(JigsawConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        return 'jigsaw'


class RotationConfig(PretrainConfigBase):
    """Configurations for Rotation."""
    def __init__(self, args=None, **kwargs):
        super(RotationConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        return 'rotation'


class BiGANConfig(PretrainConfigBase):
    """Configurations for BiGAN."""
    def __init__(self, args=None, **kwargs):
        super(BiGANConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        raise NotImplementedError

    @property
    def task(self):
        return 'bigan'


class PIRLConfig(PretrainConfigBase):
    """Configurations for PIRL."""
    def __init__(self, args=None, **kwargs):
        super(PIRLConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        parser = argparse.ArgumentParser('PIRL', add_help=False)
        parser.add_argument('--projector_type', type=str, default='linear', choices=('linear', 'mlp'))
        parser.add_argument('--projector_size', type=int, default=128, help='Dimension of projector head.')
        parser.add_argument('--temperature',  type=float, default=0.07, help='Logit scaling factor.')
        parser.add_argument('--num_negatives', type=int, default=5000, help='Number of negative examples.')
        parser.add_argument('--loss_weight', type=float, default=0.5, help='Weighting factor of two loss terms, [0, 1].')
        return parser

    @property
    def task(self):
        return 'pirl'


class SimCLRConfig(PretrainConfigBase):
    """Configurations for SimCLR."""
    def __init__(self, args=None, **kwargs):
        super(SimCLRConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        parser = argparse.ArgumentParser('SimCLR', add_help=False)
        parser.add_argument('--projector_type', type=str, default='mlp', choices=('linear', 'mlp'))
        parser.add_argument('--projector_size', type=int, default=128, help='Dimension of projection head.')
        parser.add_argument('--temperature', type=float, default=0.07, help='Logit scaling factor.')
        parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from.')
        return parser

    @property
    def task(self):
        return 'simclr'


class SemiCLRConfig(SimCLRConfig):
    """Configurations for SemiCLR."""
    def __init__(self, args=None, **kwargs):
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


class DownstreamConfigBase(ConfigBase):
    """Configurations for downstream tasks."""
    def __init__(self, args=None, **kwargs):
        super(DownstreamConfigBase, self).__init__(args, **kwargs)


class ClassificationConfig(DownstreamConfigBase):
    def __init__(self, args=None, **kwargs):
        super(ClassificationConfig, self).__init__(args, **kwargs)

    @staticmethod
    def task_specific_parser():
        parser = argparse.ArgumentParser('Linear evaluation of pre-trained model.', add_help=False)
        parser.add_argument('--label_proportion', type=float, default=1.0, help='Size of labeled data (0, 1].')
        parser.add_argument('--pretrained_model_file', type=str, default=None, help='Path to pretrained model file (.pt).')
        parser.add_argument('--pretrained_model_type', type=str, default=None, help='Type of pretraining task.')
        parser.add_argument('--freeze', action='store_true')
        parser.add_argument('--label_smoothing', type=float, default=0.0, help='Ratio of label smoothing, [0, 1].')
        parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate in fully-connected layers, [0, 1].')
        return parser

    @property
    def task(self):
        if self.pretrained_model_type is None:
            return 'from_scratch'
        else:
            if self.freeze:
                return f'linear_eval_{self.pretrained_model_type}'
            else:
                return f'finetune_{self.pretrained_model_type}'


class MixupConfig(ClassificationConfig):
    def __init__(self, args=None, **kwargs):
        super(MixupConfig, self).__init__(args, **kwargs)

    @property
    def task(self):
        return 'mixup'
