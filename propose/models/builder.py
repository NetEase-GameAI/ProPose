from torch import nn

from propose.utils import Registry, build_from_cfg


MODEL_FACTORY = Registry('model')
LOSS_FACTORY = Registry('loss')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_model(cfg):
    return build(cfg, MODEL_FACTORY)


def build_loss(cfg, **kwargs):
    return build(cfg, LOSS_FACTORY, default_args=kwargs)
