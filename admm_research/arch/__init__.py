""" Define the architecture. Modified from torchvision
"""
from .network import FCN8, FCN16, FCN32, UNet, SegNet, PSPNet
from .enet import Enet
from .joseent import ENet as JEnet

"""
Package
"""
# A Map from string to arch callables
ARCH_CALLABLES = {}


def _register_arch(arch, callable, alias=None):
    """ Private method to register the architecture to the ARCH_CALLABLES
        :param arch: A str
        :param callable: The callable that return the nn.Module
        :param alias: None, or a list of string, or str
    """
    if arch in ARCH_CALLABLES:
        raise ValueError('{} already exists!'.format(arch))
    ARCH_CALLABLES[arch] = callable
    if alias:
        if isinstance(alias, str):
            alias = [alias]
        for other_arch in alias:
            if other_arch in ARCH_CALLABLES:
                raise ValueError('alias {} for {} already exists!'.format(other_arch, arch))
            ARCH_CALLABLES[other_arch] = callable


# Adding architecture (new architecture goes here...)
_register_arch('fcn8', FCN8)
_register_arch('fcn16', FCN16)
_register_arch('fcn32', FCN32)
_register_arch('unet', UNet)
_register_arch('segnet', SegNet)
_register_arch('enet', Enet)
_register_arch('jenet', JEnet)

"""
Public interface
"""


def get_arch(arch, kwargs):
    """ Get the architecture. Return a torch.nn.Module """
    arch_callable = ARCH_CALLABLES.get(arch)
    kwargs.pop('arch')
    assert arch_callable, "Architecture {} is not found!".format(arch)
    return arch_callable(**kwargs)
