from .ADMM_refactor import AdmmSize, AdmmGCSize
from .fullysupervised import FullySupervisedWrapper
from .ADMM_refactor_3D import AdmmGCSize3D

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
_register_arch('size', AdmmSize)
_register_arch('gc_size', AdmmGCSize)
_register_arch('gc_size_3d', AdmmGCSize3D)
_register_arch('fs', FullySupervisedWrapper)

"""
Public interface
"""


# def get_method(method_, torchnet, **kwargs):
#     """ Get the architecture. Return a torch.nn.Module """
#     arch_callable = ARCH_CALLABLES.get(method_)
#     assert arch_callable, "Architecture {} is not found!".format(method_)
#     return arch_callable(torchnet, kwargs)


def get_method_class(method_, ):
    """ Get the architecture. Return a torch.nn.Module """
    arch_callable = ARCH_CALLABLES.get(method_)
    assert arch_callable, "Architecture {} is not found!".format(method_)
    return arch_callable
