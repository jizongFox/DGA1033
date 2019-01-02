from .ADMM import AdmmSize, ModelMode, AdmmGCSize
from .fullysupervised import FullySupervisedWrapper
from .ADMM_in import ADMM_size_inequality, ADMM_reg_size_inequality

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
_register_arch('admm_size', AdmmSize)
_register_arch('admm_gc_size', AdmmGCSize)
_register_arch('fullysupervised', FullySupervisedWrapper)
_register_arch('admm_size_in', ADMM_size_inequality)
_register_arch('admm_gc_size_in', ADMM_reg_size_inequality)

"""
Public interface
"""


def get_method(method_, torchnet, **kwargs):
    """ Get the architecture. Return a torch.nn.Module """
    arch_callable = ARCH_CALLABLES.get(method_)
    assert arch_callable, "Architecture {} is not found!".format(method_)
    return arch_callable(torchnet, kwargs)


def get_method_class(method_, ):
    """ Get the architecture. Return a torch.nn.Module """
    arch_callable = ARCH_CALLABLES.get(method_)
    assert arch_callable, "Architecture {} is not found!".format(method_)
    return arch_callable
