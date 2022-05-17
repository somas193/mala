"""Interface function for getting Targets."""
from .ldos import LDOS
from .dos import DOS
from .density import Density
from .energy_density import EnergyDensity


def TargetInterface(params):
    """
    Return correct target calculator/parser, based on parameters.

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this TargetBase object.

    Returns
    -------
    target : mala.targets.targets.TargetBase or derivative

    """
    if params.targets.target_type == 'LDOS':
        return LDOS(params)
    elif params.targets.target_type == 'DOS':
        return DOS(params)
    elif params.targets.target_type == 'Density':
        return Density(params)
    elif params.targets.target_type == 'Energy density':
        return EnergyDensity(params)
    else:
        raise Exception("Unknown type of target parser requested.")
