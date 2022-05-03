"""Electronic density calculation class."""
from .target_base import TargetBase
from .calculation_helpers import *
from .cube_parser import read_cube
import warnings
import ase.io
from ase.units import Rydberg
from mala.common.parameters import printout
try:
    import total_energy as te
except ModuleNotFoundError:
    warnings.warn("You either don't have the QuantumEspresso total_energy "
                  "python module installed or it is not "
                  "configured correctly. Using a density calculator will "
                  "still mostly work, but trying to "
                  "access the total energy of a system WILL fail.",
                  stacklevel=2)


class EnergyDensity(TargetBase):
    """Postprocessing / parsing functions for the electronic density.

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this TargetBase object.
    """

    te_mutex = False

    def __init__(self, params):
        super(EnergyDensity, self).__init__(params)
        # We operate on a per gridpoint basis. Per gridpoint,
        # there is one value for the density (spin-unpolarized calculations).
        self.target_length = 1

    @staticmethod
    def convert_units(array, in_units="None"):
        """
        Convert the units of a SNAP descriptor.

        Since these do not really have units this function does nothing yet.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array.

        Returns
        -------
        converted_array : numpy.array
            Data in MALA units.
        """
        if in_units == "None":
            return array
        else:
            raise Exception("Unsupported unit for electronic density.")

    def read_from_cube(self, file_name, directory, units=None):
        """
        Read the density data from a cube file.

        Parameters
        ----------
        file_name :
            Name of the cube file.

        directory :
            Directory containing the cube file.

        units : string
            Units the density is saved in. Usually none.
        """
        printout("Reading density from .cube file in ", directory)
        data, meta = read_cube(directory + file_name)
        return data

    def get_entropy_contribution(self, energy_density_data):
        """
        Calculate the entropy contribution to the total energy.

        Parameters
        ----------
        energy_density_data : numpy.array
            Energy density data as numpy array.

        Returns
        -------
        entropy_contribution : float
            S/beta in eV.
        """
        # @SOM: Here the integration has to happen.
        pass
