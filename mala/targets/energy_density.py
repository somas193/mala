"""Energy density calculation class."""
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
    """Postprocessing / parsing functions for the energy density.

    Parameters
    ----------
    params : mala.common.parameters.Parameters
        Parameters used to create this TargetBase object.
    """

    te_mutex = False

    def __init__(self, params):
        super(EnergyDensity, self).__init__(params)
        # We operate on a per gridpoint basis. Per gridpoint,
        # there are two components of the energy density:
        # Band energy and Entropy contribution
        self.target_length = 2

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
            raise Exception("Unsupported unit for energy density.")

    def read_from_cube(self, file_name, directory, units=None):
        """
        Read the energy density data from a cube file.

        Parameters
        ----------
        file_name :
            Name of the cube file.

        directory :
            Directory containing the cube file.

        units : string
            Units the energy density is saved in. Usually none.
        """
        printout("Reading density from .cube file in ", directory)
        data, meta = read_cube(directory + file_name)
        return data

    def get_integrated_quantities(self, energy_density_data, grid_spacing_bohr=None,
                             integration_method="summation"):
        """
        Integrate quantities defined per grid point to compute band energy, 
        entropy contribution etc.

        Parameters
        ----------
        energy_density_data : numpy.array
            Energy density data as numpy array. Has to either be of the form
            gridpoints or gridx x gridy x gridz.
        
        grid_spacing_bohr : float
            Grid spacing (in Bohr) used to construct this grid. As of now,
            only equidistant grids are supported.

        integration_method : str
            Integration method used to integrate density on the grid.
            Currently supported:

            - "trapz" for trapezoid method
            - "simps" for Simpson method.
            - "summation" for summation and scaling of the values (recommended)

        Returns
        -------
        entropy_contribution/band energy : float
            
        """

        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr

        # Check input data for correctness.
        data_shape = np.shape(np.squeeze(energy_density_data))
        if len(data_shape) != 3:
            if len(data_shape) != 1:
                raise Exception("Unknown Energy density shape, cannot calculate "
                                "the Entropy contribution.")
            elif integration_method != "summation":
                raise Exception("If using a 1D energy_density array, you can only"
                                " use summation as integration method.")

        # We integrate along the three axis in space.
        # If there is only one point in a certain direction we do not
        # integrate, but rather reduce in this direction.
        # Integration over one point leads to zero.

        physical_quantity = None
        if integration_method != "summation":
            physical_quantity = energy_density_data

            # X
            if data_shape[0] > 1:
                physical_quantity = \
                    integrate_values_on_spacing(physical_quantity,
                                                grid_spacing_bohr, axis=0,
                                                method=integration_method)
            else:
                physical_quantity =\
                    np.reshape(physical_quantity, (data_shape[1],
                                                     data_shape[2]))
                physical_quantity *= grid_spacing_bohr

            # Y
            if data_shape[1] > 1:
                physical_quantity = \
                    integrate_values_on_spacing(physical_quantity,
                                                grid_spacing_bohr, axis=0,
                                                method=integration_method)
            else:
                physical_quantity = \
                    np.reshape(physical_quantity, (data_shape[2]))
                physical_quantity *= grid_spacing_bohr

            # Z
            if data_shape[2] > 1:
                physical_quantity = \
                    integrate_values_on_spacing(physical_quantity,
                                                grid_spacing_bohr, axis=0,
                                                method=integration_method)
            else:
                physical_quantity *= grid_spacing_bohr
        else:
            if len(data_shape) == 3:
                physical_quantity = np.sum(energy_density_data, axis=(0, 1, 2)) \
                                      * (grid_spacing_bohr ** 3)
            if len(data_shape) == 1:
                physical_quantity = np.sum(energy_density_data, axis=0) * \
                                      (grid_spacing_bohr ** 3)

        return physical_quantity