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


class Density(TargetBase):
    """Postprocessing / parsing functions for the electronic density."""

    te_mutex = False

    def __init__(self, params):
        """
        Create a Density object.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this TargetBase object.

        """
        super(Density, self).__init__(params)
        # We operate on a per gridpoint basis. Per gridpoint,
        # there is one value for the density (spin-unpolarized calculations).
        self.target_length = 1

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

    def get_number_of_electrons(self, density_data, grid_spacing_bohr=None,
                                integration_method="summation"):
        """
        Calculate the number of electrons from given density data.

        Parameters
        ----------
        density_data : numpy.array
            Electronic density on the given grid. Has to either be of the form
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
        """
        if grid_spacing_bohr is None:
            grid_spacing_bohr = self.grid_spacing_Bohr

        # Check input data for correctness.
        data_shape = np.shape(density_data)
        if len(data_shape) != 3:
            if len(data_shape) != 1:
                raise Exception("Unknown Density shape, cannot calculate "
                                "number of electrons.")
            elif integration_method != "summation":
                raise Exception("If using a 1D density array, you can only"
                                " use summation as integration method.")

        # We integrate along the three axis in space.
        # If there is only one point in a certain direction we do not
        # integrate, but rather reduce in this direction.
        # Integration over one point leads to zero.

        number_of_electrons = None
        if integration_method != "summation":
            number_of_electrons = density_data

            # X
            if data_shape[0] > 1:
                number_of_electrons = \
                    integrate_values_on_spacing(number_of_electrons,
                                                grid_spacing_bohr, axis=0,
                                                method=integration_method)
            else:
                number_of_electrons =\
                    np.reshape(number_of_electrons, (data_shape[1],
                                                     data_shape[2]))
                number_of_electrons *= grid_spacing_bohr

            # Y
            if data_shape[1] > 1:
                number_of_electrons = \
                    integrate_values_on_spacing(number_of_electrons,
                                                grid_spacing_bohr, axis=0,
                                                method=integration_method)
            else:
                number_of_electrons = \
                    np.reshape(number_of_electrons, (data_shape[2]))
                number_of_electrons *= grid_spacing_bohr

            # Z
            if data_shape[2] > 1:
                number_of_electrons = \
                    integrate_values_on_spacing(number_of_electrons,
                                                grid_spacing_bohr, axis=0,
                                                method=integration_method)
            else:
                number_of_electrons *= grid_spacing_bohr
        else:
            if len(data_shape) == 3:
                number_of_electrons = np.sum(density_data, axis=(0, 1, 2)) \
                                      * (grid_spacing_bohr ** 3)
            if len(data_shape) == 1:
                number_of_electrons = np.sum(density_data, axis=0) * \
                                      (grid_spacing_bohr ** 3)

        return number_of_electrons

    def get_density(self, density_data, convert_to_threedimensional=False,
                    grid_dimensions=None):
        """
        Get the electronic density, based on density data.

        This function only does reshaping, no calculations.

        Parameters
        ----------
        density_data : numpy.array
            Electronic density data, this array will be returned unchanged
            depending on the other parameters.

        convert_to_threedimensional : bool
            If True, then a density saved as a 1D array will be converted to
            a 3D array (gridsize -> gridx * gridy * gridz)

        grid_dimensions : list
            Provide a list of dimensions to be used in the transformation
            1D -> 3D. If None, MALA will attempt to use the values read with
            TargetBase.read_additional_read_additional_calculation_data .
            If that cannot be done, this function will raise an exception.

        Returns
        -------
        density_data : numpy.array
            Electronic density data in the desired shape.
        """
        if len(density_data.shape) == 3:
            return density_data
        elif len(density_data.shape) == 1:
            if convert_to_threedimensional:
                if grid_dimensions is None:
                    grid_dimensions = self.grid_dimensions
                return density_data.reshape(grid_dimensions)
            else:
                return density_data
        else:
            raise Exception("Unknown density data shape.")

    def get_energy_contributions(self, density_data, create_file=True,
                                 atoms_Angstrom=None, qe_input_data=None,
                                 qe_pseudopotentials=None):
        """
        Extract density based energy contributions from Quantum Espresso.

        Done via a Fortran module accesible through python using f2py.
        Returns: e_rho_times_v_hxc, e_hartree,  e_xc, e_ewald

        Parameters
        ----------
        density_data : numpy.array
            Density data on a grid.

        create_file : bool
            If False, the last mala.pw.scf.in file will be used as input for
            Quantum Espresso. If True (recommended), MALA will create this
            file according to calculation parameters.

        atoms_Angstrom : ase.Atoms
            ASE atoms object for the current system. If None, MALA will
            create one.

        qe_input_data : dict
            Quantum Espresso parameters dictionary for the ASE<->QE interface.
            If None (recommended), MALA will create one.

        qe_pseudopotentials : dict
            Quantum Espresso pseudopotential dictionaty for the ASE<->QE
            interface. If None (recommended), MALA will create one.

        Returns
        -------
        energies : list
            A list containing, in order, the following energy contributions:
                - n*V_xc
                - E_Hartree
                - E_xc
                - e_Ewald
        """
        # noinspection PyShadowingNames
        import total_energy as te
        if create_file:
            # If not otherwise specified, use values as read in.
            if qe_input_data is None:
                qe_input_data = self.qe_input_data
            if qe_pseudopotentials is None:
                qe_pseudopotentials = self.qe_pseudopotentials
            if atoms_Angstrom is None:
                atoms_Angstrom = self.atoms
            ase.io.write("mala.pw.scf.in", atoms_Angstrom, "espresso-in",
                         input_data=qe_input_data,
                         pseudopotentials=qe_pseudopotentials)

        # initialize the total energy module.
        # FIXME: So far, the total energy module can only be initialized once.
        # This is ok when the only thing changing
        # are the atomic positions. But no other parameter can currently be
        # changed in between runs...
        # There should be some kind of de-initialization function that allows
        # for this.

        if Density.te_mutex is False:
            printout("MALA: Starting QuantumEspresso to get density-based"
                     " energy contributions.")
            te.initialize()
            Density.te_mutex = True
            printout("MALA: QuantumEspresso setup done.")
        else:
            printout("MALA: QuantumEspresso is already running. Except for"
                     " the atomic positions, no new parameters will be used.")

        # Before we proceed, some sanity checks are necessary.
        # Is the calculation spinpolarized?
        nr_spin_channels = te.get_nspin()
        if nr_spin_channels != 1:
            raise Exception("Spin polarization is not yet implemented.")

        # If we got values through the ASE parser - is everything consistent?
        number_of_atoms = te.get_nat()
        if number_of_atoms != atoms_Angstrom.get_global_number_of_atoms():
            raise Exception("Number of atoms is inconsistent between MALA "
                            "and Quantum Espresso.")

        # We need to find out if the grid dimensions are consistent.
        # That depends on the form of the density data we received.
        number_of_gridpoints = te.get_nnr()
        if len(density_data.shape) == 3:
            number_of_gridpoints_mala = density_data.shape[0] * \
                                        density_data.shape[1] * \
                                        density_data.shape[2]
        elif len(density_data.shape) == 1:
            number_of_gridpoints_mala = density_data.shape[0]
        else:
            raise Exception("Density data has wrong dimensions. ")
        if number_of_gridpoints_mala != number_of_gridpoints:
            raise Exception("Grid is inconsistent between MALA and"
                            " Quantum Espresso")

        # Now we need to reshape the density.
        density_for_qe = None
        if len(density_data.shape) == 3:
            density_for_qe = np.reshape(density_data, [number_of_gridpoints,
                                                       1], order='F')
        elif len(density_data.shape) == 1:
            warnings.warn("Using 1D density to calculate the total energy"
                          " requires reshaping of this data. "
                          "This is unproblematic, as long as you provided t"
                          "he correct grid_dimensions.")
            density_for_qe = self.get_density(density_data,
                                              convert_to_threedimensional=True)
            density_for_qe = np.reshape(density_for_qe, [number_of_gridpoints,
                                                         1], order='F')

        # Reset the positions. For some reason creating the positions
        # directly from ASE (see above) sometimes
        # causes slight errors. This is more accurate.
        te.set_positions(np.transpose(atoms_Angstrom.get_scaled_positions()),
                         number_of_atoms)

        # Now we can set the new density.
        te.set_rho_of_r(density_for_qe, number_of_gridpoints, nr_spin_channels)

        # Get and return the energies.
        energies = np.array(te.get_energies())*Rydberg
        return energies

    @classmethod
    def from_ldos(cls, ldos_object):
        """
        Create a density object from an LDOS object.

        Parameters
        ----------
        ldos_object : mala.targets.ldos.LDOS
            LDOS object used as input.

        Returns
        -------
        dos_object : Density
            Density object created from LDOS object.


        """
        return_density_object = Density(ldos_object.parameters)
        return_density_object.fermi_energy_eV = ldos_object.fermi_energy_eV
        return_density_object.temperature_K = ldos_object.temperature_K
        return_density_object.grid_spacing_Bohr = ldos_object.grid_spacing_Bohr
        return_density_object.number_of_electrons = ldos_object.\
            number_of_electrons
        return_density_object.band_energy_dft_calculation = ldos_object.\
            band_energy_dft_calculation
        return_density_object.grid_dimensions = ldos_object.grid_dimensions
        return_density_object.atoms = ldos_object.atoms
        return_density_object.qe_input_data = ldos_object.qe_input_data
        return_density_object.qe_pseudopotentials = ldos_object.\
            qe_pseudopotentials
        return_density_object.total_energy_dft_calculation = \
            ldos_object.total_energy_dft_calculation

        return return_density_object
