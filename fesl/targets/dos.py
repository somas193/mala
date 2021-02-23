from .target_base import TargetBase
from .calculation_helpers import *
from scipy import integrate, interpolate
from scipy.optimize import toms748
from ase.units import Rydberg
from fesl.common.parameters import printout


class DOS(TargetBase):
    """Postprocessing / parsing functions for the density of states (DOS)."""

    def __init__(self, params):
        """
        Create a DOS object.

        Parameters
        ----------
        params : fesl.common.parameters.Parameters
            Parameters used to create this TargetBase object.

        """
        super(DOS, self).__init__(params)
        self.target_length = self.parameters.ldos_gridsize

    @staticmethod
    def convert_units(array, in_units="1/eV"):
        """
        Convert the units of an array into the FESL units.

        FESL units for the DOS means 1/eV.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array. Currently supported are:

                 - 1/eV (no conversion, FESL unit)
                 - 1/Ry

        Returns
        -------
        converted_array : numpy.array
            Data in 1/eV.
        """
        if in_units == "1/eV":
            return array
        elif in_units == "1/Ry":
            return array / Rydberg
        else:
            printout(in_units)
            raise Exception("Unsupported unit for LDOS.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of an array from FESL units into desired units.

        FESL units for the DOS means 1/eV.

        Parameters
        ----------
        array : numpy.array
            Data in 1/eV.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.
        """
        if out_units == "1/eV":
            return array
        elif out_units == "1/Ry":
            return array * Rydberg
        else:
            printout(out_units)
            raise Exception("Unsupported unit for LDOS.")

    def read_from_qe_dos_txt(self, file_name, directory):
        """
        Read the DOS from a Quantum Espresso generated file.

        These files do not have a specified file ending, so I will call them
        qe.dos.txt here. QE saves the DOS in 1/eV.

        Parameters
        ----------
        file_name : string
            Name of the file containing the DOS.

        directory : string
            Directory containing the file file_name.

        Returns
        -------
        dos_data:
            DOS data in 1/eV.
        """
        # Create the desired/specified energy grid. We will use this to
        # check whether we have a correct file.

        energy_grid = np.arange(self.parameters.ldos_gridoffset_ev,
                                self.parameters.ldos_gridoffset_ev +
                                self.parameters.ldos_gridsize *
                                self.parameters.ldos_gridspacing_ev,
                                self.parameters.ldos_gridspacing_ev)
        return_dos_values = []

        # Open the file, then iterate through its contents.
        with open(directory+file_name, 'r') as infile:
            lines = infile.readlines()
            i = 0

            for dos_line in lines:
                # The first column contains the energy value.
                if "#" not in dos_line and i < self.parameters.ldos_gridsize:
                    e_val = float(dos_line.split()[0])
                    dosval = float(dos_line.split()[1])
                    if np.abs(e_val-energy_grid[i]) < self.parameters.\
                            ldos_gridspacing_ev*0.98:
                        return_dos_values.append(dosval)
                        i += 1

        return np.array(return_dos_values)

    def get_energy_grid(self):
        """
        Get energy grid.

        Returns
        -------
        e_grid : numpy.array
            Energy grid on which the DOS is defined.
        """
        return np.arange(self.parameters.ldos_gridoffset_ev,
                         self.parameters.ldos_gridoffset_ev +
                         self.parameters.ldos_gridsize *
                         self.parameters.ldos_gridspacing_ev,
                         self.parameters.ldos_gridspacing_ev)


    def get_band_energy(self, dos_data, fermi_energy_eV=None,
                        temperature_K=None, integration_method="analytical",
                        shift_energy_grid=True):
        """
        Calculate the band energy from given DOS data.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid].

        fermi_energy_eV : float
            Fermi energy level in eV.

        temperature_K : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        shift_energy_grid : bool
            When using the analytical integration, one has to shift the energy
            grid by setting this parameter to True. Elsewise keep on False.

        Returns
        -------
        band_energy : float
            Band energy in eV.
        """
        # Parse the parameters.
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K

        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.\
            ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        return self.__band_energy_from_dos(dos_data, emin, emax,
                                           self.parameters.ldos_gridspacing_ev,
                                           fermi_energy_eV, temperature_K,
                                           integration_method,
                                           shift_energy_grid)

    def get_number_of_electrons(self, dos_data, fermi_energy_eV=None,
                                temperature_K=None,
                                integration_method="analytical",
                                shift_energy_grid=True):
        """
        Calculate the number of electrons from given DOS data.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid].

        fermi_energy_eV : float
            Fermi energy level in eV.

        temperature_K : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        shift_energy_grid : bool
            When using the analytical integration, one has to shift the energy
            grid by setting this parameter to True. Elsewise keep on False.

        Returns
        -------
        number_of_electrons : float
            Number of electrons.
        """
        # Parse the parameters.
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.\
            ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        return self.__number_of_electrons_from_dos(dos_data, emin, emax,
                                                   self.parameters.
                                                   ldos_gridspacing_ev,
                                                   fermi_energy_eV,
                                                   temperature_K,
                                                   integration_method,
                                                   shift_energy_grid)

    def get_entropy_contribution(self, dos_data, fermi_energy_eV=None,
                                 temperature_K=None,
                                 integration_method="analytical",
                                 shift_energy_grid=True):
        """
        Calculate the entropy contribution to the total energy.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid].

        fermi_energy_eV : float
            Fermi energy level in eV.

        temperature_K : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        shift_energy_grid : bool
            When using the analytical integration, one has to shift the energy
            grid by setting this parameter to True. Elsewise keep on False.

        Returns
        -------
        entropy_contribution : float
            S/beta in eV.
        """
        if fermi_energy_eV is None:
            fermi_energy_eV = self.fermi_energy_eV
        if temperature_K is None:
            temperature_K = self.temperature_K
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize*self.parameters.\
            ldos_gridspacing_ev+self.parameters.ldos_gridoffset_ev
        return self.\
            __entropy_contribution_from_dos(dos_data, emin, emax,
                                            self.parameters.
                                            ldos_gridspacing_ev,
                                            fermi_energy_eV, temperature_K,
                                            integration_method,
                                            shift_energy_grid)



    def get_self_consistent_fermi_energy_ev(self, dos_data,
                                            temperature_K=None,
                                            integration_method="analytical",
                                            shift_energy_grid=True):
        """
        Calculate the self-consistent Fermi energy.

        "Self-consistent" does not mean self-consistent in the DFT sense,
        but rather the Fermi energy, for which DOS integration reproduces
        the exact number of electrons. The term "self-consistent" stems
        from how this quantity is calculated.

        Parameters
        ----------
        dos_data : numpy.array
            DOS data with dimension [energygrid].

        temperature_K : float
            Temperature in K.

        integration_method : string
            Integration method to be used. Currently supported:

                - "trapz" for trapezoid method
                - "simps" for Simpson method.
                - "analytical" for analytical integration. Recommended.

        shift_energy_grid : bool
            When using the analytical integration, one has to shift the energy
            grid by setting this parameter to True. Elsewise keep on False.

        Returns
        -------
        fermi_energy_self_consistent : float
            E_F in eV.
        """
        # Parse the parameters.
        if temperature_K is None:
            temperature_K = self.temperature_K
        emin = self.parameters.ldos_gridoffset_ev
        emax = self.parameters.ldos_gridsize * self.parameters.\
            ldos_gridspacing_ev + self.parameters.ldos_gridoffset_ev

        fermi_energy_sc = toms748(lambda fermi_sc:
                                  (self.
                                   __number_of_electrons_from_dos
                                   (dos_data, emin, emax,
                                    self.parameters.ldos_gridspacing_ev,
                                    fermi_sc, temperature_K,
                                    integration_method, shift_energy_grid)
                                    - self.number_of_electrons), a=emin,
                                  b=emax)
        return fermi_energy_sc

    def get_density_of_states(self, dos_data):
        """Get the density of states."""
        return dos_data


    @classmethod
    def from_ldos(cls, ldos_object):
        """
        Create a DOS object from an LDOS object.

        Parameters
        ----------
        ldos_object : fesl.targets.ldos.LDOS
            LDOS object used as input.

        Returns
        -------
        dos_object : DOS
            DOS object created from LDOS object.


        """
        return_dos_object = DOS(ldos_object.parameters)
        return_dos_object.fermi_energy_eV = ldos_object.fermi_energy_eV
        return_dos_object.temperature_K = ldos_object.temperature_K
        return_dos_object.grid_spacing_Bohr = ldos_object.grid_spacing_Bohr
        return_dos_object.number_of_electrons = ldos_object.number_of_electrons
        return_dos_object.band_energy_dft_calculation = \
            ldos_object.band_energy_dft_calculation
        return_dos_object.atoms = ldos_object.atoms
        return_dos_object.qe_input_data = ldos_object.qe_input_data
        return_dos_object.qe_pseudopotentials = ldos_object.qe_pseudopotentials
        return_dos_object.total_energy_dft_calculation = \
            ldos_object.total_energy_dft_calculation

        return return_dos_object



    @staticmethod
    def __number_of_electrons_from_dos(dos_data, emin, emax,
                                       energy_grid_spacing, fermi_energy_eV,
                                       temperature_K, integration_method,
                                       shift_energy_grid):
        """
        Calculate the number of electrons from DOS data.

        I don't fully understand why shift_energy_grid is needed yet.
        But it definitely is.
        """
        # Calculate the energy levels and the Fermi function.

        # energy_vals = np.linspace(emin, emax, nr_energy_levels)
        if shift_energy_grid and integration_method == "analytical":
            emin += energy_grid_spacing
            emax += energy_grid_spacing
        energy_vals = np.arange(emin, emax, energy_grid_spacing)
        fermi_vals = fermi_function(energy_vals, fermi_energy_eV, temperature_K)

        # Calculate the number of electrons.
        if integration_method == "trapz":
            number_of_electrons = integrate.trapz(dos_data * fermi_vals,
                                                  energy_vals, axis=-1)
        elif integration_method == "simps":
            number_of_electrons = integrate.simps(dos_data * fermi_vals,
                                                  energy_vals, axis=-1)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_vals, dos_data)
            number_of_electrons, abserr = integrate.quad(
                lambda e: dos_pointer(e) * fermi_function(e, fermi_energy_eV,
                                                          temperature_K),
                energy_vals[0], energy_vals[-1], limit=500,
                points=fermi_energy_eV)
        elif integration_method == "analytical":
            number_of_electrons = analytical_integration(dos_data, "F0", "F1",
                                                         fermi_energy_eV,
                                                         energy_vals,
                                                         temperature_K)
        else:
            raise Exception("Unknown integration method.")

        return number_of_electrons

    @staticmethod
    def __band_energy_from_dos(dos_data, emin, emax, energy_grid_spacing,
                               fermi_energy_eV,
                               temperature_K, integration_method,
                               shift_energy_grid):
        """Calculate the band energy from DOS data."""
        # Calculate the energy levels and the Fermi function.
        if shift_energy_grid and integration_method == "analytical":
            emin += energy_grid_spacing
            emax += energy_grid_spacing
        energy_vals = np.arange(emin, emax, energy_grid_spacing)
        fermi_vals = fermi_function(energy_vals, fermi_energy_eV, temperature_K)

        # Calculate the band energy.
        if integration_method == "trapz":
            band_energy = integrate.trapz(dos_data * (energy_vals *
                                                      fermi_vals),
                                          energy_vals, axis=-1)
        elif integration_method == "simps":
            band_energy = integrate.simps(dos_data * (energy_vals *
                                                      fermi_vals),
                                          energy_vals, axis=-1)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_vals, dos_data)
            band_energy, abserr = integrate.quad(
                lambda e: dos_pointer(e) * e * fermi_function(e,
                                                              fermi_energy_eV,
                                                              temperature_K),
                energy_vals[0], energy_vals[-1], limit=500,
                points=fermi_energy_eV)
        elif integration_method == "analytical":
            number_of_electrons = analytical_integration(dos_data, "F0", "F1",
                                                         fermi_energy_eV,
                                                         energy_vals,
                                                         temperature_K)
            band_energy_minus_uN = analytical_integration(dos_data, "F1", "F2",
                                                          fermi_energy_eV,
                                                          energy_vals,
                                                          temperature_K)
            band_energy = band_energy_minus_uN+fermi_energy_eV *\
                number_of_electrons
        else:
            raise Exception("Unknown integration method.")

        return band_energy

    @staticmethod
    def __entropy_contribution_from_dos(dos_data, emin, emax,
                                        energy_grid_spacing, fermi_energy_eV,
                                        temperature_K, integration_method,
                                        shift_energy_grid):
        r"""
        Calculate the entropy contribution to the total energy from DOS data.

        More specifically, this gives -\beta^-1*S_S
        """
        # Calculate the energy levels and the Fermi function.
        if shift_energy_grid and integration_method == "analytical":
            emin += energy_grid_spacing
            emax += energy_grid_spacing
        energy_vals = np.arange(emin, emax, energy_grid_spacing)
        fermi_vals = fermi_function(energy_vals, fermi_energy_eV,
                                    temperature_K)

        # Calculate the entropy contribution to the energy.
        if integration_method == "trapz":
            multiplicator = entropy_multiplicator(energy_vals,
                                                  fermi_energy_eV,
                                                  temperature_K)
            entropy_contribution = integrate.trapz(dos_data * multiplicator,
                                                   energy_vals, axis=-1)
            entropy_contribution /= get_beta(temperature_K)
        elif integration_method == "simps":
            multiplicator = entropy_multiplicator(energy_vals, fermi_energy_eV,
                                                  temperature_K)
            entropy_contribution = integrate.simps(dos_data * multiplicator,
                                                   energy_vals, axis=-1)
            entropy_contribution /= get_beta(temperature_K)
        elif integration_method == "quad":
            dos_pointer = interpolate.interp1d(energy_vals, dos_data)
            entropy_contribution, abserr = integrate.quad(
                lambda e: dos_pointer(e) *
                          entropy_multiplicator(e, fermi_energy_eV,
                                                temperature_K),
                energy_vals[0], energy_vals[-1], limit=500,
                points=fermi_energy_eV)
            entropy_contribution /= get_beta(temperature_K)
        elif integration_method == "analytical":
            entropy_contribution = analytical_integration(dos_data, "S0", "S1",
                                                          fermi_energy_eV,
                                                          energy_vals,
                                                          temperature_K)
        else:
            raise Exception("Unknown integration method.")

        return entropy_contribution
