"""Density descriptor class."""
import warnings
import ase
import ase.io
from .descriptor_base import DescriptorBase


class Density(DescriptorBase):
    """Class for calculation and parsing of Density descriptors.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters object used to create this object.
    """

    def __init__(self, parameters):
        super(Density, self).__init__(parameters)
        self.in_format_ase = ""

    @staticmethod
    def convert_units(array, in_units="None"):
        """
        Convert the units of a Density descriptor.

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
            raise Exception("Unsupported unit for Density.")

    @staticmethod
    def backconvert_units(array, out_units):
        """
        Convert the units of a Density descriptor.

        Since these do not really have units this function does nothing yet.

        Parameters
        ----------
        array : numpy.array
            Data in MALA units.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.
        """
        if out_units == "None":
            return array
        else:
            raise Exception("Unsupported unit for Density.")