import torch
import mala
import numpy as np

#Pseudopotential path
psp_path = "/home/rofl/MALA/test-data/Be2/Be.pbe-n-rrkjus_psl.1.0.0.UPF"

#LDOS path
data_path = "/home/rofl/MALA/test-data/Be2/training_data/Be_snapshot3.out.npy"

####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################
test_parameters = mala.Parameters()

# Specify the correct LDOS parameters.
test_parameters.targets.target_type = "LDOS"
test_parameters.targets.ldos_gridsize = 11
test_parameters.targets.ldos_gridspacing_ev = 2.5
test_parameters.targets.ldos_gridoffset_ev = -5
# To perform a total energy calculation one also needs to provide
# a pseudopotential(path).
test_parameters.targets.pseudopotential_path = psp_path

####################
# TARGETS
# Create a target calculator to postprocess data.
# Use this calculator to perform various operations.
####################

ldos = mala.TargetInterface(test_parameters)

# Read additional information about the calculation.
# By doing this, the calculator is able to know e.g. the temperature
# at which the calculation took place or the lattice constant used.
ldos.read_additional_calculation_data("qe.out",
                                      "/home/rofl/MALA/test-data/Be2/densities_gp/additional_info_qeouts/snapshot3.out")

ldos_data = np.load(data_path)
print(ldos_data.shape)
print(ldos_data[0,2,3,:])

# Get quantities of interest.
# For better values in the post processing, it is recommended to
# calculate the "self-consistent Fermi energy", i.e. the Fermi energy
# at which the (L)DOS reproduces the exact number of electrons.
# This Fermi energy usually differs from the one outputted by the
# QuantumEspresso calculation, due to numerical reasons. The difference
# is usually very small.
self_consistent_fermi_energy = ldos.\
    get_self_consistent_fermi_energy_ev(ldos_data)
#Compute band energy and entropy contribution
energy_density = ldos.get_energy_density(ldos_data, self_consistent_fermi_energy)
ed_calculator = mala.targets.EnergyDensity(test_parameters)
ed_calculator.read_additional_calculation_data('qe.out', "/home/rofl/MALA/test-data/Be2/densities_gp/additional_info_qeouts/snapshot3.out")
#Band energy and entropy calculation through integration
band_energy_integrated = ed_calculator.get_integrated_quantities(energy_density[:,0])
entropy_integrated = ed_calculator.get_integrated_quantities(energy_density[:,1])
#Band energy and entropy calculation directly from LDOS
band_energy_direct = ldos.get_band_energy(ldos_data, self_consistent_fermi_energy)
entropy_direct = ldos.get_entropy_contribution(ldos_data, self_consistent_fermi_energy)
#Error calculation
be_error = (np.abs(band_energy_direct - band_energy_integrated) / band_energy_direct) * 100
be_dft_error = (np.abs(ldos.band_energy_dft_calculation - band_energy_integrated) / ldos.band_energy_dft_calculation) * 100
ent_error = (np.abs(entropy_direct - entropy_integrated) / entropy_direct) * 100

print("\nBand energy integrated: {}, Band energy direct: {}".format(band_energy_integrated, band_energy_direct))
print("Error in band energy: {:.2e} %".format(be_error))
print("Band energy integrated: {}, Band energy DFT: {}".format(band_energy_integrated, ldos.band_energy_dft_calculation))
print("Error in band energy DFT: {:.3f} %".format(be_dft_error))
print("Entropy integrated: {}, Entropy direct: {}".format(entropy_integrated, entropy_direct))
print("Error in entropy: {:.2e} %\n".format(ent_error))
