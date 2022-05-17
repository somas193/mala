from mala.targets.calculation_helpers import get_beta
import torch
import mala

path_in = "/home/rofl/MALA/test-data/Be2/densities_gp/qe_inputs/snapshot4/"
path_out = "/home/rofl/MALA/test-data/Be2/densities_gp/additional_info_qeouts/snapshot4.out"
params = mala.Parameters()
ec_calculator = mala.targets.EnergyDensity(params)
ec_calculator.read_additional_calculation_data('qe.out', path_out)
snap0_data = ec_calculator.read_from_cube('Be_snapshot4_entropy.cube', path_in)
print(snap0_data)

entropy = ec_calculator.get_entropy_contribution(snap0_data) #/ get_beta(ec_calculator.temperature_K)
print(entropy)
