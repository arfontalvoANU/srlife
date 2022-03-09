import numpy as np
import os, sys

sys.path.append('../..')

from srlife import receiver, solverparams, library, thermal, structural, system, damage, managers

def setup_problem(Ro, th, H_rec, Nr, Nt, Nz, times, fluid_temp, h_flux, pressure, T_base, folder=None, days=1):
	# Setup the base receiver
	period = 24.0                                                         # Loading cycle period, hours
	days = days                                                           # Number of cycles represented in the problem 
	panel_stiffness = "disconnect"                                        # Panels are disconnected from one another
	model = receiver.Receiver(period, days, panel_stiffness)              # Instatiating a receiver model

	# Setup each of the two panels
	tube_stiffness = "rigid"
	panel_0 = receiver.Panel(tube_stiffness)

	# Basic receiver geometry (Updated to Gemasolar)
	r_outer = Ro*1000                                                     # Panel tube outer radius (mm)
	thickness = th*1000                                                   # Panel tube thickness (mm)
	height = H_rec*1000                                                   # Panel tube height (mm)

	# Tube discretization
	nr = Nr                                                               # Number of radial elements in the panel tube cross-section
	nt = Nt                                                               # Number of circumferential elements in the panel tube cross-section
	nz = Nz                                                               # Number of axial elements in the panel tube

	# Setup Tube 0 in turn and assign it to the correct panel
	tube_0 = receiver.Tube(r_outer, thickness, height, nr, nt, nz, T0 = T_base)
	tube_0.set_times(times)
	tube_0.set_bc(receiver.ConvectiveBC(r_outer-thickness, height, nz, times, fluid_temp), "inner")
	tube_0.set_bc(receiver.HeatFluxBC(r_outer, height, nt, nz, times, h_flux), "outer")
	tube_0.set_pressure_bc(receiver.PressureBC(times, pressure))

	# Tube 1
	tube_1 = receiver.Tube(r_outer, thickness, height, nr, nt, nz, T0 = T_base)
	tube_1.set_times(times)
	tube_1.set_bc(receiver.ConvectiveBC(r_outer-thickness, height, nz, times, fluid_temp), "inner")
	tube_1.set_bc(receiver.HeatFluxBC(r_outer, height, nt, nz, times, h_flux), "outer")
	tube_1.set_pressure_bc(receiver.PressureBC(times, pressure))

	# Assign to panel 0
	panel_0.add_tube(tube_0, "tube0")
	panel_0.add_tube(tube_1, "tube1")

	# Assign the panels to the receiver
	model.add_panel(panel_0, "panel0")

	# Save the receiver to an HDF5 file
	if folder==None:
		fileName = 'model.hdf5'
	else:
		fileName = '%s/model.hdf5'%folder
	model.save('model.hdf5')

def run_problem(zpos,nz,progress_bar=True,folder=None,nthreads=4):
	# Load the receiver we previously saved
	if folder==None:
		fileName = 'model.hdf5'
	else:
		fileName = '%s/solartherm/examples/model.hdf5'%os.path.expanduser('~')
	model = receiver.Receiver.load(fileName)

	# Choose the material models
	fluid_mat = library.load_fluid("nitratesalt", "base")
	thermal_mat, deformation_mat, damage_mat = library.load_material("A230", "base", "base", "base")

	# Cut down on run time for now by making the tube analyses 1D
	# This is not recommended for actual design evaluation
	for panel in model.panels.values():
		for tube in panel.tubes.values():
			tube.make_2D(tube.h/nz*zpos)

	# Setup some solver parameters
	params = solverparams.ParameterSet()
	params['progress_bars'] = progress_bar         # Print a progress bar to the screen as we solve
	params['nthreads'] = nthreads                  # Solve will run in multithreaded mode, set to number of available cores
	params['system']['atol'] = 1.0e-4              # During the standby very little happens, lower the atol to accept this result

	# Choose the solvers, i.e. how we are going to solve the thermal,
	# single tube, structural system, and damage calculation problems.
	# Right now there is only one option for each
	# Define the thermal solver to use in solving the heat transfer problem
	thermal_solver = thermal.FiniteDifferenceImplicitThermalSolver(params["thermal"])
	# Define the structural solver to use in solving the individual tube problems
	structural_solver = structural.PythonTubeSolver(params["structural"])
	# Define the system solver to use in solving the coupled structural system
	system_solver = system.SpringSystemSolver(params["system"])
	# Damage model to use in calculating life
	damage_model = damage.TimeFractionInteractionDamage(params['damage'])

	# The solution manager
	solver = managers.SolutionManager(model, thermal_solver, thermal_mat, fluid_mat,
		structural_solver, deformation_mat, damage_mat,
		system_solver, damage_model, pset = params)

	# Actually solve for life
	try:
		life = solver.solve_life()
		model.save('model_solved.hdf5')
	except RuntimeError:
		life = np.empty(3)
		life[:] = np.NaN
	print("Best estimate life position %d: %f daily cycles" % (zpos,life[0]))
	result = np.append(life,zpos)
	return result

