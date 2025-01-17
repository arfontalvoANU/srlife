#!/usr/bin/env python3

import sys
sys.path.append('../..')

from srlife import receiver, solverparams, spring, structural, thermal, system, library, managers, damage

def sample_parameters():
  params = solverparams.ParameterSet()

  params["nthreads"] = 3
  params["progress_bars"] = True

  params["thermal"]["rtol"] = 1.0e-6
  params["thermal"]["atol"] = 1.0e-6
  params["thermal"]["miter"] = 20

  params["structural"]["rtol"] = 1.0e-6
  params["structural"]["atol"] = 1.0e-6
  params["structural"]["miter"] = 20

  params["system"]["rtol"] = 1.0e-6
  params["system"]["atol"] = 1.0e-6
  params["system"]["miter"] = 20
  params["system"]["verbose"] = False
  
  # Extrapolate damage based on the last cycle
  params["damage"]["extrapolate"] = "last"
  
  return params

if __name__ == "__main__":
  # Load the receiver datastructure containing the:
  #     Receiver topology
  #     Tube geometry
  #     Thermal boundary conditions
  #     Pressure boundary conditions
  #     Interconnect stiffnesses
  model = receiver.Receiver.load("example-receiver.hdf5")

  # Load some customized solution parameters
  # These are all optional, all the solvers have default values
  # for parameters not provided by the user
  params = sample_parameters()

  # Define the thermal solver to use in solving the heat transfer problem
  thermal_solver = thermal.FiniteDifferenceImplicitThermalSolver(
      params["thermal"])
  # Define the structural solver to use in solving the individual tube problems
  structural_solver = structural.PythonTubeSolver(params["structural"])
  # Define the system solver to use in solving the coupled structural system
  system_solver = system.SpringSystemSolver(params["system"])
  # Damage model to use in calculating life
  damage_model = damage.TimeFractionInteractionDamage(params["damage"])

  # Load the materials
  fluid = library.load_fluid("salt", "base")
  thermal, deformation, damage = library.load_material("740H", "base", 
      "elastic_model", "base")
  
  # The solution manager
  solver = managers.SolutionManager(model, thermal_solver, thermal, fluid,
      structural_solver, deformation, damage, system_solver, damage_model, 
      pset = params)

  # Reset the temperatures each night
  solver.add_heuristic(managers.CycleResetHeuristic())

  # Report the best-estimate life of the receiver 
  life = solver.solve_life()
  
  print("Best estimate life: %f daily cycles" % life)
  
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      tube.write_vtk("tube-%s-%s" % (pi, ti))
