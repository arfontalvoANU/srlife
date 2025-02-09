#!/usr/bin/env python3

import sys
sys.path.append('../..')

from srlife import receiver, solverparams, spring, structural, thermal, system, library, managers, damage

def sample_parameters():
  params = solverparams.ParameterSet()

  params["nthreads"] = 1
  params["progress_bars"] = True
  # If true store results on disk (slower, but less memory)
  params["page_results"] = False

  params["thermal"]["rtol"] = 1.0e-6
  params["thermal"]["atol"] = 1.0e-4
  params["thermal"]["miter"] = 20

  params["structural"]["rtol"] = 1.0e-6
  params["structural"]["atol"] = 1.0e-8
  params["structural"]["miter"] = 50
  params["structural"]["verbose"] = False

  params["system"]["rtol"] = 1.0e-6
  params["system"]["atol"] = 1.0e-8
  params["system"]["miter"] = 10
  params["system"]["verbose"] = False
  
  # How to extrapolate damage forward in time based on the cycles provided
  # Options:
  #     "lump" = D_future = sum(D_simulated) / N * days
  #     "last" = D_future = sum(D_simulated[:-1]) + D_simulated[-1] * days
  #     "poly" = polynomial extrapolation with order given by the "order" param
  params["damage"]["extrapolate"] = "lump"
  params["damage"]["order"] = 2
  
  return params

if __name__ == "__main__":
  # Load the receiver datastructure containing the:
  #     Receiver topology
  #     Tube geometry
  #     Thermal boundary conditions
  #     Pressure boundary conditions
  #     Interconnect stiffnesses
  model = receiver.Receiver.load("example-small.hdf5")

  # Cut down on run time for now
  for panel in model.panels.values():
    for tube in panel.tubes.values():
      tube.make_1D(tube.h/2, 0.0)

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

  # Heuristics
  solver.add_heuristic(managers.CycleResetHeuristic())

  # Report the best-estimate life of the receiver 
  life = solver.solve_life()
  
  print("Best estimate life: %f daily cycles" % life)
  
  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      tube.write_vtk("tube-%s-%s" % (pi, ti))
