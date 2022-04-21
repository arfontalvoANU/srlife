import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Times'
import os, sys, math, scipy.io, argparse
import time, ctypes
from numpy.ctypeslib import ndpointer
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

sys.path.append('../..')

from srlife import receiver, solverparams, library, thermal, structural, system, damage, managers

#   Thermal and transport properties of nitrate salt and sodium (verified)
#   Nitrate salt:
#     ZAVOICO, Alexis B. Solar power tower design basis document, revision 0; Topical. Sandia National Labs., 2001.
#   Liquid sodium:
#     FINK, J. K.; LEIBOWITZ, L. Thermodynamic and transport properties of sodium liquid and vapor. Argonne National Lab., 1995.

strMatNormal = lambda a: [''.join(s).rstrip() for s in a]
strMatTrans  = lambda a: [''.join(s).rstrip() for s in zip(*a)]
sign = lambda x: math.copysign(1.0, x)

class receiver_cyl:
	def __init__(self,coolant = 'salt', Ri = 57.93/2000, Ro = 60.33/2000, T_in = 290, T_out = 565,
                      nz = 450, nt = 46, R_fouling = 0.0, ab = 0.94, em = 0.88, kp = 16.57, H_rec = 10.5, D_rec = 8.5,
                      nbins = 50, alpha = 15.6e-6, Young = 186e9, poisson = 0.31,
                      debugfolder = os.path.expanduser('~'), debug = False, verification = False):
		self.coolant = coolant
		self.Ri = Ri
		self.Ro = Ro
		self.thickness = Ro - Ri
		self.T_in = T_in + 273.15
		self.T_out = T_out + 273.15
		self.nz = nz
		self.nt = nt
		self.R_fouling = R_fouling
		self.ab = ab
		self.em = em
		self.kp = kp
		self.H_rec = H_rec
		self.D_rec = D_rec
		self.dz = H_rec/nbins
		self.nbins=nbins
		self.debugfolder = debugfolder
		self.debug = debug
		self.verification = verification
		self.sigma = 5.670374419e-8
		# Discretisation parameters
		self.dt = np.pi/(nt-1)
		# Tube section diameter and area
		self.d = 2.*Ri                               # Tube inner diameter (m)
		self.area = 0.25 * np.pi * pow(self.d,2.)    # Tube flow area (m2)
		self.ln = np.log(Ro/Ri)                      # Log of Ro/Ri simplification
		#Auxiliary variables
		cosines = np.cos(np.linspace(0.0, np.pi, nt))
		self.cosines = np.maximum(cosines, np.zeros(nt))
		self.theta = np.linspace(-np.pi, np.pi,self.nt*2-2)
		self.n = 3
		self.l = alpha
		self.E = Young
		self.nu = poisson
		l = self.E*self.nu/((1+self.nu)*(1-2*self.nu));
		m = 0.5*self.E/(1+self.nu);
		props = l*np.ones((3,3)) + 2*m*np.identity(3)
		self.invprops = np.linalg.inv(props)
		# Loading dynamic library
		so_file = "%s/solartherm/SolarTherm/Resources/Include/stress/stress.so"%os.path.expanduser('~')
		stress = ctypes.CDLL(so_file)
		self.fun = stress.curve_fit
		self.fun.argtypes = [ctypes.c_int,
						ctypes.c_double,
						ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
						ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

	def import_mat(self,fileName):
		mat = scipy.io.loadmat(fileName, chars_as_strings=False)
		names = strMatTrans(mat['name']) # names
		descr = strMatTrans(mat['description']) # descriptions
		self.data = np.transpose(mat['data_2'])

		self._vars = {}
		self._blocks = []
		for i in range(len(names)):
			d = mat['dataInfo'][0][i] # data block
			x = mat['dataInfo'][1][i]
			c = abs(x)-1  # column
			s = sign(x)   # sign
			if c:
				self._vars[names[i]] = (descr[i], d, c, s)
				if not d in self._blocks:
					self._blocks.append(d)
				else:
					absc = (names[i], descr[i])
		del mat

	def density(self,T):
		if self.coolant == 'salt':
			d = 2090.0 - 0.636 * (T - 273.15)
		else:
			d = 219.0 + 275.32 * (1.0 - T / 2503.7) + 511.58 * np.sqrt(1.0 - T / 2503.7)
		return d

	def dynamicViscosity(self,T):
		if self.coolant == 'salt':
			eta = 0.001 * (22.714 - 0.120 * (T - 273.15) + 2.281e-4 * pow((T - 273.15),2) - 1.474e-7 * pow((T - 273.15),3))
		else:
			eta = np.exp(-6.4406 - 0.3958 * np.log(T) + 556.835/T)
		return eta

	def thermalConductivity(self,T):
		if self.coolant == 'salt':
			k = 0.443 + 1.9e-4 * (T - 273.15)
		else:
			k = 124.67 - 0.11381 * T + 5.5226e-5 * pow(T,2) - 1.1842e-8 * pow(T,3);
		return k;

	def specificHeatCapacityCp(self,T):
		if self.coolant == 'salt':
			C = 1396.0182 + 0.172 * T
		else:
			C = 1000 * (1.6582 - 8.4790e-4 * T + 4.4541e-7 * pow(T,2) - 2992.6 * pow(T,-2))
		return C

	def htfs(self, v_flow, Tf):
		# Flow and thermal variables
		"""
			hf: Heat transfer coefficient due to internal forced-convection
			mu: HTF dynamic viscosity (Pa-s)
			kf: HTF thermal conductivity (W/m-K)
			C:  HTF specific heat capacity (J/kg-K)
			Re: HTF Reynolds number
			Pr: HTF Prandtl number
			Nu: Nusselt number due to internal forced convection
		"""

		# HTF thermo-physical properties
		mu = self.dynamicViscosity(Tf)                 # HTF dynamic viscosity (Pa-s)
		kf = self.thermalConductivity(Tf)              # HTF thermal conductivity (W/m-K)
		C = self.specificHeatCapacityCp(Tf)            # HTF specific heat capacity (J/kg-K)
		rho = self.density(Tf)                         # HTF specific heat capacity (J/kg-K)

		# HTF internal flow variables
		Re = rho * v_flow * self.d/ mu                 # HTF Reynolds number
		Pr = mu * C / kf                               # HTF Prandtl number

		if self.coolant == 'salt':
			Nu = 0.023 * pow(Re, 0.8) * pow(Pr, 0.4)
		else:
			Nu = 5.6 + 0.0165 * pow(Re*Pr, 0.85) * pow(Pr, 0.01);

		# HTF internal heat transfer coefficient
		if self.R_fouling==0:
			hf = Nu * kf / self.d
		else:
			hf = Nu * kf / self.d / (1. + Nu * kf / self.d * self.R_fouling)

		return hf

	def Temperature(self, m_flow, Tf, Tamb, CG, h_ext):
		# Flow and thermal variables
		"""
			hf: Heat transfer coefficient due to internal forced-convection
			mu: HTF dynamic viscosity (Pa-s)
			kf: HTF thermal conductivity (W/m-K)
			C:  HTF specific heat capacity (J/kg-K)
			Re: HTF Reynolds number
			Pr: HTF Prandtl number
			Nu: Nusselt number due to internal forced convection
		"""

		Tf,temp = np.meshgrid(np.ones(self.nt),Tf)
		Tf = Tf*temp

		# HTF thermo-physical properties
		mu = self.dynamicViscosity(Tf)                 # HTF dynamic viscosity (Pa-s)
		kf = self.thermalConductivity(Tf)              # HTF thermal conductivity (W/m-K)
		C = self.specificHeatCapacityCp(Tf)            # HTF specific heat capacity (J/kg-K)

		m_flow,temp = np.meshgrid(np.ones(self.nt), m_flow)
		m_flow = m_flow*temp

		Tamb,temp = np.meshgrid(np.ones(self.nt), Tamb)
		Tamb = Tamb*temp

		h_ext,temp = np.meshgrid(np.ones(self.nt), h_ext)
		h_ext = h_ext*temp

		# HTF internal flow variables
		Re = m_flow * self.d / (self.area * mu)    # HTF Reynolds number
		Pr = mu * C / kf                           # HTF Prandtl number

		if self.coolant == 'salt':
			Nu = 0.023 * pow(Re, 0.8) * pow(Pr, 0.4)
		else:
			Nu = 5.6 + 0.0165 * pow(Re*Pr, 0.85) * pow(Pr, 0.01);

		# HTF internal heat transfer coefficient
		if self.R_fouling==0:
			hf = Nu * kf / self.d
		else:
			hf = Nu * kf / self.d / (1. + Nu * kf / self.d * self.R_fouling)

		# Calculating heat flux at circumferential nodes
		cosinesm,fluxes = np.meshgrid(self.cosines,CG)
		qabs = fluxes*cosinesm 
		a = -((self.em*(self.kp + hf*self.ln*self.Ri)*self.Ro*self.sigma)/((self.kp + hf*self.ln*self.Ri)*self.Ro*(self.ab*qabs + self.em*self.sigma*pow(Tamb,4)) + hf*self.kp*self.Ri*Tf + (self.kp + hf*self.ln*self.Ri)*self.Ro*Tamb*(h_ext)))
		b = -((hf*self.kp*self.Ri + (self.kp + hf*self.ln*self.Ri)*self.Ro*(h_ext))/((self.kp + hf*self.ln*self.Ri)*self.Ro*(self.ab*qabs + self.em*self.sigma*pow(Tamb,4)) + hf*self.kp*self.Ri*Tf + (self.kp + hf*self.ln*self.Ri)*self.Ro*Tamb*(h_ext)))
		c1 = 9.*a*pow(b,2.) + np.sqrt(3.)*np.sqrt(-256.*pow(a,3.) + 27.*pow(a,2)*pow(b,4))
		c2 = (4.*pow(2./3.,1./3.))/pow(c1,1./3.) + pow(c1,1./3.)/(pow(2.,1./3.)*pow(3.,2./3.)*a)
		To = -0.5*np.sqrt(c2) + 0.5*np.sqrt((2.*b)/(a*np.sqrt(c2)) - c2)
		Ti = (To + hf*self.Ri*self.ln/self.kp*Tf)/(1 + hf*self.Ri*self.ln/self.kp)
		qnet = hf*(Ti - Tf)
		Qnet = qnet.sum(axis=1)*self.Ri*self.dt*self.dz
		self.qnet = np.concatenate((qnet[:,1:],qnet[:,::-1]),axis=1)
		self.Ti = np.concatenate((Ti[:,1:],Ti[:,::-1]),axis=1)

		for t in range(Ti.shape[0]):
			BDp = self.Fourier(Ti[t,:])

		# Fourier coefficients
		self.s, self.e, self.Tbar_i, self.Tbar_o, self.BP, self.BPP = self.stress(Ti,To)
		return Qnet

	def Fourier(self,T):
		coefs = np.empty(21)
		self.fun(self.nt, self.dt, T, coefs)
		return coefs

	def stress(self, Ti, To):
		stress = np.zeros(Ti.shape[0])
		strain = np.zeros(Ti.shape[0])
		Tbar_i = np.zeros(Ti.shape[0])
		Tbar_o = np.zeros(Ti.shape[0])
		BP = np.zeros(Ti.shape[0])
		BPP = np.zeros(Ti.shape[0])
		for t in range(Ti.shape[0]):
			BDp = self.Fourier(Ti[t,:])
			BDpp = self.Fourier(To[t,:])
			stress[t], strain[t] = self.Thermoelastic(To[t,0], self.Ro, 0., BDp, BDpp)
			Tbar_i[t] = BDp[0]
			Tbar_o[t] = BDpp[0]
			BP[t] = BDp[0]
			BPP[t] = BDpp[0]
		return stress,strain,Tbar_i,Tbar_o,BP,BPP

	def Thermoelastic(self, T, r, theta, BDp, BDpp):
		Tbar_i = BDp[0]; BP = BDp[1]; DP = BDp[2];
		Tbar_o = BDpp[0]; BPP = BDpp[1]; DPP = BDpp[2];
		a = self.Ri; b = self.Ro; a2 = a*a; b2 = b*b; r2 = r*r; r4 = pow(r,4);

		C = self.l*self.E/(2.*(1. - self.nu));
		D = 1./(2.*(1. + self.nu));
		kappa = (Tbar_i - Tbar_o)/np.log(b/a);
		kappa_theta = r*a*b/(b2 - a2)*((BP*b - BPP*a)/(b2 + a2)*np.cos(theta) + (DP*b - DPP*a)/(b2 + a2)*np.sin(theta));
		kappa_tau   = r*a*b/(b2 - a2)*((BP*b - BPP*a)/(b2 + a2)*np.sin(theta) - (DP*b - DPP*a)/(b2 + a2)*np.cos(theta));

		T_theta = T - ((Tbar_i - Tbar_o) * np.log(b/r)/np.log(b/a)) - Tbar_o;

		Qr = kappa*C*(0 -np.log(b/r) -a2/(b2 - a2)*(1 -b2/r2)*np.log(b/a) ) \
					+ kappa_theta*C*(1 - a2/r2)*(1 - b2/r2);
		Qtheta = kappa*C*(1 -np.log(b/r) -a2/(b2 - a2)*(1 +b2/r2)*np.log(b/a) ) \
					+ kappa_theta*C*(3 -(a2 +b2)/r2 -a2*b2/r4);
		Qz = kappa*self.nu*C*(1 -2*np.log(b/r) -2*a2/(b2 - a2)*np.log(b/a) ) \
					+ kappa_theta*2*self.nu*C*(2 -(a2 + b2)/r2) -self.l*self.E*T_theta;
		Qrtheta = kappa_tau*C*(1 -a2/r2)*(1 -b2/r2);

		Q_Eq = np.sqrt(0.5*(pow(Qr -Qtheta,2) + pow(Qr -Qz,2) + pow(Qz -Qtheta,2)) + 6*pow(Qrtheta,2));
		Q = np.zeros(3)
		Q[0] = Qr; Q[1] = Qtheta; Q[2] = Qz;
		e = np.dot(self.invprops, Q)
		e_Eq = np.sqrt(2.)*D*np.sqrt(pow(e[0] - e[1],2) + pow(e[0] - e[2],2) + pow(e[2] - e[1],2));

		if self.verification:
			print("=============== NPS Sch. 5S 1\" S31609 at 450degC ===============")
			print("Biharmonic coefficients:")
			print("Tbar_i [K]:      749.6892       %4.4f"%Tbar_i)
			print("  B'_1 [K]:      45.1191        %4.4f"%BP)
			print("  D'_1 [K]:      -0.0000        %4.4f"%DP)
			print("Tbar_o [K]:      769.7119       %4.4f"%Tbar_o)
			print(" B''_1 [K]:      79.4518        %4.4f"%BPP)
			print(" D''_1 [K]:      0.0000         %4.4f\n"%DPP)
			print("Stress at outside tube crown:")
			print("Q_r [MPa]:       0.0000         %4.4f"%(Qr/1e6))
			print("Q_rTheta [MPa]:  0.0000         %4.4f"%(Qrtheta/1e6))
			print("Q_theta [MPa]:  -101.0056       %4.4f"%(Qtheta/1e6))
			print("Q_z [MPa]:      -389.5197       %4.4f"%(Qz/1e6))
			print("Q_Eq [MPa]:      350.1201       %4.4f"%(Q_Eq/1e6))

		return Q_Eq,e_Eq

def setup_problem(Ro, th, H_rec, Nr, Nt, Nz, times, fluid_temp, h_flux, pressure, T_base, folder=None, days=1):
	# Setup the base receiver
	period = 24.0                                                         # Loading cycle period, hours
	days = days                                                           # Number of cycles represented in the problem 
	panel_stiffness = "disconnect"                                        # Panels are disconnected from one another
	model = receiver.Receiver(period, days, panel_stiffness)              # Instatiating a receiver model

	# Setup each of the two panels
	tube_stiffness = "disconnect"                                         # Tube stiffness (N/mm), could be also "rigid" or "disconnect"
	panel_0 = receiver.Panel(tube_stiffness)

	# Basic receiver geometry (Updated to Gemasolar)
	r_outer = Ro*1000                                                     # Panel tube outer radius (mm)
	thickness = th*1000                                                   # Panel tube thickness (mm)
	r_inner = r_outer-thickness                                           # Panel tube inner radius (mm)
	height = H_rec*1000                                                   # Panel tube height (mm)

	# Tube discretization
	nr = Nr                                                               # Number of radial elements in the panel tube cross-section
	nt = Nt                                                               # Number of circumferential elements in the panel tube cross-section
	nz = Nz                                                               # Number of axial elements in the panel tube

	# Setup Tube 0 in turn and assign it to the correct panel
	tube_0 = receiver.Tube(r_outer, thickness, height, nr, nt, nz, T0 = T_base)
	tube_0.set_times(times)
	tube_0.set_bc(receiver.FixedTempBC(r_inner, height, nt, nz, times, fluid_temp), "inner")
	tube_0.set_bc(receiver.HeatFluxBC(r_outer, height, nt, nz, times, h_flux), "outer")
	tube_0.set_pressure_bc(receiver.PressureBC(times, pressure))

	# Assign to panel 0
	panel_0.add_tube(tube_0, "tube0")

	# Assign the panels to the receiver
	model.add_panel(panel_0, "panel0")

	# Save the receiver to an HDF5 file
	if folder==None:
		fileName = 'model.hdf5'
	else:
		fileName = '%s/model.hdf5'%folder
	model.save('model.hdf5')

def run_problem(zpos,nz,progress_bar=True,folder=None,nthreads=4,load_state0=False,savestate=False,resfolder='.'):
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
			tube.folder=resfolder
			tube.load_state0=load_state0
			tube.savestate=savestate

	# Setup some solver parameters
	params = solverparams.ParameterSet()
	params['progress_bars'] = progress_bar         # Print a progress bar to the screen as we solve
	params['nthreads'] = nthreads                  # Solve will run in multithreaded mode, set to number of available cores

	params["thermal"]["steady"] = True             # Ignore thermal mass and use conduction only
	params["thermal"]["rtol"] = 1.0e-6             # Iteration relative tolerance
	params["thermal"]["atol"] = 1.0e-4             # Iteration absolute tolerance
	params["thermal"]["miter"] = 20                # Maximum iterations
	params["thermal"]["substep"] = 1               # Divide user-provided time increments into smaller values

	params["structural"]["rtol"] = 1.0e-6          # Relative tolerance for NR iterations
	params["structural"]["atol"] = 1.0e-8          # Absolute tolerance for NR iterations
	params["structural"]["miter"] = 50             # Maximum newton-raphson iterations
	params["structural"]["verbose"] = False        # Verbose solve

	params["system"]["rtol"] = 1.0e-6              # Relative tolerance
	params["system"]["atol"] = 1.0e-8              # Absolute tolerance
	params["system"]["miter"] = 20                 # Number of permissible nonlinear iterations
	params["system"]["verbose"] = False            # Print a lot of debug info

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

	#solver.add_heuristic(managers.CycleResetHeuristic())

	# Actually solve for life
	solver.solve_heat_transfer()
	solver.solve_structural()
	result = 1
	return result

def pre_processing_gemasolar(clearSky):
	model = receiver_cyl(Ri = 20.0/2000, Ro = 22.4/2000)   # Instantiating model with the gemasolar geometry
	nr = 9                                                 # Number of radial nodes

	# Importing data from Modelica
	fileName = 'GemasolarSystemOperation_res.mat'
	model.import_mat(fileName)

	# Importing times
	times = model.data[:,0]
	ele = model.data[:,model._vars['heliostatField.ele'][2]]
	Tamb = model.data[:,model._vars['receiver.Tamb'][2]]
	h_ext = model.data[:,model._vars['receiver.h_conv'][2]]
	if not clearSky:
		CG = model.data[:,model._vars['heliostatField.CG[1]'][2]:model._vars['heliostatField.CG[450]'][2]+1]
		m_flow_tb = model.data[:,model._vars['heliostatField.m_flow_tb'][2]]
	else:
		CG = model.data[:,model._vars['heliostatField.CGCS[1]'][2]:model._vars['heliostatField.CGCS[450]'][2]+1]
		m_flow_tb = model.data[:,model._vars['heliostatField.m_flow_tb_cs'][2]]

	# Filtering times
	index = []
	pre_times = []
	time_lb = 0
	time_ub = 365*86400
	for i,v in enumerate(times):
		if v%1800.==0 and v not in pre_times and time_lb<=v and v<=time_ub:
			index.append(i)
			pre_times.append(v)

	# Getting inputs based on filtered times
	times = times[index]/3600.
	CG = CG[index,:]
	m_flow_tb = m_flow_tb[index]
	Tamb = Tamb[index]
	h_ext = h_ext[index]

	# Getting internal pressure
	p_max = 1.5 # MPa
	pressure = np.where(m_flow_tb>0, p_max, m_flow_tb)
	pressure = pressure.flatten()

	state = 0
	times_new = np.array([0])
	for i,t in enumerate(times):
		if state==0 and (t%24==0  or (t-23)==0 or (t-1)%24==0 or (t-23)%24==0):
			times_new = np.append(times_new, t)
		elif state==0 and m_flow_tb[i]>0:
			state=1
			times_new = np.append(times_new,np.arange(times[i-1],times[i],0.1))
		elif state==1 and m_flow_tb[i]>0:
			state=2
			times_new = np.append(times_new, [times[i-1],times[i]])
		elif state==2 and m_flow_tb[i]>0:
			times_new = np.append(times_new, t)
		elif state==1 and m_flow_tb[i]==0:
			state=3
			times_new = np.append(times_new,np.arange(times[i-1],times[i],0.1))
		elif state==2 and m_flow_tb[i]==0:
			state=3
			times_new = np.append(times_new,np.arange(times[i-1],times[i],0.1))
		elif state==3 and m_flow_tb[i]==0:
			state=0
			times_new = np.append(times_new, t)
		elif state==3 and m_flow_tb[i]>0:
			state=1
			times_new = np.append(times_new,np.arange(times[i-1],times[i],0.1))
	times_refined = np.unique(times_new)
	z_bc = np.linspace(1,model.nz,model.nz)

	interp_mflow = interp1d(times, m_flow_tb, bounds_error=False, fill_value=0)
	fun_mflow = lambda t: interp_mflow(t)

	interp_Tamb = interp1d(times, Tamb, bounds_error=False, fill_value=0)
	fun_Tamb = lambda t: interp_Tamb(t)

	interp_h_ext = interp1d(times, h_ext, bounds_error=False, fill_value=0)
	fun_h_ext = lambda t: interp_h_ext(t)

	interp_pressure = interp1d(times, pressure, bounds_error=False, fill_value=0)
	fun_pressure = lambda t: interp_pressure(t)

	interp_CG = RegularGridInterpolator((times, z_bc), CG, bounds_error=False, fill_value=0)
	fun_CG = lambda t, z: interp_CG((t,z))

	times_refined_2d, z_bc_2d = np.meshgrid(times_refined, z_bc, indexing='ij')

	times = times_refined
	m_flow_tb = fun_mflow(times_refined)
	Tamb = fun_Tamb(times_refined)
	h_ext = fun_h_ext(times_refined)
	pressure = fun_pressure(times_refined)
	CG = fun_CG(times_refined_2d, z_bc_2d)


	# Instantiating variables
	stress = np.zeros((times.shape[0],model.nz))
	Tf = model.T_in*np.ones((times.shape[0],model.nz+1))
	m_flow_zero = np.where(m_flow_tb<1e-4)[0]
	m_flow_tb[m_flow_zero] = 1e-4
	for i in m_flow_zero:
		Tf[i,:] = Tamb[i]*np.ones((model.nz+1,))
	qnet = np.zeros((times.shape[0],2*model.nt-1,model.nz))
	Ti = np.zeros((times.shape[0],2*model.nt-1,model.nz))
	Tbar_i = np.zeros((times.shape[0],model.nz))
	Tbar_o = np.zeros((times.shape[0],model.nz))
	BP = np.zeros((times.shape[0],model.nz))
	BPP = np.zeros((times.shape[0],model.nz))

	# Running thermal model
	for k in tqdm(range(model.nz)):
		Qnet = model.Temperature(m_flow_tb, Tf[:,k], Tamb, CG[:,k], h_ext)
		C = model.specificHeatCapacityCp(Tf[:,k])*m_flow_tb
		Tf[:,k+1] = Tf[:,k] + np.divide(Qnet, C, out=np.zeros_like(C), where=C!=0)
		stress[:,k] = model.s/1e6
		qnet[:,:,k] = model.qnet/1e6
		Ti[:,:,k] = model.Ti
		Tbar_i[:,k] = model.Tbar_i
		Tbar_o[:,k] = model.Tbar_o
		BP[:,k] = model.BP
		BPP[:,k] = model.BPP

	mydict={}
	mydict['Ro'] = model.Ro
	mydict['thickness'] = model.thickness
	mydict['H_rec'] = model.H_rec
	mydict['nr'] = nr
	mydict['nt'] = 2*model.nt-1
	mydict['nbins'] = model.nbins
	mydict['times'] = times
	mydict['Ti'] = Ti
	mydict['qnet'] = qnet
	mydict['pressure'] = pressure
	mydict['Tbase'] = Tamb[0]
	mydict['CG'] = CG
	mydict['m_flow_tb'] = m_flow_tb
	mydict['Tamb'] = Tamb
	mydict['h_ext'] = h_ext

	if clearSky:
		scipy.io.savemat('input_clear_sky.mat',mydict)
	else:
		scipy.io.savemat('input_tmy_data.mat',mydict)

def run_gemasolar(clearSky,days,panel,position,nthreads,load_state0,savestate):
	if clearSky:
		mydict = scipy.io.loadmat('input_clear_sky.mat')
		case = 'clear'
	else:
		mydict = scipy.io.loadmat('input_tmy_data.mat')
		case = 'tmy'

	resfolder = os.path.join(os.getcwd(),'results')
	if not os.path.isdir(resfolder):
		os.mkdir(resfolder)

	Ro = float(mydict['Ro'])
	thickness = float(mydict['thickness'])
	H_rec = float(mydict['H_rec'])
	nr = int(mydict['nr'])
	nt = int(mydict['nt'])
	nbins = int(mydict['nbins'])

	times = mydict['times'].flatten()
	index = np.where((times>=days[0]*24) & (times<=days[1]*24))[0]

	# Selecting panel boundaries (inlet and outlet)
	lb = nbins*(panel-1)
	ub = lb + nbins
	ndays = days[1] - days[0]
	tm = np.mod(times, 24)
	inds = list(np.where(tm == 0)[0])

	# Creating the hdf5 model
	setup_problem(Ro,
	              thickness,
	              H_rec,
	              nr,
	              nt,
	              nbins,
	              times[index],
	              mydict['Ti'][index,:,lb:ub],
	              mydict['qnet'][index,:,lb:ub],
	              mydict['pressure'].flatten()[index],
	              T_base = mydict['Tbase'][0],
	              days=ndays)

	if days[0]>0:
		load_state0 = True
	# Running srlife
	life = run_problem(
	              position,
	              mydict['nbins'],
	              nthreads=nthreads,
	              load_state0=load_state0,
	              savestate=True,
	              resfolder=resfolder)

	folder_old = resfolder
	resfolder = os.path.join(os.getcwd(),'results_%s_d%s'%(case,days[1]))

	shutil.copytree(folder_old, resfolder)

	scipy.io.savemat('%s/inputs.mat'%resfolder,{'times':times[index]})

	# Plotting thermal results
	fig, axes = plt.subplots(1,3, figsize=(18,4))

	axes[0].plot(times[index], mydict['Ti'][index,0,lb:ub])
	axes[0].set_ylabel(r'$T_\mathrm{crown,i}$ [K]')
	axes[0].set_xlabel(r'$t$ [h]')

	axes[1].plot(times[index], mydict['qnet'][index,0,lb:ub])
	axes[1].set_ylabel(r'$q^{\prime\prime}_\mathrm{net}$ [MW/m$^2$]')
	axes[1].set_xlabel(r'$t$ [h]')

	axes[2].plot(times[index], mydict['pressure'].flatten()[index])
	axes[2].set_ylabel(r'$P$ [MPa]')
	axes[2].set_xlabel(r'$t$ [h]')

	plt.tight_layout()
	plt.savefig('%s/qnet_Ti_pressure_%s.png'%(resfolder,case))

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--panel', type=int, default=1, help='Panel to be simulated. Default=1')
	parser.add_argument('--position', type=float, default=1, help='Panel position to be simulated. Default=1')
	parser.add_argument('--days', nargs=2, type=int, default=[0,1], help='domain of days to simulate')
	parser.add_argument('--nthreads', type=int, default=4, help='Number of processors. Default=4')
	parser.add_argument('--clearSky', type=bool, default=False, help='Run clear sky DNI (requires to have the solartherm results)')
	parser.add_argument('--load_state0', type=bool, default=False, help='Load state from a previous simulation')
	parser.add_argument('--savestate', type=bool, default=False, help='Save the last state of the last simulated day')
	parser.add_argument('--preprocess', type=bool, default=False, help='Just preprocess SRLIFE input')
	args = parser.parse_args()

	tinit = time.time()
	if args.preprocess:
		pre_processing_gemasolar(clearSky)
	else:
		run_gemasolar(args.clearSky, args.days, args.panel, args.position, args.nthreads, args.load_state0, args.savestate)
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

