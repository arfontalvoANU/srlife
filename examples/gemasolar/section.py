import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Times'
import os, sys, math, scipy.io, argparse
from scipy.interpolate import interp1d, RegularGridInterpolator
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
		net_zero = np.where(Qnet<0)[0]
		Qnet[net_zero] = 0.0
		_qnet = np.concatenate((qnet[:,1:],qnet[:,::-1]),axis=1)
		_qnet[net_zero,:] = 0.0
		self.qnet = _qnet

		for t in range(Ti.shape[0]):
			BDp = self.Fourier(Ti[t,:])

		# Fourier coefficients
		self.s, self.e = self.stress(Ti,To)
		return Qnet

	def Fourier(self,T):
		coefs = np.empty(21)
		self.fun(self.nt, self.dt, T, coefs)
		return coefs

	def stress(self, Ti, To):
		stress = np.zeros(Ti.shape[0])
		strain = np.zeros(Ti.shape[0])
		for t in range(Ti.shape[0]):
			BDp = self.Fourier(Ti[t,:])
			BDpp = self.Fourier(To[t,:])
			stress[t], strain[t] = self.Thermoelastic(To[t,0], self.Ro, 0., BDp, BDpp)
		stress = np.array(stress)
		strain = np.array(strain)
		return stress,strain

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

def run_problem(zpos,nz,progress_bar=True,folder=None,nthreads=4,load_state0=False,savestate=False,savefolder='.',loadfolder='.'):
	# Load the receiver we previously saved
	if folder==None:
		fileName = 'model.hdf5'
	else:
		fileName = '%s/solartherm/examples/model.hdf5'%os.path.expanduser('~')
	model = receiver.Receiver.load(fileName)

	# Choose the material models
	fluid_mat = library.load_fluid("nitratesalt", "base")
	mat =     "A230"
	thermat = "base"                               # base
	defomat = "const_elastic_creep"                # base | elastic_creep | elastic_model | const_elastic_creep
	damat =   "base"                               # base
	thermal_mat, deformation_mat, damage_mat = library.load_material(mat, thermat, defomat, damat)

	# Cut down on run time for now by making the tube analyses 1D
	# This is not recommended for actual design evaluation
	for panel in model.panels.values():
		for tube in panel.tubes.values():
			tube.make_2D(tube.h/nz*zpos)
			tube.savefolder=savefolder
			tube.loadfolder=loadfolder
			tube.load_state0=load_state0
			tube.savestate=savestate

	# Setup some solver parameters
	params = solverparams.ParameterSet()
	params['progress_bars'] = progress_bar         # Print a progress bar to the screen as we solve
	params['nthreads'] = nthreads                  # Solve will run in multithreaded mode, set to number of available cores

	params["thermal"]["steady"] = False            # Ignore thermal mass and use conduction only
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

	# Actually solve for life
	solver.solve_heat_transfer()
	solver.solve_structural()
	result = 1
	return result

def run_gemasolar(panel,position,days,nthreads,clearSky,load_state0,savestate):

	print('Verification of inputs:')
	print('panel %s, pos %s, days %s-%s, nthreads=%s, clearSky=%s, load_state0=%s, savestate=%s'%(panel,position,days[0],days[1],nthreads,clearSky,load_state0,savestate))
	model = receiver_cyl(Ri = 20.0/2000, Ro = 22.4/2000)   # Instantiating model with the gemasolar geometry
	nr = 9                                                 # Number of radial nodes

	# Importing data from Modelica
	if clearSky:
		fileName = 'GemasolarSystemOperationCS_res.mat'
		case = 'clear'
	else:
		fileName = 'GemasolarSystemOperation_res.mat'
		case = 'tmy'
	model.import_mat(fileName)

	# Importing times
	times = model.data[:,0]
	# Importing flux
	CG = model.data[:,model._vars['heliostatField.CG[1]'][2]:model._vars['heliostatField.CG[450]'][2]+1]
	m_flow_tb = model.data[:,model._vars['heliostatField.m_flow_tb'][2]]
	Tamb = model.data[:,model._vars['receiver.Tamb'][2]]
	h_ext = model.data[:,model._vars['receiver.h_conv'][2]]
	ele = model.data[:,model._vars['heliostatField.ele'][2]]
	on_forecast = model.data[:,model._vars['heliostatField.on_hf_forecast'][2]]

	# Filtering times
	index = []
	_times = []
	time_lb = days[0]*86400
	time_ub = days[1]*86400
	for i in range(len(times)):
		if ele[i]==0:
			if times[i]%7200.==0 and times[i] not in _times and time_lb<=times[i] and times[i]<=time_ub:
				index.append(i)
				_times.append(times[i])
		else:
			if times[i]%1800.==0 and times[i] not in _times and time_lb<=times[i] and times[i]<=time_ub:
				index.append(i)
				_times.append(times[i])

	# Getting inputs based on filtered times
	times = times[index]/3600.
	times = times.flatten()
	CG = CG[index,:]
	m_flow_tb = m_flow_tb[index]
	Tamb = Tamb[index]
	h_ext = h_ext[index]
	ele = ele[index]
	on_forecast = on_forecast[index]

	tm = np.mod(times,24)
	inds = np.where(tm==0)[0]

	for i in range(len(inds)-1):
		if not np.any(on_forecast[inds[i]:inds[i+1]]==1):
			m_flow_tb[inds[i]:inds[i+1]]=0
			CG[inds[i]:inds[i+1],:]=0

	# Instantiating variables
	field_off = [0]; start = []; stop = []
	for i in range(1,times.shape[0]-1):
		if m_flow_tb[i]==0 and m_flow_tb[i+1]==0 and m_flow_tb[i-1]==0:
			field_off.append(i)
		if m_flow_tb[i]==0 and m_flow_tb[i+1]>0 and m_flow_tb[i-1]==0:
			start.append(i)
		if m_flow_tb[i]==0 and m_flow_tb[i+1]==0 and m_flow_tb[i-1]>0:
			stop.append(i)
	field_off.append(times.shape[0]-1)
	stress = np.zeros((times.shape[0],model.nz))
	Tf = model.T_in*np.ones((times.shape[0],model.nz+1))
	for i in field_off:
		Tf[i,:] = 293.15*np.ones((model.nz+1,))
	for i in start:
		Tf[i,:] = 533.15*np.ones((model.nz+1,))
	for i in stop:
		Tf[i,:] = 533.15*np.ones((model.nz+1,))
	qnet = np.zeros((times.shape[0],2*model.nt-1,model.nz))

	# Running thermal model
	for k in tqdm(range(model.nz)):
		Qnet = model.Temperature(m_flow_tb, Tf[:,k], Tamb, CG[:,k], h_ext)
		C = model.specificHeatCapacityCp(Tf[:,k])*m_flow_tb
		Tf[:,k+1] = Tf[:,k] + np.divide(Qnet, C, out=np.zeros_like(C), where=C!=0)
		stress[:,k] = model.s/1e6
		qnet[:,:,k] = model.qnet/1e6

	# Getting internal pressure
	pressure = np.where(m_flow_tb>0, 0.1, m_flow_tb)
	pressure = pressure.flatten()
	lb = model.nbins*(panel-1)
	ub = lb + model.nbins

	ndays = (days[1]-days[0])

	loadfolder = os.path.join(os.getcwd(),'results_%s_d%s'%(case,days[0]))
	savefolder = os.path.join(os.getcwd(),'results_%s_d%s'%(case,days[1]))
	if not os.path.isdir(savefolder):
		os.mkdir(savefolder)

	Ro = model.Ro
	thickness = model.thickness
	H_rec = model.H_rec
	nt = 2*model.nt-1
	nbins = model.nbins

	# Creating the hdf5 model
	setup_problem(Ro,
	              thickness,
	              H_rec,
	              nr,
	              nt,
	              nbins,
	              times,
	              Tf[:,lb:ub],
	              qnet[:,:,lb:ub],
	              pressure,
	              T_base = 293.15,
	              days=ndays)

	# Running srlife
	if days[0]>0:
		load_state0 = True
	life = run_problem(
	              position,
	              nbins,
	              nthreads=nthreads,
	              load_state0=load_state0,
	              savestate=True,
	              loadfolder=loadfolder,
	              savefolder=savefolder)

	scipy.io.savemat('%s/inputs.mat'%savefolder,{
	              'times':times,
	              'qnet':qnet,
	              'Tf':Tf,
	              'pressure':pressure})

	# Plotting thermal results
	fig, axes = plt.subplots(2,3, figsize=(18,8))

	axes[0,0].plot(times, Tf[:,lb:ub])
	axes[0,0].set_ylabel(r'$T_\mathrm{f}$ [K]')
	axes[0,0].set_xlabel(r'$t$ [h]')

	axes[0,1].plot(times, qnet[:,0,lb:ub])
	axes[0,1].set_ylabel(r'$q^{\prime\prime}_\mathrm{net}$ [MW/m$^2$]')
	axes[0,1].set_xlabel(r'$t$ [h]')

	axes[0,2].plot(times, pressure)
	axes[0,2].set_ylabel(r'$P$ [MPa]')
	axes[0,2].set_xlabel(r'$t$ [h]')

	quadrature_results = scipy.io.loadmat('%s/quadrature_results.mat'%(savefolder))

	vm = np.sqrt((
	              (quadrature_results['stress_xx'] - quadrature_results['stress_yy'])**2.0 + 
	              (quadrature_results['stress_yy'] - quadrature_results['stress_zz'])**2.0 + 
	              (quadrature_results['stress_zz'] - quadrature_results['stress_xx'])**2.0 + 
	              6.0 * (quadrature_results['stress_xy']**2.0 + 
	              quadrature_results['stress_yz']**2.0 + 
	              quadrature_results['stress_xz']**2.0))/2.0)

	em = np.sqrt((
	              (quadrature_results['mechanical_strain_xx'] - quadrature_results['mechanical_strain_yy'])**2.0 + 
	              (quadrature_results['mechanical_strain_yy'] - quadrature_results['mechanical_strain_zz'])**2.0 + 
	              (quadrature_results['mechanical_strain_zz'] - quadrature_results['mechanical_strain_xx'])**2.0 + 
	              6.0 * (quadrature_results['mechanical_strain_xy']**2.0 + 
	              quadrature_results['mechanical_strain_yz']**2.0 + 
	              quadrature_results['mechanical_strain_xz']**2.0))/2.0)

	axes[1,0].plot(times,vm[:,0,0],label='Inner')
	axes[1,0].plot(times,vm[:,727,0],label='Outer')
	axes[1,0].set_xlabel(r'$t$ [h]')
	axes[1,0].set_ylabel(r'$\sigma_\mathrm{crown,eq}$ [MPa]')
	axes[1,0].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)

	axes[1,1].plot(times,quadrature_results['temperature'][:,0,0]-273.15,label='Inner')
	axes[1,1].plot(times,quadrature_results['temperature'][:,727,0]-273.15,label='Outer')
	axes[1,1].set_xlabel(r'$t$ [h]')
	axes[1,1].set_ylabel(r'$T_\mathrm{crown}$ [\textdegree C]')
	axes[1,1].set_ylim([-0.05,700])
	axes[1,1].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)

	axes[1,2].plot(times,em[:,0,0],label='Inner')
	axes[1,2].plot(times,em[:,727,0],label='Outer')
	axes[1,2].set_xlabel(r'$t$ [h]')
	axes[1,2].set_ylabel(r'$\epsilon_\mathrm{crown,eq}$ [mm/mm]')
	axes[1,2].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)

	plt.tight_layout()
	plt.savefig('%s/results_%s_d%s'%(savefolder,case,days[1]))

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--panel', type=int, default=4, help='Panel to be simulated. Default=1')
	parser.add_argument('--position', type=float, default=30, help='Panel position to be simulated. Default=1')
	parser.add_argument('--days', nargs=2, type=int, default=[0,1], help='domain of days to simulate')
	parser.add_argument('--nthreads', type=int, default=4, help='Number of processors. Default=4')
	parser.add_argument('--clearSky', type=bool, default=False, help='Run clear sky DNI (requires to have the solartherm results)')
	parser.add_argument('--load_state0', type=bool, default=False, help='Load state from a previous simulation')
	parser.add_argument('--savestate', type=bool, default=True, help='Save the last state of the last simulated day')
	args = parser.parse_args()

	tinit = time.time()
	run_gemasolar(args.panel,args.position,args.days,args.nthreads,args.clearSky,args.load_state0,args.savestate)
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))
