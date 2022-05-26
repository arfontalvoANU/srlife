import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Times'
import os, sys, math, scipy.io, argparse
from scipy.interpolate import interp1d, RegularGridInterpolator
import scipy.optimize as opt
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

def id_cycles(times, period, days):
	"""
		Helper to separate out individual cycles by index

		Parameters:
			times       Tube times
			period      Period of a single cycle
			days        Number of days in simulation
	"""
	tm = np.mod(times, period)
	inds = list(np.where(tm == 0)[0])
	if len(inds) != (days + 1):
		raise ValueError("Tube times not compatible with the receiver number of days and cycle period!")
	return inds


def cycle_fatigue(strains, temperatures, material, nu = 0.5):
	"""
		Calculate fatigue damage for a single cycle

		Parameters:
			strains         single cycle strains
			temperatures    single cycle temperatures
			material        damage model

		Additional parameters:
			nu              effective Poisson's ratio to use
	"""
	pt_temps = np.max(temperatures, axis = 0)

	pt_eranges = np.zeros(pt_temps.shape)

	nt = strains.shape[1]
	for i in range(nt):
		for j in range(nt):
			de = strains[:,j] - strains[:,i]
			eq = np.sqrt(2) / (2*(1+nu)) * np.sqrt(
					(de[0] - de[1])**2 + (de[1]-de[2])**2 + (de[2]-de[0])**2.0
					+ 3.0/2.0 * (de[3]**2.0 + de[4]**2.0 + de[5]**2.0)
					)
			pt_eranges = np.maximum(pt_eranges, eq)

	dmg = np.zeros(pt_eranges.shape)
	# pylint: disable=not-an-iterable
	for ind in np.ndindex(*dmg.shape):
		dmg[ind] = 1.0 / material.cycles_to_fail("nominalFatigue", pt_temps[ind], pt_eranges[ind])

	return dmg


def calculate_max_cycles(Dc, Df, material, rep_min = 1, rep_max = 1e6):
	"""
		Actually calculate the maximum number of repetitions for a single point

		Parameters:
			Dc          creep damage per simulated cycle
			Df          fatigue damage per simulated cycle
			material    damaged material properties
	"""
	if not material.inside_envelope("cfinteraction", Df(rep_min), Dc(rep_min)):
		return 0

	if material.inside_envelope("cfinteraction", Df(rep_max), Dc(rep_max)):
		return np.inf

	return opt.brentq(lambda N: material.inside_envelope("cfinteraction", Df(N), Dc(N)) - 0.5,
			rep_min, rep_max)


def make_extrapolate(D, extrapolate="lump",order=1):
	"""
		Return a damage extrapolation function based on extrapolate
		giving the damage for the nth cycle

		Parameters:
			D:      raw, per cycle damage
	"""
	if extrapolate == "lump":
		return lambda N, D = D: N * np.sum(D) / len(D)
	elif extrapolate == "last":
		def Dfn(N, D = D):
			N = int(N)
			if N < len(D)-1:
				return np.sum(D[:N])
			else:
				return np.sum(D[:-1]) + D[-1] * N

		return Dfn
	elif extrapolate == "poly":
		p = np.polyfit(np.array(list(range(len(D))))+1, D, order)
		return lambda N, p=p: np.polyval(p, N)
	else:
		raise ValueError("Unknown damage extrapolation approach %s!" % extrapolate)

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
		self.sigmaEq, self.epsilon = self.crown_stress(Ti,To)
		self.Ti = Ti[:,0]
		self.To = To[:,0]
		return Qnet

	def Fourier(self,T):
		coefs = np.empty(21)
		self.fun(self.nt, self.dt, T, coefs)
		return coefs

	def crown_stress(self, Ti, To):
		stress = np.zeros((Ti.shape[0],2))
		strain = np.zeros((Ti.shape[0],2,6))
		ntimes = Ti.shape[0]
		for t in range(ntimes):
			BDp = self.Fourier(Ti[t,:])
			BDpp = self.Fourier(To[t,:])
			stress[t,0], strain[t,0,:] = self.Thermoelastic(To[t,0], self.Ro, 0., BDp, BDpp)
			stress[t,1], strain[t,1,:] = self.Thermoelastic(Ti[t,0], self.Ri, 0., BDp, BDpp)
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
		e = np.zeros((6,))
		e[0] = 1/self.E*(Qr - self.nu*(Qtheta + Qz))
		e[1] = 1/self.E*(Qtheta - self.nu*(Qr + Qz))

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

		return Q_Eq,e

def run_heuristics(days,step,gemasolar_600=False):

	model = receiver_cyl(Ri = 20.0/2000, Ro = 22.4/2000)                   # Instantiating model with the gemasolar geometry

	# Importing data from Modelica
	if gemasolar_600:
		model.import_mat('nitrate_salt_600degC.mat')
		# Importing times
		times = model.data[:,0]
		# Importing flux
		CG = model.data[:,model._vars['CG[1]'][2]:model._vars['CG[450]'][2]+1]
		m_flow_tb = model.data[:,model._vars['m_flow_tb'][2]]
		Tamb = model.data[:,model._vars['data.Tdry'][2]]
		h_ext = model.data[:,model._vars['h_conv'][2]]
		ele = model.data[:,model._vars['ele'][2]]
		on_forecast = model.data[:,model._vars['on_hf'][2]]

	else:
		model.import_mat('GemasolarSystemOperation_res.mat')
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
	for i in tqdm(range(len(times))):
		if times[i]%step==0 and times[i] not in _times and time_lb<=times[i] and times[i]<=time_ub:
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
	sigmaEq_o = np.zeros((times.shape[0],model.nz))
	sigmaEq_i = np.zeros((times.shape[0],model.nz))
	epsilon_o = np.zeros((times.shape[0],model.nz,6))
	epsilon_i = np.zeros((times.shape[0],model.nz,6))
	T_o = np.zeros((times.shape[0],model.nz))
	T_i = np.zeros((times.shape[0],model.nz))
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
		sigmaEq_o[:,k] = model.sigmaEq[:,0]/1e6
		sigmaEq_i[:,k] = model.sigmaEq[:,1]/1e6
		epsilon_o[:,k,:] = model.epsilon[:,0,:]
		epsilon_i[:,k,:] = model.epsilon[:,1,:]
		qnet[:,:,k] = model.qnet/1e6
		T_o[:,k] = model.To
		T_i[:,k] = model.Ti

	# Choose the material models
	mat =     "A230"
	thermat = "base"                               # base
	defomat = "const_base"                         # base | elastic_creep | elastic_model | const_elastic_creep | const_base
	damat =   "base"                               # base
	thermal_mat, deformation_mat, damage_mat = library.load_material(mat, thermat, defomat, damat)

	# Time to rupture
	period = 24
	tR = damage_mat.time_to_rupture("averageRupture", T_o, sigmaEq_o)
	dts = np.diff(times)
	time_dmg = dts[:,np.newaxis]/tR[1:]

	# Break out to cycle damage
	inds = id_cycles(times, period, days[1]-days[0])

	# Cycle damage
	Dc = np.array([np.sum(time_dmg[inds[i]:inds[i+1]], axis = 0) for i in range(days[1]-days[0])])

	### Fatigue cycles ###

	# Run through each cycle and ID max strain range and fatigue damage
	strain_names = [0, 1, 2, 3, 4, 5] #[ex,ey,ez,exy,exz,eyz]
	strain_factors = [1.0,1.0,1.0,2.0, 2.0, 2.0]

	Df =  np.array([cycle_fatigue(np.array([ef*epsilon_o[inds[i]:inds[i+1],:,en] for en,ef in zip(strain_names, strain_factors)]),
	                              T_o[inds[i]:inds[i+1]], 
	                              damage_mat) for i in range(days[1]-days[0])])

	### Calculating the number of cycles

	# Defining the number of columns as the number of days
	# This is used to create an array with nrows = nelements x nquad,
	# and ncols = number of days
	nc = days[1] - days[0]
	max_cycles = []

	for c,f in zip(Dc.reshape(nc,-1).T, Df.reshape(nc,-1).T):
		# The damage is extrapolated and the number of cycles is determined
		# There are three extrapolation approaches. Here we use the 'lump' one
		max_cycles.append(calculate_max_cycles(make_extrapolate(c), make_extrapolate(f), damage_mat))

	max_cycles = np.array(max_cycles)

	# Saving
	np.save('qnet.npy',qnet)
	np.save('Tf.npy',Tf)

	data = {}
	data['sigmaEq_o'] = sigmaEq_o
	data['sigmaEq_i'] = sigmaEq_i
	data['epsilon_o'] = epsilon_o
	data['epsilon_i'] = epsilon_i
	data['T_o'] = T_o
	data['T_i'] = T_i
	data['times'] = times
	data['Dc'] = Dc
	data['Df'] = Df
	data['max_cycles'] = max_cycles
	data['min_cycle'] = np.argmin(max_cycles)
	data['max_creep'] = np.argmin(np.cumsum(Dc, axis=0))
	data['max_fatig'] = np.argmin(np.cumsum(Df, axis=0))
	scipy.io.savemat('heuristics_res.mat',data)

	'''fig, axes = plt.subplots(2,2, figsize=(12,8))

	axes[0,0].plot(times,sigmaEq_i[:,180],label='Inner')
	axes[0,0].plot(times,sigmaEq_o[:,180],label='Outer')
	axes[0,0].set_xlabel(r'$t$ [h]')
	axes[0,0].set_ylabel(r'$\sigma_\mathrm{crown,eq}$ [MPa]')
	axes[0,0].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)

	plt.show()'''

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--days', nargs=2, type=int, default=[0,1])
	parser.add_argument('--step', type=float, default=900)
	args = parser.parse_args()

	tinit = time.time()
	run_heuristics(args.days,args.step)
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))
