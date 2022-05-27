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
from math import exp, log, sqrt, pi, ceil, floor, asin
import nashTubeStress as nts
import coolant

strMatNormal = lambda a: [''.join(s).rstrip() for s in a]
strMatTrans  = lambda a: [''.join(s).rstrip() for s in zip(*a)]
sign = lambda x: math.copysign(1.0, x)

class receiver_cyl:
	def __init__(self,coolant = 'salt', Ri = 57.93/2000, Ro = 60.33/2000, T_in = 290, T_out = 565,
                      nz = 450, nt = 46, R_fouling = 0.0, ab = 0.94, em = 0.88, kp = 16.57, H_rec = 10.5, D_rec = 8.5,
                      nbins = 50, alpha = 15.6e-6, Young = 186e9, poisson = 0.31,
                      debugfolder = os.path.expanduser('~'), debug = False, verification = False, Dittus=True):
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
		self.Dittus = Dittus

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
		"""
		   Thermal and transport properties of nitrate salt and sodium
		   Nitrate salt:  Zavoico, A. B. Solar power tower design basis document, revision 0; Topical. Sandia National Labs., 2001.
		   Liquid sodium: Fink, J. K.; Leibowitz, L. Thermodynamic and transport properties of sodium liquid and vapor. Argonne National Lab., 1995.
		"""
		if self.coolant == 'salt':
			d = 2090.0 - 0.636 * (T - 273.15)
		else:
			d = 219.0 + 275.32 * (1.0 - T / 2503.7) + 511.58 * np.sqrt(1.0 - T / 2503.7)
		return d

	def dynamicViscosity(self,T):
		"""
		   Thermal and transport properties of nitrate salt and sodium
		   Nitrate salt:  Zavoico, A. B. Solar power tower design basis document, revision 0; Topical. Sandia National Labs., 2001.
		   Liquid sodium: Fink, J. K.; Leibowitz, L. Thermodynamic and transport properties of sodium liquid and vapor. Argonne National Lab., 1995.
		"""
		if self.coolant == 'salt':
			eta = 0.001 * (22.714 - 0.120 * (T - 273.15) + 2.281e-4 * pow((T - 273.15),2) - 1.474e-7 * pow((T - 273.15),3))
		else:
			eta = np.exp(-6.4406 - 0.3958 * np.log(T) + 556.835/T)
		return eta

	def thermalConductivity(self,T):
		"""
		   Thermal and transport properties of nitrate salt and sodium
		   Nitrate salt:  Zavoico, A. B. Solar power tower design basis document, revision 0; Topical. Sandia National Labs., 2001.
		   Liquid sodium: Fink, J. K.; Leibowitz, L. Thermodynamic and transport properties of sodium liquid and vapor. Argonne National Lab., 1995.
		"""
		if self.coolant == 'salt':
			k = 0.443 + 1.9e-4 * (T - 273.15)
		else:
			k = 124.67 - 0.11381 * T + 5.5226e-5 * pow(T,2) - 1.1842e-8 * pow(T,3);
		return k;

	def specificHeatCapacityCp(self,T):
		"""
		   Thermal and transport properties of nitrate salt and sodium
		   Nitrate salt:  Zavoico, A. B. Solar power tower design basis document, revision 0; Topical. Sandia National Labs., 2001.
		   Liquid sodium: Fink, J. K.; Leibowitz, L. Thermodynamic and transport properties of sodium liquid and vapor. Argonne National Lab., 1995.
		"""
		if self.coolant == 'salt':
			C = 1396.0182 + 0.172 * T
		else:
			C = 1000 * (1.6582 - 8.4790e-4 * T + 4.4541e-7 * pow(T,2) - 2992.6 * pow(T,-2))
		return C

	def Temperature(self, m_flow, Tf, Tamb, CG, h_ext):
		"""
		    Flow and thermal variables:
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
			if self.Dittus:
				Nu = 0.023 * pow(Re, 0.8) * pow(Pr, 0.4)
			else:
				f = pow(1.82*np.log10(Re) - 1.64, -2)
				Nu = (f/8)*(Re - 1000)*Pr/(1 + 12.7*pow(f/8, 0.5)*(pow(Pr,0.66)-1))
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
			BDpp = self.Fourier(To[t,:])
			B0 = (BDpp[0] - BDp[0])/self.ln
			Qnet[t] = 2*np.pi*self.kp*B0*self.dz*np.ones_like(Qnet)

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

def thermal_verification():

	model = receiver_cyl(Ri = 42/2000, Ro = 45/2000, R_fouling=8.808e-5, ab = 0.93, em = 0.87, kp = 21.0, Dittus=False)                   # Instantiating model with the gemasolar geometry
	Tf = model.T_in*np.ones((1,model.nz+1))
	Tf_nts = model.T_in*np.ones((1,model.nz+1))
	m_flow_tb = np.array([4.3])
	CG = np.genfromtxt('fluxInput.csv', delimiter=',').reshape(1,model.nz)*1e6
	Tamb = np.array([298.15])
	h_ext = np.array([0])
	T_o = np.zeros((m_flow_tb.shape[0],model.nz))
	T_i = np.zeros((m_flow_tb.shape[0],model.nz))
	T_i_nts = np.zeros((m_flow_tb.shape[0],model.nz))
	T_o_nts = np.zeros((m_flow_tb.shape[0],model.nz))

	# nashTubeStress objetcs
	g = nts.Grid(nr=30, nt=model.nt, rMin=model.Ri, rMax=model.Ro)
	s = nts.Solver(g, debug=False, CG=CG[0,0], k=model.kp, T_int=model.T_in, R_f=model.R_fouling,
                   A=model.ab, epsilon=model.em, T_ext=Tamb[0], h_ext=h_ext[0],
                   P_i=0e5, alpha=model.l, E=model.E, nu=model.nu, n=1,
                   bend=False)
	s.extBC = s.extTubeHalfCosFluxRadConvAdiabaticBack
	s.intBC = s.intTubeConv

	# Running thermal model
	for k in tqdm(range(model.nz)):
		Qnet = model.Temperature(m_flow_tb, Tf[:,k], Tamb, CG[:,k], h_ext)
		C = model.specificHeatCapacityCp(Tf[:,k])*m_flow_tb
		Tf[:,k+1] = Tf[:,k] + np.divide(Qnet, C, out=np.zeros_like(C), where=C!=0)
		T_o[:,k] = model.To
		T_i[:,k] = model.Ti
		# begin nashTubeStress
		salt = coolant.nitrateSalt(False)
		salt.update(Tf_nts[0,k])
		h_int, dP = coolant.HTC(False, salt, model.Ri, model.Ro, 20, 'Dittus', 'mdot', m_flow_tb[0])
		s.h_int = h_int
		s.CG = CG[0,k]
		ret = s.solve(eps=1e-6)
		s.postProcessing()
		Q = 2*np.pi*model.kp*s.B0*model.dz
		Tf_nts[0,k+1] = Tf_nts[0,k] + Q/(model.specificHeatCapacityCp(Tf_nts[0,k])*m_flow_tb[0])
		T_i_nts[0,k] = s.T[0,0]
		T_o_nts[0,k] = s.T[0,-1]
		# end nashTubeStress

	x = np.linspace(0,model.H_rec*model.nz/model.nbins,model.nz)
	data = np.genfromtxt('verification.csv',delimiter=",", skip_header=1)
	xfd = data[:,6]
	yfd = data[:,7]
	xpd = data[:,4]
	ypd = data[:,5]

	xf = xfd[np.logical_not(np.isnan(xfd))]
	yf = yfd[np.logical_not(np.isnan(yfd))]
	xp = xpd[np.logical_not(np.isnan(xpd))]
	yp = ypd[np.logical_not(np.isnan(ypd))]

	data = {}
	# Sanchez-Gonzalez et al.
	data['xf'] = xf
	data['yf'] = yf
	data['xp'] = xp
	data['yp'] = yp
	# Proposed model data
	data['z'] = x
	data['T_o'] = T_o
	data['T_i'] = T_i
	data['Tf'] = Tf
	# Logie et al.
	data['z_nts'] = x
	data['T_i_nts'] = T_i_nts
	data['T_o_nts'] = T_o_nts
	data['Tf_nts'] = Tf_nts
	scipy.io.savemat('verification_res.mat',data)

def plottingTemperatures():
	data = scipy.io.loadmat('verification_res.mat')

	fig, axes = plt.subplots(1,2, figsize=(18,4))
	axes[0].plot(data['xp'].reshape(-1),data['yp'].reshape(-1),label='Sánchez-González et al.]')
	axes[0].plot(data['z_nts'].reshape(-1),data['T_i_nts'].reshape(-1)-273.15,label='Logie et al.')
	axes[0].plot(data['z'].reshape(-1),data['T_i'].reshape(-1)-273.15,label='Proposed model')
	axes[0].set_xlabel(r'$z$ [m]')
	axes[0].set_ylabel(r'$T_\mathrm{crown}$ [\textdegree C]')
	axes[0].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)

	axes[1].plot(data['xf'].reshape(-1),data['yf'].reshape(-1),label='Sánchez-González et al.')
	axes[1].plot(data['z_nts'].reshape(-1),data['Tf_nts'].reshape(-1)[1:]-273.15,label='Logie et al.')
	axes[1].plot(data['z'].reshape(-1),data['Tf'].reshape(-1)[1:]-273.15,label='Proposed model')
	axes[1].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)
	plt.show()

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--days', nargs=2, type=int, default=[0,1])
	parser.add_argument('--step', type=float, default=900)
	args = parser.parse_args()

	tinit = time.time()
	thermal_verification()
	plottingTemperatures()
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))
