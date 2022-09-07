import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'Times'
import os, sys, math, scipy.io, argparse
from scipy import optimize
import time, ctypes, pickle
from numpy.ctypeslib import ndpointer
from tqdm import tqdm
import nashTubeStress as nts
import coolant
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../')
from section import *

from CoolProp.CoolProp import PropsSI

from mdbapy.cal_sun import SunPosition
import colorama
colorama.init()
def yellow(text):
	return colorama.Fore.YELLOW + colorama.Style.BRIGHT + text + colorama.Style.RESET_ALL

def htfExt(height, diameter, pipe_radius, velocity, T_wall, T_amb):
	# External cylinder loss from Siebers and Kraabel (https://www.osti.gov/servlets/purl/6906848):
	g = 9.81 # gravity acceleration

	H = height # receiver height, m
	D = diameter # Receiver diameter, m
	v = velocity # velocity
	T_wall = T_wall # Average wall temperature K+6
	T_amb = T_amb # K
	
	T_m = (T_wall+T_amb)/2.

	rho = PropsSI('D', 'T', T_m, 'P', 101325., 'Air') # density
	mu = PropsSI('V', 'T', T_m, 'P', 101325., 'Air') # dynamic viscosity
	Cp = PropsSI('CP0MASS', 'T', T_m, 'P', 101325., 'Air') # specific enthalpy
	k = PropsSI('L', 'T', T_m, 'P', 101325., 'Air') # thermal conductivity
	B = PropsSI('isobaric_expansion_coefficient', 'T', T_m, 'P', 101325., 'Air') # bulk expansion coefficient

	nu = mu/rho # cinematic viscosity

	Re = rho*v*D/mu
	Pr = mu*Cp/k
	Gr = g*B*(T_wall-T_amb)*H**3./(nu**2.)

	# Forced convection:
	ks_D = pipe_radius/D

	def Nu_f_0(Re):
		return 0.3+0.488*Re**0.5*(1.+(Re/282000.)**0.625)**0.8

	def Nu_f_1(Re):
		return 0.0455*Re**0.81

	def interpolate(xs, ys, x):
		if ys[0] != ys[1]:
			a = (ys[1]-ys[0])/(xs[1]-xs[0])
			b = ys[1]/(a*xs[1])
			y = a*x+b
		else:
			y = ys[0]
		return y

	def Nu_f_interpolated(ks_D, Re):
		ks_D_data = np.r_[0., 75e-5, 300e-5, 900e-5]
		x0, x1 = np.amax(ks_D_data[ks_D_data<=ks_D]), np.amin(ks_D_data[ks_D_data>=ks_D])
		x_inter = [x0, x1]
		Nu_f_inter = []
		for x in x_inter:		
			if x == 0:
				Nu_f_inter.append(Nu_f_0(Re))
			if x == 75e-5:
				if Re<=7e5:
					Nu_f_inter.append(Nu_f_0(Re))
				if 7e5<Re<2.2e7:
					Nu_f_inter.append(2.57e-3*Re*0.98)
				if Re>=2.2e7:
					Nu_f_inter.append(Nu_f_1(Re))
			if x == 300e-5:
				if Re<=1.8e5:
					Nu_f_inter.append(Nu_f_0(Re))
				if 1.8e5<Re<4e6:
					Nu_f_inter.append(0.0135e-3*Re**0.89)
				if Re>=4e6:
					Nu_f_inter.append(Nu_f_1(Re))
			if x == 900e-5:
				if Re<=1e5:
					Nu_f_inter.append(Nu_f_0(Re))
				if Re>1e5:
					Nu_f_inter.append(Nu_f_1(Re))

		Nu_f = interpolate(x_inter, Nu_f_inter, ks_D)
		return Nu_f
	
	h_f = k*Nu_f_interpolated(ks_D, Re)/D

	# Natural convection:
	Nu_n = 0.098*Gr**(1./3.)*(T_wall/T_amb)**-0.14
	h_n = k*Nu_n/H
	#print height, diameter, pipe_radius, velocity, T_wall, T_amb,h_n,h_f
	if h_f>1e-6:
		h = (h_f**3.2+(np.pi/2.*h_n)**3.2)**(1./3.2)
	else:
		h=0.
		
	return h

def kp800H(T):
	return 4.503E+00+2.887E-02*T-2.020E-05*T**2+1.018E-08*T**3

def secant(fun,x0,tol=1e-2, maxiter=100):
	x1 = (1.-1e-5)*x0
	f0 = fun(x0)
	f1 = fun(x1)
	i = 0
	xs = [x0,x1]
	fs = [f0,f1]
	alpha = 1
	while abs(f1) > tol and i < maxiter:
		x2 = float(f1 - f0)/(x1 - x0)
		x = x1 - alpha*float(f1)/x2

		if x<0:
			alpha=alpha/2.
		elif np.isnan(fun(x)):
			alpha=alpha/2.
		else:
			x0 = x1
			x1 = x
			f0 = f1
			f1 = fun(x1)
			alpha = 1

		xs.append(x1)
		fs.append(f1)
		i += 1

	fmin = min(fs)
	imin = fs.index(fmin)
	xmin = xs[imin]
	return xmin

class mdba_results:
	def __init__(self,filename,model):
		fileo = open(filename,'rb')
		data = pickle.load(fileo)
		fileo.close()
		q_net = data['q_net']
		areas = data['areas']
		self.Tset = data['T_out']
		self.m_flow = data['m'][0]/data['n_tubes'][0]
		self.CG = q_net[data['fp'][0]]/areas[data['fp'][0]]
		self.flux_in = data['flux_in'][0]
		self.Tamb = data['T_amb']
		self.h_ext = data['h_conv_ext']
		try:
			self.fl = data['flux_lim']
		except:
			self.fl = 0.
		self.model = model
	def f(self,m):
		Tf = self.model.T_in*np.ones(self.model.nz+1)
		for k in range(self.model.nz):
			Qnet = self.model.Temperature(m, Tf[k], self.Tamb, self.CG[k], self.h_ext)
			C = self.model.specificHeatCapacityCp(Tf[k])*m
			Tf[k+1] = Tf[k] + Qnet/C
		return abs(self.Tset-Tf[self.model.nz])
	def solve(self):
		prev = time.time()
		cons = []
		l = {'type': 'ineq', 'fun': lambda x, lb=0.0000: x - lb}
		u = {'type': 'ineq', 'fun': lambda x, ub=self.m_flow: ub - x}
		cons.append(l)
		cons.append(u)
		res=optimize.minimize(self.f, 0.75*self.m_flow, constraints=cons, method='COBYLA', options={'disp':True})
		secs = time.time() - prev
		mm, ss = divmod(secs, 60)
		hh, mm = divmod(mm, 60)
		print('	Simulation time: {:d}:{:02d}:{:02d}'.format(int(hh), int(mm), int(ss)))
		return res.x[0]

def get_mass_flow(filename):

	# Instantiating receiver model
	model = receiver_cyl(Ri = 42/2000, Ro = 45/2000, R_fouling=8.808e-5,
	                     ab = 0.93, em = 0.87, kp = 18.3, Dittus=False)

	# Importing solar flux
	mdba = mdba_results(filename,model) # filename: flux-table
	mdba.Tset = 873.15
	CG = mdba.flux_in
	h_ext = mdba.h_ext
	Tamb = mdba.Tamb
	# Mass flow rate calculated from MDBA output
	m_flow_tb = secant(mdba.f, mdba.m_flow)
	print(m_flow_tb)
	print(mdba.f(m_flow_tb))

class mflow_adjust:
	def __init__(self, T=565, folder='.', latitude=37.56):
		self.latitude = latitude
		self.folder = folder
		self.T = T

	def get_mflow(self): # OELT and RELT generations

		# Instantiating receiver model
		model = receiver_cyl(Ri = 42/2000, Ro = 45/2000, R_fouling=8.808e-5,
		                     ab = 0.93, em = 0.87, kp = 18.3, Dittus=False)

		months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

		print(yellow('	Getting mass flow rates'))

		N=10  # for lammda, ecliptic longitude
		M=24  # for omega
		F=np.arange((N+1)*(M+1)*7,dtype=float).reshape(N+1,M+1,7) # the OELT, 0-field eff, 1-unavail,2-cosine, 3-reflective,4-sb,5-att,6-spi

		Lammda=np.linspace(-np.pi,np.pi,N+1)
		Omega=np.linspace(-np.pi,np.pi,M+1)
		DNI_ratio=[0.56,0.87,1.00,1.39]
		sun=SunPosition()

		for fp in range(2):
			f = open('%s/mflow_N08811_salt_path%s_600_sa.motab'%(self.folder,fp+1),'w+')
			f.write('#1\n')
			for ratio in range(len(DNI_ratio)):
				d = DNI_ratio[ratio]
				T = np.genfromtxt('%s/Tbool_%s.csv'%(self.folder,d),dtype=bool,delimiter=',')
				print('	DNI ratio',d,'flow path #',fp+1)
				F[:,:,:]=0.
				for n in range(3,8):
					for m in range(int(0.5*M)+1):
						delta = 23.4556*np.sin(Lammda[n])
						theta=sun.zenith(self.latitude, delta, Omega[m]/np.pi*180.)
						phi=sun.azimuth(self.latitude, theta, delta, Omega[m]/np.pi*180.)
						elevation=90.-theta
						if elevation<=8.:
							continue

						try:
							# Importing solar flux
							mdba = mdba_results('%s/flux_table_n%d_m%d_d%s'%(self.folder,n,m,d),model) # filename: flux-table
							mdba.Tset = 873.15
							CG = mdba.flux_in
							h_ext = mdba.h_ext
							Tamb = mdba.Tamb
							# Mass flow rate calculated from MDBA output
							F[n,m,:] = secant(mdba.f, mdba.m_flow)
						except:
							print('No such file or directory: ./gemasolar_annual_N08811_565/flux_table_n%d_m%d_d%s'%(n,m,d))
							print((270.-phi)%360.,elevation)
							continue

					for m in range(int(0.5*M)+1,M+1):
						F[n,m,:]=F[n,M-m,:]

				for n in range(3):
					F[n,:,:]=F[5-n,:,:]

				for n in range(8,11):
					F[n,:,:]=F[15-n,:,:]

				F_output=np.arange((N+2)*(M+2),dtype=float).reshape(N+2,M+2)
				F_output[0,1:]=Omega/np.pi*180.
				F_output[1:,0]=Lammda/np.pi*180.
				F_output[1:,1:]=F[:,:,0]

				f.write('# DNI ratio: %s\n'%(d))
				f.write('double mflow_%s(%d,%d)\n'%(ratio+1, N+2, M+2))
				for i in range(F_output.shape[0]):
					for j in range(F_output.shape[1]):
						if i==0:
							f.write('%.1f '%F_output[i,j])
						elif j==0:
							f.write('%.1f '%F_output[i,j])
						else:
							if T[i-1,j-1]:
								f.write('%s '%F_output[i,j])
							else:
								f.write('0.0 ')
					f.write('\n')
				f.write('\n')
			f.close()

def thermal_verification(mdba_verification,filename):

	# Instantiating receiver model
	model = receiver_cyl(Ri = 42/2000, Ro = 45/2000, R_fouling=8.808e-5, ab = 0.93, em = 0.87, kp = 18.3, Dittus=False)
	if mdba_verification:
		# Importing solar flux from MDBA output using flux limits from Sanchez-Gonzalez et al. (2017): https://doi.org/10.1016/j.solener.2015.12.055
		mdba = mdba_results(filename,model)
		T_set=mdba.Tset
		CG = mdba.flux_in
		h_ext = mdba.h_ext
		fl = mdba.fl
		Tamb = mdba.Tamb
		# Mass flow rate calculated from MDBA output using flux limits from Sanchez-Gonzalez et al. (2017): https://doi.org/10.1016/j.solener.2015.12.055
		m_flow_tb = mdba.m_flow
	else:
		T_set=565+273.15
		# Importing solar flux Sanchez-Gonzalez et al. (2017): https://doi.org/10.1016/j.solener.2015.12.055
		CG = np.genfromtxt('fluxInput.csv', delimiter=',')*1e6
		# Flow and thermal parameters Sanchez-Gonzalez et al. (2017): https://doi.org/10.1016/j.solener.2015.12.055
		m_flow_tb = 4.2
		Tamb = 35+273.15
		h_ext = htfExt(10.5, 8.5, 45./2000., 0., 700.65, Tamb)

	# Instantiating variables
	z = np.linspace(0,model.H_rec*model.nz/model.nbins,model.nz)
	Tf = model.T_in*np.ones(model.nz+1)
	Tf_nts = model.T_in*np.ones(model.nz+1)
	T_o = np.zeros(model.nz)
	T_i = np.zeros(model.nz)
	T_i_nts = np.zeros(model.nz)
	T_o_nts = np.zeros(model.nz)
	s_nts = np.zeros(model.nz)
	s_eq = np.zeros(model.nz)

	# nashTubeStress objetcs
	g = nts.Grid(nr=3, nt=model.nt, rMin=model.Ri, rMax=model.Ro)
	s = nts.Solver(g, debug=False, CG=CG[0], k=model.kp, T_int=model.T_in, R_f=model.R_fouling,
                   A=model.ab, epsilon=model.em, T_ext=Tamb, h_ext=h_ext,
                   P_i=0e5, alpha=model.l, E=model.E, nu=model.nu, n=1,
                   bend=False)
	s.extBC = s.extTubeHalfCosFluxRadConv
	s.intBC = s.intTubeConv

	# Running thermal model
	for k in tqdm(range(model.nz)):
		model.kp=kp800H(Tf[k])
		Qnet = model.Temperature(m_flow_tb, Tf[k], Tamb, CG[k], h_ext)
		C = model.specificHeatCapacityCp(Tf[k])*m_flow_tb
		Tf[k+1] = Tf[k] + Qnet/C
		T_o[k] = model.To
		T_i[k] = model.Ti
		# Reference data Logie et al. (2018): https://doi.org/10.1016/j.solener.2017.12.003
		salt = coolant.nitrateSalt(False)
		salt.update(Tf_nts[k])
		h_int, dP = coolant.HTC(False, salt, model.Ri, model.Ro, model.kp, 'Gnielinski', 'mdot', m_flow_tb)
		s.h_int = h_int
		s.CG = CG[k]
		s.T_int = Tf_nts[k]
		s.k=kp800H(Tf_nts[k])
		ret = s.solve(eps=1e-6)
		s.postProcessing()
		Q = 2*np.pi*s.k*s.B0*model.dz
		Tf_nts[k+1] = Tf_nts[k] + Q/salt.Cp/m_flow_tb
		T_i_nts[k] = s.T[0,0]
		T_o_nts[k] = s.T[0,-1]
		s_nts[k] = s.sigmaEq[0,-1]

		sigmaEq_o = np.sqrt((
		(model.stress[:,:,0] - model.stress[:,:,1])**2.0 + 
		(model.stress[:,:,1] - model.stress[:,:,2])**2.0 + 
		(model.stress[:,:,2] - model.stress[:,:,0])**2.0 + 
		6.0 * (model.stress[:,:,3]**2.0 + model.stress[:,:,4]**2.0 + model.stress[:,:,5]**2.0))/2.0)

		s_eq[k] = sigmaEq_o[0,0]
		# end nashTubeStress
	print('	m_flow: %s	T_out: %.2f	res: %g'%(m_flow_tb,Tf[-1],abs(T_set-Tf[-1])))

	# Reference data Sanchez-Gonzalez et al. (2017): https://doi.org/10.1016/j.solener.2015.12.055
	data = np.genfromtxt('verification.csv',delimiter=",", skip_header=1)
	xfd = data[:,6]
	yfd = data[:,7]
	xpd = data[:,4]
	ypd = data[:,5]

	# Reference data Soo Too et al. (2019): https://doi.org/10.1016/j.applthermaleng.2019.03.086
	xj = data[:,12]
	yfj = data[:,15]
	ypj = data[:,13]

	# Removing nan from data
	xf = xfd[np.logical_not(np.isnan(xfd))]
	yf = yfd[np.logical_not(np.isnan(yfd))]
	xp = xpd[np.logical_not(np.isnan(xpd))]
	yp = ypd[np.logical_not(np.isnan(ypd))]
	xj = xj[np.logical_not(np.isnan(xj))]
	yfj = yfj[np.logical_not(np.isnan(yfj))]
	ypj = ypj[np.logical_not(np.isnan(ypj))]

	# Creating directory
	data = {}

	# Saving reference data
	data['CG'] = CG
	data['xf'] = xf
	data['yf'] = yf
	data['xp'] = xp
	data['yp'] = yp
	data['xj'] = xj
	data['yfj'] = yfj
	data['ypj'] = ypj
	data['z_nts'] = z
	data['T_i_nts'] = T_i_nts
	data['T_o_nts'] = T_o_nts
	data['Tf_nts'] = Tf_nts
	data['s_nts'] = s_nts/1e6
	data['s_eq'] = s_eq/1e6

	# Saving proposed model data
	data['z'] = z
	data['T_o'] = T_o
	data['T_i'] = T_i
	data['Tf'] = Tf
	scipy.io.savemat('verification_res.mat',data)

	if mdba_verification:
		case = '%d'%(T_set-273.15)
		f = open('mdba_flux_%s.csv'%(case),'w+')
		f.write('z_flux_mdba_%s,flux_mdba_%s,limit_mdba_%s,Ti_mdba_%s,Tf_mdba_%s,s_mdba_%s\n'%(case,case,case,case,case,case))
		for i,v in enumerate(z):
			f.write('%s,%s,%s,%s,%s,%s\n'%(v,mdba.CG[i]/1e6,fl[i],T_i[i]-273.15,Tf[i+1]-273.15,s_eq[i]/1e6))
		f.close()

def plottingTemperatures():
	# Importing data
	data = scipy.io.loadmat('verification_res.mat')

	# Creating subplots
	fig, axes = plt.subplots(1,2, figsize=(12,4))

	# Inner surface temperature of front tube
	axes[0].plot(data['xp'].reshape(-1),data['yp'].reshape(-1),label='Sánchez-González et al. (2017)')
	axes[0].plot(data['z_nts'].reshape(-1),data['T_i_nts'].reshape(-1)-273.15,label='Logie et al. (2018)')
	axes[0].plot(data['xj'].reshape(-1),data['ypj'].reshape(-1),label='Soo Too et al. (2019)')
	axes[0].plot(data['z'].reshape(-1),data['T_i'].reshape(-1)-273.15,label='Fontalvo (2022)')
	axes[0].set_xlabel(r'$z$ [m]')
	axes[0].set_ylabel(r'$T_\mathrm{crown}$ [\textdegree C]')
	axes[0].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)

	# Fluid temperature
	axes[1].plot(data['xf'].reshape(-1),data['yf'].reshape(-1),label='Sánchez-González et al. (2017)')
	axes[1].plot(data['z_nts'].reshape(-1),data['Tf_nts'].reshape(-1)[1:]-273.15,label='Logie et al. (2018)')
	axes[1].plot(data['xj'].reshape(-1),data['yfj'].reshape(-1),label='Soo Too et al. (2019)')
	axes[1].plot(data['z'].reshape(-1),data['Tf'].reshape(-1)[1:]-273.15,label='Fontalvo (2022)')
	axes[1].set_xlabel(r'$z$ [m]')
	axes[1].set_ylabel(r'$T_\mathrm{fluid}$ [\textdegree C]')
	axes[1].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)

	plt.tight_layout()
	axes[0].text(0.05,0.9,'(a)', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
	axes[1].text(0.05,0.9,'(b)', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
	plt.savefig('Verification.png',dpi=300)

	# Creating subplots
	fig, axes = plt.subplots(1,1, figsize=(6,4))

	# Stress
	axes.plot(data['z_nts'].reshape(-1),data['s_nts'].reshape(-1),label='Logie et al. (2018)')
	axes.plot(data['z'].reshape(-1),data['s_eq'].reshape(-1),ls='None',marker='o',markersize=2.5,markerfacecolor='#ffffff',label='Fontalvo (2022)')#markeredgewidth=2.5,
	axes.set_xlabel(r'$z$ [m]')
	axes.set_ylabel(r'$\sigma_\mathrm{crown}$ [MPa]')
	axes.legend(loc="best", borderaxespad=0, ncol=1, frameon=False)
	plt.tight_layout()
	plt.savefig('Stress_verification.png',dpi=300)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--mdba_verification', type=bool, default=False)
	parser.add_argument('--flux_filename', type=str, default='flux-table')
	args = parser.parse_args()
	tinit = time.time()
	if args.mdba_verification:
		thermal_verification(args.mdba_verification,args.flux_filename)
	else:
		#get_mass_flow(args.flux_filename)
		model=mflow_adjust(T=565, folder='/home/arfontalvo/ownCloud/phd_update/damage/gemasolar_annual_N08811_565')
		model.get_mflow()
	plottingTemperatures()
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))
