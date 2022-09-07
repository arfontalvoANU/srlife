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

import colorama
colorama.init()
def yellow(text):
	return colorama.Fore.YELLOW + colorama.Style.BRIGHT + text + colorama.Style.RESET_ALL

sys.path.append('../..')

from srlife import receiver, solverparams, library, thermal, structural, system, damage, managers
from neml import uniaxial
from section import *

def print_table_latex(model, data):
	print('')
	print('Cumulative creep and fatigue damage and projected receiver life for a nitrate-salt Gemasolar-like receiver. Material: UNS N08811.')
	for i in range(int(model.nz/model.nbins)):
		lb = model.nbins*i
		ub = lb + model.nbins - 1
		j = np.argmin(data['max_cycles'][lb:ub])
		print('%d & %4.2f & %.1e & %.1e & %4.3f \\\\'%(
			i+1,
			model.H_rec*(j+1)/model.nbins,
			np.cumsum(data['Dc'],axis=0)[-1,lb+j],
			np.cumsum(data['Df'],axis=0)[-1,lb+j],
			data['max_cycles'][lb+j]/365,
		))

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

def run_heuristics(days,step,solartherm_res,mat='800H'):
	# Instantiating receiver model
	if mat =='A230':
		kp = 17.97; alpha = 15.61e-06; Young = 184e9; poisson = 0.31;
	elif mat == '800H':
		kp = 18.30; alpha = 18.28e-06; Young = 171e9; poisson = 0.31;
	model = receiver_cyl(Ri = 42.0/2000, Ro = 45.0/2000, R_fouling=8.808e-5,
	                     alpha = alpha, Young = Young, poisson = poisson,
	                     ab = 0.93, em = 0.87, kp = kp, Dittus=False)

	model.import_mat(solartherm_res)
	# Importing times
	times = model.data[:,0]
	# Importing flux
	CG = model.data[:,model._vars['heliostatField.CG[1]'][2]:model._vars['heliostatField.CG[450]'][2]+1]
	m_flow_tb = model.data[:,model._vars['heliostatField.m_flow_tb'][2]]
	Tamb = model.data[:,model._vars['receiver.Tamb'][2]]
	h_ext = model.data[:,model._vars['receiver.h_conv'][2]]
	ele = model.data[:,model._vars['heliostatField.ele'][2]]

	# Filtering times
	index = []
	_times = []
	time_lb = days[0]*86400
	time_ub = days[1]*86400
	print(yellow('	Sorting times'))
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

	tm = np.mod(times,24)
	inds = np.where(tm==0)[0]

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
	sigma_o = np.zeros((times.shape[0],model.nz,6))
	sigma_i = np.zeros((times.shape[0],model.nz,6))
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
	print(yellow('	Running thermal model'))
	for k in tqdm(range(model.nz)):
		Qnet = model.Temperature(m_flow_tb, Tf[:,k], Tamb, CG[:,k], h_ext)
		C = model.specificHeatCapacityCp(Tf[:,k])*m_flow_tb
		Tf[:,k+1] = Tf[:,k] + np.divide(Qnet, C, out=np.zeros_like(C), where=C!=0)
		sigma_o[:,k,:] = model.stress[:,0,:]/1e6
		sigma_i[:,k,:] = model.stress[:,1,:]/1e6
		epsilon_o[:,k,:] = model.epsilon[:,0,:]
		epsilon_i[:,k,:] = model.epsilon[:,1,:]
		qnet[:,:,k] = model.qnet/1e6
		T_o[:,k] = model.To
		T_i[:,k] = model.Ti

	sigmaEq_o = np.sqrt((
		(sigma_o[:,:,0] - sigma_o[:,:,1])**2.0 + 
		(sigma_o[:,:,1] - sigma_o[:,:,2])**2.0 + 
		(sigma_o[:,:,2] - sigma_o[:,:,0])**2.0 + 
		6.0 * (sigma_o[:,:,3]**2.0 + sigma_o[:,:,4]**2.0 + sigma_o[:,:,5]**2.0))/2.0)

	sigmaEq_i = np.sqrt((
		(sigma_i[:,:,0] - sigma_i[:,:,1])**2.0 + 
		(sigma_i[:,:,1] - sigma_i[:,:,2])**2.0 + 
		(sigma_i[:,:,2] - sigma_i[:,:,0])**2.0 + 
		6.0 * (sigma_i[:,:,3]**2.0 + sigma_i[:,:,4]**2.0 + sigma_i[:,:,5]**2.0))/2.0)

	# Choose the material models
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
	data = {}
	data['sigma_o'] = sigma_o
	data['sigma_i'] = sigma_i
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
	data['nbins'] = model.nbins
	print_table_latex(model,data)
	scipy.io.savemat('heuristics_res.mat',data)

	if (days[1] - days[0])==1:
		# Creating subplots
		fig, axes = plt.subplots(2,4, figsize=(18,8))
		# Tube front stress (inner)
		axes[0,0].plot(times,sigmaEq_i)
		axes[0,0].set_xlabel(r'$t$ [h]')
		axes[0,0].set_ylabel(r'$\sigma_\mathrm{max,i}$ [MPa]')
		# Tube front stress (outer)
		axes[0,1].plot(times,sigmaEq_o)
		axes[0,1].set_xlabel(r'$t$ [h]')
		axes[0,1].set_ylabel(r'$\sigma_\mathrm{max,o}$ [MPa]')
		# Tube front temperatures
		axes[0,2].plot(times,T_i-273.15)
		axes[0,2].set_xlabel(r'$t$ [h]')
		axes[0,2].set_ylabel(r'$T_\mathrm{i}$ [\textdegree C]')
		# Tube front temperatures
		axes[0,3].plot(times,T_o-273.15)
		axes[0,3].set_xlabel(r'$t$ [h]')
		axes[0,3].set_ylabel(r'$T_\mathrm{o}$ [\textdegree C]')
		# Tube front temperature vs z
		z = np.linspace(0,model.H_rec*model.nz/model.nbins,model.nz)
		axes[1,0].plot(z,np.transpose(T_i)-273.15)
		axes[1,0].set_xlabel(r'$z$ [m]')
		axes[1,0].set_ylabel(r'$T_\mathrm{front}$ [\textdegree C]')
		# Tube front temperature vs z
		axes[1,1].plot(z,np.transpose(T_o)-273.15)
		axes[1,1].set_xlabel(r'$z$ [m]')
		axes[1,1].set_ylabel(r'$T_\mathrm{front}$ [\textdegree C]')
		# Fluid temperature vs z
		axes[1,2].plot(z,np.transpose(Tf)[1:,:]-273.15)
		axes[1,2].set_xlabel(r'$z$ [m]')
		axes[1,2].set_ylabel(r'$T_\mathrm{fluid}$ [\textdegree C]')
		# Fluid temperature vs z
		axes[1,3].plot(times,Tf[:,-1]-273.15)
		axes[1,3].set_xlabel(r'$t$ [h]')
		axes[1,3].set_ylabel(r'$T_\mathrm{fluid}$ [\textdegree C]')
		plt.tight_layout()
		plt.savefig('Heuristics.png',dpi=300)

class heuristics:
	def __init__(self, res='heuristics_res.mat', mat='800H', thermat = 'base', defomat='const_base', damat='base', folder='.'):
		self.folder = folder
		self.mat = mat
		self.defomat = defomat                                    # base | elastic_creep | elastic_model | const_elastic_creep | const_base
		self.thermal_mat, self.deformation_mat, self.damage_mat = library.load_material(mat, thermat, defomat, damat)
		self.res = res
		data = scipy.io.loadmat('%s/%s'%(self.folder,self.res))
		self.times = data['times'].flatten()
		self.nbins = data['nbins'].flatten()[0]
		self.epsilon_o = data['epsilon_o']
		self.sigma_o = data['sigma_o']
		self.T_o = data['T_o']
		self.strain = np.sqrt(2.0)/3.0 * np.sqrt(
			  (self.epsilon_o[:,:,0] - self.epsilon_o[:,:,1])**2.0
			+ (self.epsilon_o[:,:,1] - self.epsilon_o[:,:,2])**2.0
			+ (self.epsilon_o[:,:,2] - self.epsilon_o[:,:,0])**2.0
			+ 6.0 * (self.epsilon_o[:,:,3]**2.0
			+ self.epsilon_o[:,:,4]**2.0
			+ self.epsilon_o[:,:,5]**2.0)
			)
		self.stress = np.sqrt((
			(self.sigma_o[:,:,0] - self.sigma_o[:,:,1])**2.0 + 
			(self.sigma_o[:,:,1] - self.sigma_o[:,:,2])**2.0 + 
			(self.sigma_o[:,:,2] - self.sigma_o[:,:,0])**2.0 + 
			6.0 * (self.sigma_o[:,:,3]**2.0 + 
			self.sigma_o[:,:,4]**2.0 + 
			self.sigma_o[:,:,5]**2.0))/2.0)
		del data

	def uniaxial_neml(self,panel,position):
		lb = int(self.nbins*(panel-1) + position)
		print('	File: %s'%self.res)
		stress_corr = self.unneml(lb)
		filename = '%s/%s'%(self.folder,self.res)
		self.plot_stress(filename,stress_corr,lb)
		tR,time_dmg = self.creepdmg(self.T_o[:,lb],stress_corr)
		self.tablecsv(filename,stress_corr,tR,time_dmg,lb)

	def creepdmg(self,T,stress_corr):
		period = 24
		tR = self.damage_mat.time_to_rupture("averageRupture", T, stress_corr)
		dts = np.diff(self.times)
		time_dmg = dts/tR[1:]

		# Break out to cycle damage
		nt = int((self.times[-1]-self.times[0])/24)
		inds = id_cycles(self.times, period, nt)

		# Cycle damage
		Dc = np.array([np.sum(time_dmg[inds[i]:inds[i+1]], axis = 0) for i in range(nt)])
		print('	Creep damage: %s\n'%np.cumsum(Dc)[-1])

		return tR,time_dmg

	def tablecsv(self,filename,stress_corr,tR,time_dmg,lb):
		e = self.epsilon_o[:,lb,:]
		f = open('%s.csv'%(filename),'+w')
		f.write('time_neml,Temp_neml,stress_nts,stress_nml,e_xx_neml,e_yy_neml,e_zz_neml,e_xy_neml,e_xz_neml,e_yz_neml,tR,time_dmg\n')
		for t,T,sn,scorr,e1,e2,e3,e4,e5,e6,tr,td in zip(self.times,self.T_o[:,lb],self.stress[:,lb],stress_corr,e[:,0],e[:,1],e[:,2],e[:,3],e[:,4],e[:,5],tR,time_dmg):
			f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n'%(t,T,sn,scorr,e1,e2,e3,e4,e5,e6,tr,td))
		f.close()

	def plot_stress(self,filename,stress_corr,lb):
		fig,axes = plt.subplots(2,1,figsize=(6,8))

		axes[0].plot(self.times,self.stress[:,lb],label=r'$\sigma^{E}_\mathrm{eq}$')
		axes[0].plot(self.times,stress_corr      ,label=r'$\sigma_\mathrm{eq,NEML}$')
		axes[0].set_xlabel(r't (h)')
		axes[0].set_ylabel(r'$\sigma_\mathrm{eq}$ (MPa)')
		axes[0].legend(loc='best')

		axes[1].plot(self.times,self.T_o[:,lb]-273.15)
		axes[1].set_xlabel(r't (h)')
		axes[1].set_ylabel(r'$T_\mathrm{o}$ (\textdegree C)')

		plt.tight_layout()
		plt.savefig('%s.png'%(filename),dpi=300)

	def plot_strains(self):
		fig,axes = plt.subplots(1,1,figsize=(6,4))

		axes.plot(times,data['epsilon_o'][:,lb,0],label=r'$\varepsilon^{E}_\mathrm{rr}$')
		axes.plot(times,data['epsilon_o'][:,lb,1],label=r'$\varepsilon^{E}_{\theta\theta}$')
		axes.plot(times,data['epsilon_o'][:,lb,2],label=r'$\varepsilon^{E}_\mathrm{zz}$')
		axes.set_xlabel(r't (h)')
		axes.set_ylabel(r'$\varepsilon$ (mm/mm)')
		axes.legend(loc='best')

		plt.tight_layout()
		plt.savefig('elastic_strain.png',dpi=300)

	def unneml(self, lb):
		# Choose the material models
		deformation_mat = library.load_deformation(self.mat, self.defomat)
		neml_model = deformation_mat.get_neml_model()
		umodel = uniaxial.UniaxialModel(neml_model, verbose = False)

		hn = umodel.init_store()
		en = self.strain[0,lb]
		sn = self.stress[0,lb]
		Tn = self.T_o[0,lb]
		tn = self.times[0]
		un = 0.0
		pn = 0.0;
		es = [en]
		ss = [sn]
		ep=0

		for enp1, Tnp1, tnp1, snt in tqdm(zip(self.strain[1:,lb],self.T_o[1:,lb],self.times[1:],self.stress[1:,lb]),total=len(self.times[1:])):
			snp1, hnp1, Anp1, unp1, pnp1 = umodel.update(enp1, en, Tnp1, Tn, tnp1, tn, sn, hn, un, pn)
			ss.append(abs(snp1))
			sn = snp1
			hn = np.copy(hnp1)
			en = enp1
			Tn = Tnp1
			tn = tnp1
			un = unp1
			pn = pnp1
		return np.array(ss)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	# Heuristics
	parser.add_argument('--days', nargs=2, type=int, default=[0,365])
	parser.add_argument('--step', type=float, default=1800)
	parser.add_argument('--heuristics', type=int, default=1)
	parser.add_argument('--heuristics_res', type=str, default='GemasolarSystemOperation_res_0.mat')
	# UniaxialModel
	parser.add_argument('--neml', type=int, default=1)
	parser.add_argument('--neml_res', type=str, default='heuristics_res.mat')
	parser.add_argument('--panel', type=float, default=4)
	parser.add_argument('--position', type=float, default=30)
	args = parser.parse_args()

	tinit = time.time()

	# Heuristics
	if args.heuristics:
		run_heuristics(args.days,args.step,args.heuristics_res,mat="A230")
		os.system('mv heuristics_res.mat ./results_heuristics/heuristics565_N06230_el.mat')

	# UniaxialModel
	cases = ['565_N06230_el','565el','565in','565','600sa','600na']
	if args.neml:
		for f,z in zip(cases,[26,14,7,12,10,8]):
			model = heuristics(res='./results_heuristics/heuristics%s.mat'%f)
			model.uniaxial_neml(1,z)

	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))
