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
from section import *

def print_table_latex(model, data):
	print('')
	print('Cumulative creep and fatigue damage and projected receiver life for a nitrate-salt Gemasolar-like receiver. Material: UNS N06230.')
	for i in range(int(model.nz/model.nbins)):
		lb = model.nbins*i
		ub = lb + model.nbins - 1
		j = np.argmin(data['max_cycles'][lb:ub])
		print('%d & %d & %.1e & %.1e & %4.2f \\\\'%(
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

def run_heuristics(days,step,gemasolar_600=False):
	# Instantiating receiver model
	model = receiver_cyl(Ri = 20.0/2000, Ro = 22.4/2000, R_fouling=8.808e-5, ab = 0.93, em = 0.87)

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
	print_table_latex(model,data)
	scipy.io.savemat('heuristics_res.mat',data)

	if (days[1] - days[0])==1:
		# Creating subplots
		fig, axes = plt.subplots(2,2, figsize=(12,8))
		# Front tube stress
		axes[0,0].plot(times,sigmaEq_i[:,180],label='Inner')
		axes[0,0].plot(times,sigmaEq_o[:,180],label='Outer')
		axes[0,0].set_xlabel(r'$t$ [h]')
		axes[0,0].set_ylabel(r'$\sigma_\mathrm{crown,eq}$ [MPa]')
		axes[0,0].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)
		# Front tube temperatures
		axes[0,1].plot(times,T_i[:,180]-273.15,label='Inner')
		axes[0,1].plot(times,T_o[:,180]-273.15,label='Outer')
		axes[0,1].set_xlabel(r'$t$ [h]')
		axes[0,1].set_ylabel(r'$T_\mathrm{front}$ [\textdegree C]')
		axes[0,1].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)
		# Inner surface temperature of front tube vs z
		bnd = np.unravel_index(sigmaEq_o.argmax(), sigmaEq_o.shape)
		bnd = bnd[0]
		z = np.linspace(0,model.H_rec*model.nz/model.nbins,model.nz)
		axes[1,0].plot(z,np.transpose(T_i)[:,bnd]-273.15,label='Inner')
		axes[1,0].plot(z,np.transpose(T_o)[:,bnd]-273.15,label='Outer')
		axes[1,0].set_xlabel(r'$z$ [m]')
		axes[1,0].set_ylabel(r'$T_\mathrm{front}$ [\textdegree C]')
		axes[1,0].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)
		# Fluid temperature vs z
		axes[1,1].plot(z,np.transpose(Tf)[1:,bnd]-273.15,label='Inner')
		axes[1,1].set_xlabel(r'$z$ [m]')
		axes[1,1].set_ylabel(r'$T_\mathrm{fluid}$ [\textdegree C]')
		axes[1,1].legend(loc="best", borderaxespad=0, ncol=1, frameon=False)
		plt.tight_layout()
		plt.savefig('Heuristics.png',dpi=300)

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
