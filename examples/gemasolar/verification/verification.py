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
from tqdm import tqdm
import nashTubeStress as nts
import coolant

sys.path.append('../')
from section import *

def thermal_verification():
	# Instantiating receiver model
	model = receiver_cyl(Ri = 42/2000, Ro = 45/2000, R_fouling=8.808e-5, ab = 0.93, em = 0.87, kp = 21.0, Dittus=False)

	# Importing solar flux Sanchez-Gonzalez et al. (2017): https://doi.org/10.1016/j.solener.2015.12.055
	CG = np.genfromtxt('fluxInput.csv', delimiter=',')*1e6

	# Flow and thermal parameters Sanchez-Gonzalez et al. (2017): https://doi.org/10.1016/j.solener.2015.12.055
	m_flow_tb = 4.2
	Tamb = 298.15
	h_ext = 0

	# Instantiating variables
	z = np.linspace(0,model.H_rec*model.nz/model.nbins,model.nz)
	Tf = model.T_in*np.ones(model.nz+1)
	Tf_nts = model.T_in*np.ones(model.nz+1)
	T_o = np.zeros(model.nz)
	T_i = np.zeros(model.nz)
	T_i_nts = np.zeros(model.nz)
	T_o_nts = np.zeros(model.nz)

	# nashTubeStress objetcs
	g = nts.Grid(nr=30, nt=model.nt, rMin=model.Ri, rMax=model.Ro)
	s = nts.Solver(g, debug=False, CG=CG[0], k=model.kp, T_int=model.T_in, R_f=model.R_fouling,
                   A=model.ab, epsilon=model.em, T_ext=Tamb, h_ext=h_ext,
                   P_i=0e5, alpha=model.l, E=model.E, nu=model.nu, n=1,
                   bend=False)
	s.extBC = s.extTubeHalfCosFluxRadConv
	s.intBC = s.intTubeConv

	# Running thermal model
	for k in tqdm(range(model.nz)):
		Qnet = model.Temperature(m_flow_tb, Tf[k], Tamb, CG[k], h_ext)
		C = model.specificHeatCapacityCp(Tf[k])*m_flow_tb
		Tf[k+1] = Tf[k] + Qnet/C
		T_o[k] = model.To
		T_i[k] = model.Ti
		# Reference data Logie et al. (2018): https://doi.org/10.1016/j.solener.2017.12.003
		salt = coolant.nitrateSalt(False)
		salt.update(Tf_nts[k])
		h_int, dP = coolant.HTC(False, salt, model.Ri, model.Ro, model.kp, 'Dittus', 'mdot', m_flow_tb)
		s.h_int = h_int
		s.CG = CG[k]
		s.T_int = Tf_nts[k]
		ret = s.solve(eps=1e-6)
		s.postProcessing()
		Q = 2*np.pi*model.kp*s.B0*model.dz
		Tf_nts[k+1] = Tf_nts[k] + Q/salt.Cp/m_flow_tb
		T_i_nts[k] = s.T[0,0]
		T_o_nts[k] = s.T[0,-1]
		# end nashTubeStress

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
	xj = xj[np.logical_not(np.isnan(ypd))]
	yfj = yfj[np.logical_not(np.isnan(ypd))]
	ypj = ypj[np.logical_not(np.isnan(ypd))]

	# Creating directory
	data = {}

	# Saving reference data
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

	# Saving proposed model data
	data['z'] = z
	data['T_o'] = T_o
	data['T_i'] = T_i
	data['Tf'] = Tf
	scipy.io.savemat('verification_res.mat',data)

def plottingTemperatures():
	# Importing data
	data = scipy.io.loadmat('verification_res.mat')

	# Creating subplots
	fig, axes = plt.subplots(1,2, figsize=(18,4))

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
	plt.savefig('Verification.png',dpi=300)

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
