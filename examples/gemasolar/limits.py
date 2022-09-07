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
import itertools

sys.path.append('../')
from srlife import receiver, solverparams, library, thermal, structural, system, damage, managers
from neml import uniaxial
from section import *

from CoolProp.CoolProp import PropsSI

import colorama
colorama.init()
def yellow(text):
	return colorama.Fore.YELLOW + colorama.Style.BRIGHT + text + colorama.Style.RESET_ALL

class solver:
	def __init__(self, Ri = 42/2000, Ro = 45/2000, R_fouling=8.808e-5, ab = 0.93, em = 0.87, T=563.15, vf=1,
		               kp = 18.3, alpha = 18.3e-06, Young = 170.9e9, poisson = 0.31, Tamb = 298.15,
		               mat='800H', thermat = 'base', defomat='const_base', damat='base'):

		"""
		Inputs:
		   Ri:                                Tube inner radius
		   Ro:                                Tube outer radius
		   R_fouling:                         Fouling resistance
		   ab:                                Coating absorptivity
		   em:                                Coating emissivity
		   kp:                                Tube metal thermal conductivity
		   alpha:                             Tube metal coefficient of thermal expansion
		   Young:                             Tube metal Young's modulus of elasticity
		   poisson:                           Tube metal Poisson ratio
		   mat:                               Tube metal name              (as in SRLIFE data folder)
		   thermat:                           Tube metal thermal model     (as in SRLIFE data folder)
		   defomat:                           Tube metal deformation model (as in SRLIFE data folder)
		   damat:                             Tube metal damage model      (as in SRLIFE data folder)
		"""

		# Instantiating receiver model
		self.model = receiver_cyl(Ri = Ri, Ro = Ro, R_fouling=R_fouling, ab = ab,
		                          em = em, kp = kp, Dittus=False, alpha = alpha, Young = Young,
		                          poisson = poisson)

		self.CG = np.array([0,0,0.7,0.875,0.95,0.99,1,0.99,0.95,0.875,0.7,0,0])
		self.times = np.array([0,7,7.5,8,9,11,12,13,15,16,16.5,17,24])
		self.h_ext = 20.
		self.Tamb = Tamb
		self.areas = self.model.area*np.ones_like(self.CG)
		self.T = T
		self.vf = vf
		if mat=='800H':
			self.fname = 'N08811'
		elif mat=='A230':
			self.fname = 'N06230'
		self.OD = 2000*Ro
		self.WT = 1000*(Ro - Ri)

		# Getting NEML models
		self.mat = mat
		self.defomat = defomat                                    # base | elastic_creep | elastic_model | const_elastic_creep | const_base
		self.thermal_mat, self.deformation_mat, self.damage_mat = library.load_material(mat, thermat, defomat, damat)

	def plotting(stress_corr):
		# Creating subplots
		fig, axes = plt.subplots(1,1, figsize=(6,4))

		# Inner surface temperature of front tube
		axes.plot(self.times,stress_corr,label='Elastic')
		axes.set_xlabel(r't (h)')
		axes.set_ylabel(r'$\sigma_\mathrm{crown}$ [MPa]')
		axes.legend(loc="best", borderaxespad=0, ncol=1, frameon=False)

		plt.tight_layout()
		plt.show()


	def creepdmg(self,T,stress_corr,times):
		period = 24
		tR = self.damage_mat.time_to_rupture("averageRupture", T, stress_corr)
		dts = np.diff(times)
		time_dmg = dts/tR[1:]

		# Break out to cycle damage
		nt = int((times[-1]-times[0])/24)
		inds = self.id_cycles(times, period, nt)

		# Cycle damage
		Dc = np.array([np.sum(time_dmg[inds[i]:inds[i+1]], axis = 0) for i in range(nt)])
		return np.cumsum(Dc)[-1]

	def id_cycles(self, times, period, days):
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
			print(times)
			raise ValueError("Tube times not compatible with the receiver number of days and cycle period!")
		return inds

	def unneml(self,strain,stress,T,times):
		# Choose the material models
		deformation_mat = library.load_deformation(self.mat, self.defomat)
		neml_model = deformation_mat.get_neml_model()
		umodel = uniaxial.UniaxialModel(neml_model, verbose = False)

		hn = umodel.init_store()
		en = strain[0]
		sn = stress[0]
		Tn = T[0]
		tn = times[0]
		un = 0.0
		pn = 0.0;
		es = [en]
		ss = [sn]
		ep=0

		#for enp1, Tnp1, tnp1, snt in tqdm(zip(strain[1:],T[1:],times[1:],stress[1:]),total=len(strain[1:])):
		for enp1, Tnp1, tnp1, snt in zip(strain[1:],T[1:],times[1:],stress[1:]):
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

	def fun(self, flux):
		# Update density
		rho = self.model.density(self.T)
		# Calculate mass flow rate
		m = rho*self.vf*self.areas

		# Running thermal self.model
		Qnet = self.model.Temperature(m, self.T, self.Tamb, flux*self.CG, self.h_ext)

		stress = np.sqrt((
		(self.model.stress[:,0,0] - self.model.stress[:,0,1])**2.0 + 
		(self.model.stress[:,0,1] - self.model.stress[:,0,2])**2.0 + 
		(self.model.stress[:,0,2] - self.model.stress[:,0,0])**2.0 + 
		6.0 * (self.model.stress[:,0,3]**2.0 +
		self.model.stress[:,0,4]**2.0 +
		self.model.stress[:,0,5]**2.0))/2.0)*1e-6

		strain = np.sqrt(2.0)/3.0 * np.sqrt(
			  (self.model.epsilon[:,0,0] - self.model.epsilon[:,0,1])**2.0
			+ (self.model.epsilon[:,0,1] - self.model.epsilon[:,0,2])**2.0
			+ (self.model.epsilon[:,0,2] - self.model.epsilon[:,0,0])**2.0
			+ 6.0 * (self.model.epsilon[:,0,3]**2.0
			+ self.model.epsilon[:,0,4]**2.0
			+ self.model.epsilon[:,0,5]**2.0)
			)

		n = self.times.shape[0]
		times = self.times
		To = self.model.To
		Ti = self.model.Ti

		for i in range(1,365):
			times = np.concatenate((times,24*i+times[1:n]))
			stress = np.concatenate((stress,stress[1:n]))
			strain = np.concatenate((strain,strain[1:n]))
			To = np.concatenate((To,To[1:n]))
			Ti = np.concatenate((Ti,Ti[1:n]))

		try:
			stress_corr = self.unneml(strain,stress,To,times)
			dc = self.creepdmg(To,stress_corr,times)
		except:
			dc = self.creepdmg(To,stress,times)
		return 1 - dc*30

def bisection(f,a,b,T,vf,tol=1e-5, maxiter=100, debug=False):
	fa = f(a)
	fb = f(b)
	fm = f((a + b)/2.)
	it = 0
	while np.abs(fm) > tol and it < maxiter:
		if np.sign(fa) == np.sign(fm):
			a = (a + b)/2.
		elif np.sign(fb) == np.sign(fm):
			b = (a + b)/2.
		fm = f((a + b)/2.)
		it += 1
		#print('	x: %.2f'%(a/2. + b/2.) + ' '*(15 - len('%.2f'%(a/2. + b/2.))) + 'f: %s'%fm)
	return (a + b)/2.

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--mdba_verification', type=bool, default=False)
	parser.add_argument('--flux_filename', type=str, default='flux-table')
	args = parser.parse_args()
	tinit = time.time()
	#ss = solver(kp = 17.97, alpha = 15.61e-06, Young = 184e9, poisson = 0.31, mat='A230')
	ss = solver()
	T_int = np.linspace(290, 565, 12)
	T_int = np.append(T_int,600.) + 273.15
	csv = np.c_[T_int,]
	fluxSalt = np.zeros(len(T_int))
	header = '0'

	fig = plt.figure(figsize=(3.5, 3.5))
	ax = fig.add_subplot(111)

	for vf in [0.25, 0.5, 0.75]:
		for i,Ti in tqdm(enumerate(T_int), total=len(T_int)):
#		for i,Ti in enumerate(T_int):
			ss.T = Ti
			ss.vf = vf
#			print('T: %s	v: %s'%(Ti,vf))
			fluxSalt[i] = bisection(ss.fun,1e4,1e6,Ti,vf)
			seconds = time.time() - tinit
		csv = np.c_[csv,fluxSalt,]
		header += ',{0}'.format(vf)

		ax.plot(T_int-273.15,fluxSalt*1e-6, label=r'U = {0} m/s'.format(vf))
		ax.set_xlabel(r'$T_\mathrm{f}$ (\textdegree C)')
		ax.set_ylabel(r'$\vec{\phi_\mathrm{q}}$ (W$\cdot$ m$^{-2}$)')

		ax.legend(loc='best')
		fig.tight_layout()
		fig.savefig('{0}_OD{1:.2f}_WT{2:.2f}_peakFlux.pdf'.format(ss.fname, ss.OD, ss.WT),transparent=True)
		plt.close(fig)

	np.savetxt('{0}_OD{1:.2f}_WT{2:.2f}_peakFlux.csv'.format(ss.fname, ss.OD, ss.WT),
		csv, delimiter=',', header=header)
	
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

