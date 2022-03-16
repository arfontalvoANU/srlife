import numpy as np
import os, sys
import scipy.optimize as opt
import scipy.io as sio
import time
import argparse

sys.path.append('../..')

from srlife import receiver, library

def id_cycles(tube, receiver):
	"""
		Helper to separate out individual cycles by index

		Parameters:
			tube        single tube with results
			receiver    receiver, for metadata
	"""
	tm = np.mod(tube.times, receiver.period)
	inds = list(np.where(tm == 0)[0])
	if len(inds) != (receiver.days + 1):
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

def calculate_damage(fileName,update,material,paneln,tuben):

	model = receiver.Receiver.load('%s.hdf5'%fileName)

	thermal_mat, deformation_mat, damage_mat = library.load_material(material, 'base', 'base', 'base')

	tube = model.panels[paneln].tubes[tuben]

	### Creep damage ###

	# Von Mises Stress
	vm = np.sqrt((
		(tube.quadrature_results['stress_xx'] - tube.quadrature_results['stress_yy'])**2.0 + 
		(tube.quadrature_results['stress_yy'] - tube.quadrature_results['stress_zz'])**2.0 + 
		(tube.quadrature_results['stress_zz'] - tube.quadrature_results['stress_xx'])**2.0 + 
		6.0 * (tube.quadrature_results['stress_xy']**2.0 + 
		tube.quadrature_results['stress_yz']**2.0 + 
		tube.quadrature_results['stress_xz']**2.0))/2.0)

	# Time to rupture
	tR = damage_mat.time_to_rupture("averageRupture", tube.quadrature_results['temperature'], vm)
	dts = np.diff(tube.times)
	time_dmg = dts[:,np.newaxis,np.newaxis]/tR[1:]

	# Break out to cycle damage
	inds = id_cycles(tube, model)

	# Cycle damage
	Dc = np.array([np.sum(time_dmg[inds[i]:inds[i+1]], axis = 0) for i in range(model.days)])


	### Fatigue cycles ###

	# Identify cycle boundaries
	inds = id_cycles(tube, model)

	# Run through each cycle and ID max strain range and fatigue damage
	strain_names = ['mechanical_strain_xx', 'mechanical_strain_yy', 'mechanical_strain_zz',
		'mechanical_strain_yz', 'mechanical_strain_xz', 'mechanical_strain_xy']
	strain_factors = [1.0,1.0,1.0,2.0, 2.0, 2.0]

	Df =  np.array([cycle_fatigue(np.array([ef*tube.quadrature_results[en][
	  inds[i]:inds[i+1]] for 
	  en,ef in zip(strain_names, strain_factors)]), 
	  tube.quadrature_results['temperature'][inds[i]:inds[i+1]], damage_mat)
	  for i in range(model.days)])


	### Calculating the number of cycles

	# Defining the number of columns as the number of days
	# This is used to create an array with nrows = nelements x nquad,
	# and ncols = number of days
	nc = model.days
	max_cycles = []

	for c,f in zip(Dc.reshape(nc,-1).T, Df.reshape(nc,-1).T):
		# The damage is extrapolated and the number of cycles is determined
		# There are three extrapolation approaches. Here we use the 'lump' one
		max_cycles.append(calculate_max_cycles(make_extrapolate(c), make_extrapolate(f), damage_mat))

	max_cycles = np.array(max_cycles)
	print(min(max_cycles))

	if update:
		savename = 'quadrature_results.mat'
		mydict = sio.loadmat(savename)
	else:
		savename = 'damage_results.mat'
		mydict = {}
	mydict['cumDc'] = np.max(np.cumsum(Dc.reshape(nc,-1).T, axis=1), axis=0)
	mydict['cumDf'] = np.max(np.cumsum(Df.reshape(nc,-1).T, axis=1), axis=0)
	mydict['Dc'] = np.max(Dc.reshape(nc,-1).T, axis=0)
	mydict['Df'] = np.max(Df.reshape(nc,-1).T, axis=0)
	mydict['max_cycles'] = max_cycles
	sio.savemat(savename, mydict)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--filename', type=str, default='model_solved', help='hdf5 containing the final results')
	parser.add_argument('--material', type=str, default='A230', help='Damage material')
	parser.add_argument('--panel', type=str, default='panel0', help='Panel to calculate damage')
	parser.add_argument('--tube', type=str, default='tube0', help='Tube to calculate damage')
	parser.add_argument('--update', type=bool, default=False, help='Update quadrature_results from mechanical solver')
	args = parser.parse_args()
	# Running function
	tinit = time.time()
	calculate_damage(args.filename, args.update, args.material, args.panel, args.tube)
	# Estimating simulation time
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

