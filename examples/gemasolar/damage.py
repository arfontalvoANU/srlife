import numpy as np
import os, sys
import scipy.optimize as opt
import scipy.io
import time
import argparse

sys.path.append('../..')

from srlife import receiver, library

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

def calculate_damage(fileName,clearSky,days,material,source):

	if clearSky:
		mydict = scipy.io.loadmat('%s/input_clear_sky.mat'%source)
		case = 'clear'
	else:
		mydict = scipy.io.loadmat('%s/input_tmy_data.mat'%source)
		case = 'tmy'

	times = mydict['times'].flatten()
	index = np.where((times>=days[0]*24) & (times<=days[1]*24))[0]

	quadrature_results = scipy.io.loadmat(fileName)

	thermal_mat, deformation_mat, damage_mat = library.load_material(material, 'base', 'base', 'base')

	### Creep damage ###

	# Von Mises Stress
	vm = np.sqrt((
		(quadrature_results['stress_xx'] - quadrature_results['stress_yy'])**2.0 + 
		(quadrature_results['stress_yy'] - quadrature_results['stress_zz'])**2.0 + 
		(quadrature_results['stress_zz'] - quadrature_results['stress_xx'])**2.0 + 
		6.0 * (quadrature_results['stress_xy']**2.0 + 
		quadrature_results['stress_yz']**2.0 + 
		quadrature_results['stress_xz']**2.0))/2.0)

	# Time to rupture
	ntimes = np.shape(quadrature_results['temperature'])[0]
	period = 24
	days = days[1] - days[0]
	times = times[index]
	tR = damage_mat.time_to_rupture("averageRupture", quadrature_results['temperature'], vm)
	dts = np.diff(times)
	time_dmg = dts[:,np.newaxis,np.newaxis]/tR[1:]

	# Break out to cycle damage
	inds = id_cycles(times, period, days)

	# Cycle damage
	Dc = np.array([np.sum(time_dmg[inds[i]:inds[i+1]], axis = 0) for i in range(days)])


	### Fatigue cycles ###

	# Identify cycle boundaries
	inds = id_cycles(times, period, days)

	# Run through each cycle and ID max strain range and fatigue damage
	strain_names = ['mechanical_strain_xx', 'mechanical_strain_yy', 'mechanical_strain_zz',
		'mechanical_strain_yz', 'mechanical_strain_xz', 'mechanical_strain_xy']
	strain_factors = [1.0,1.0,1.0,2.0, 2.0, 2.0]

	Df =  np.array([cycle_fatigue(np.array([ef*quadrature_results[en][
	  inds[i]:inds[i+1]] for 
	  en,ef in zip(strain_names, strain_factors)]), 
	  quadrature_results['temperature'][inds[i]:inds[i+1]], damage_mat)
	  for i in range(days)])


	### Calculating the number of cycles

	# Defining the number of columns as the number of days
	# This is used to create an array with nrows = nelements x nquad,
	# and ncols = number of days
	nc = days
	max_cycles = []

	for c,f in zip(Dc.reshape(nc,-1).T, Df.reshape(nc,-1).T):
		# The damage is extrapolated and the number of cycles is determined
		# There are three extrapolation approaches. Here we use the 'lump' one
		max_cycles.append(calculate_max_cycles(make_extrapolate(c), make_extrapolate(f), damage_mat))

	max_cycles = np.array(max_cycles)
	print(min(max_cycles))

	quadrature_results['cumDc'] = np.cumsum(Dc.reshape(nc,-1).T, axis=1)
	quadrature_results['cumDf'] = np.cumsum(Df.reshape(nc,-1).T, axis=1)
	quadrature_results['Dc'] = Dc.reshape(nc,-1).T
	quadrature_results['Df'] = Df.reshape(nc,-1).T
	scipy.io.savemat('damage_results.mat', quadrature_results)
	#scipy.io.savemat(fileName, quadrature_results)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--filename', type=str, default='quadrature_results.mat', help='hdf5 containing the final results')
	parser.add_argument('--clearSky', type=bool, default=False, help='Run clear sky DNI (requires to have the solartherm results)')
	parser.add_argument('--days', nargs=2, type=int, default=[0,1], help='domain of days to simulate')
	parser.add_argument('--material', type=str, default='A230', help='Damage material')
	args = parser.parse_args()
	# Running function
	tinit = time.time()
	source = os.getcwd()
	calculate_damage(args.filename,args.clearSky,args.days,args.material,source)
	# Estimating simulation time
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

