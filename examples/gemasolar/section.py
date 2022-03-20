import numpy as np
import matplotlib.pyplot as plt
import os, scipy.io, argparse
import time
from functools import partial
from multiprocessing import Pool

from thermal import *
from mechanical import *

def run_gemasolar(panel,position,days,nthreads,clearSky,load_state0,savestate):
	model = receiver_cyl(Ri = 20.0/2000, Ro = 22.4/2000)

	data = scipy.io.loadmat('matlab.mat')
	times = data['times'].reshape(-1)
	Tf = data['fluid_temp']
	qnet = data['h_flux']
	pressure = data['pressure'].reshape(-1)

	# Creating the hdf5 model
	setup_problem(
		model.Ro,
		model.thickness,
		model.H_rec,
		9,
		2*model.nt-1,
		model.nbins,
		times,
		Tf,
		qnet,
		pressure,
		Tf[0,0],
		days=30)

	# Running srlife
	resfolder = os.path.join(os.getcwd(),'results')
	if not os.path.isdir(resfolder):
		os.mkdir(resfolder)
	life = run_problem(position, model.nbins, nthreads=nthreads, load_state0=load_state0, savestate=savestate, resfolder=resfolder)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--panel', type=int, default=1, help='Panel to be simulated. Default=1')
	parser.add_argument('--position', type=float, default=1, help='Panel position to be simulated. Default=1')
	parser.add_argument('--days', nargs=2, type=int, default=[0,1], help='domain of days to simulate')
	parser.add_argument('--nthreads', type=int, default=4, help='Number of processors. Default=4')
	parser.add_argument('--clearSky', type=bool, default=False, help='Run clear sky DNI (requires to have the solartherm results)')
	parser.add_argument('--load_state0', type=bool, default=False, help='Load state from a previous simulation')
	parser.add_argument('--savestate', type=bool, default=False, help='Save the last state of the last simulated day')
	args = parser.parse_args()

	tinit = time.time()
	run_gemasolar(4,30.0,[0,1],4,True,False,True)
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

