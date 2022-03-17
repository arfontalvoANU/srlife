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

	# Importing data from Modelica
	if clearSky:
		fileName = 'GemasolarSystemOperationCS_res.mat'
	else:
		fileName = 'GemasolarSystemOperation_res.mat'
	model.import_mat(fileName)

	# Importing times
	times = model.data[:,0]

	# Filtering times
	index = []
	_times = []
	time_lb = days[0]*86400
	time_ub = days[1]*86400+3600
	for i in range(len(times)):
		if times[i]%1800.==0 and times[i] not in _times and time_lb<=times[i] and times[i]<time_ub:
			index.append(i)
			_times.append(times[i])

	# Importing flux
	CG = model.data[:,model._vars['heliostatField.CG[1]'][2]:model._vars['heliostatField.CG[450]'][2]+1]
	m_flow_tb = model.data[:,model._vars['heliostatField.m_flow_tb'][2]]
	Tamb = model.data[:,model._vars['receiver.Tamb'][2]]
	h_ext = model.data[:,model._vars['receiver.h_conv'][2]]

	# Getting inputs based on filtered times
	times = times[index]/3600.
	times = times.flatten()
	CG = CG[index,:]
	m_flow_tb = m_flow_tb[index]
	Tamb = Tamb[index]
	h_ext = h_ext[index]

	# Instantiating variables
	stress = np.zeros((times.shape[0],model.nz))
	Tf = model.T_in*np.ones((times.shape[0],model.nz+1))
	qnet = np.zeros((times.shape[0],2*model.nt-1,model.nz))

	# Running thermal model
	for k in range(model.nz):
		Qnet = model.Temperature(m_flow_tb, Tf[:,k], Tamb, CG[:,k], h_ext)
		C = model.specificHeatCapacityCp(Tf[:,k])*m_flow_tb
		Tf[:,k+1] = Tf[:,k] + np.divide(Qnet, C, out=np.zeros_like(C), where=C!=0)
		stress[:,k] = model.s/1e6
		qnet[:,:,k] = model.qnet/1e6

	# Getting internal pressure
	pressure = np.where(m_flow_tb>0, 0.1, m_flow_tb)
	pressure = pressure.flatten()
	lb = model.nbins*(panel-1)
	ub = lb + model.nbins

	# Creating the hdf5 model
	setup_problem(
		model.Ro,
		model.thickness,
		model.H_rec,
		9,
		2*model.nt-1,
		model.nbins,
		times,
		Tf[:,lb:ub],
		qnet[:,:,lb:ub],
		pressure,
		Tf[0,0],
		days=days[1]-days[0])

	# Running srlife
	resfolder = os.path.join(os.getcwd(),'results')
	if not os.path.isdir(resfolder):
		os.mkdir(resfolder)
	life = run_problem(position, model.nbins, nthreads=nthreads, load_state0=load_state0, savestate=savestate, resfolder=resfolder)

	# Saving thermal results
	scipy.io.savemat('%s/st_nash_tube_stress_res.mat'%resfolder,{
		"times":times,
		"fluid_temp":Tf,
		"CG":CG,
		"m_flow_tb":m_flow_tb,
		"pressure":pressure,
		"h_flux":qnet/1e6
		})

	# Plotting thermal results
	fig, axes = plt.subplots(2,2, figsize=(11,8))
	# HTF temperature at several locations
	for i in range(9):
		pos = i*model.nbins
		axes[0,0].plot(times/3600.,Tf[:,pos], label='z[%s]'%pos)
	axes[0,0].set_ylabel('HTF temperature (K)')
	axes[0,0].set_xlabel('Time [h]')
	axes[0,0].legend(loc='best', frameon=False)
	# Mass flow rate
	axes[0,1].plot(times/3600.,m_flow_tb)
	axes[0,1].set_ylabel('Mass flow rate (kg/s)')
	axes[0,1].set_xlabel('Time [h]')
	# Concentrated solar fluxes
	for i in range(9):
		pos = i*model.nbins
		axes[1,0].plot(times/3600.,CG[:,pos], label='z[%s]'%pos)
	axes[1,0].set_ylabel('CG (W/m2)')
	axes[1,0].set_xlabel('Time [h]')
	axes[1,0].legend(loc='best', frameon=False)
	# Equivalent stress
	for i in range(9):
		pos = i*model.nbins
		axes[1,1].plot(times/3600.,stress[:,pos], label='z[%s]'%pos)
	axes[1,1].set_ylabel('Equivalent stress [MPa]')
	axes[1,1].set_xlabel('Time [h]')
	axes[1,1].legend(loc='best', frameon=False)
	# Show
	plt.savefig('%s/st_nash_tube_stress_fig.png'%resfolder)

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
	run_gemasolar(args.panel,args.position,args.days,args.nthreads,args.clearSky,args.load_state0,args.savestate)
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

