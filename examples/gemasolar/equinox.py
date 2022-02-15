import numpy as np
import matplotlib.pyplot as plt
import os, scipy.io, argparse
import time
from functools import partial
from multiprocessing import Pool

from thermal import *
from mechanical import *

def run_gemasolar(panel,nthreads=4):
	model = receiver_cyl()

	# Importing data from Modelica
	fileName = '%s/solartherm/examples/GemasolarSystemOperation_res.mat'%(os.path.expanduser('~'))
	model.import_mat(fileName)

	# Importing times
	times = model.data[:,0]

	# Filtering times
	index = []
	for i in range(len(times)):
		if times[i]%1800.==0 and times[i]>times[i-1] and i>0 and 6825600.0<=times[i]<=6915600.0:
			index.append(i)

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
		Tf[0,0])

	# Writting damage and life file headers
	lifeName = 'st_nash_tube_stress_res_%s.txt'%panel
	import datetime
	dt = datetime.datetime.now()
	ds = dt.strftime("%Y-%m-%d-%H:%M:%S %p")
	if not os.path.isfile(lifeName):
		f = open(lifeName,'a')
		f.write('Simulation: %s\n'%ds)
		f.write('Section,life,Dc,Df\n')
		f.close()
	else:
		f = open(lifeName,'a')
		f.write('Simulation: %s\n'%ds)
		f.close()

	# Running srlife
	fun = partial(run_problem, nz=model.nbins, progress_bar=False)
	with Pool(nthreads) as p:
		lives = p.map(fun, np.linspace(1,model.nbins,num=model.nbins,dtype=int))
	f = open(lifeName,'a')
	for i in lives:
		f.write('%s,%s,%s,%s\n'%(int(i[3]-1),i[0],i[1],i[2]))
	f.close()

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Estimates average damage of a representative tube in a receiver panel')
	parser.add_argument('--panel', type=int, default=1, help='Panel to be simulated. Default=1')
	args = parser.parse_args()

	tinit = time.time()
	run_gemasolar(args.panel)
	seconds = time.time() - tinit
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	print('Simulation time: {:d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

