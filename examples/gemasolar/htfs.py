import numpy as np

from thermal import *

def gen_htfs():
	Ri = 20.0/2000
	Ro = 22.4/2000
	model = receiver_cyl(Ri = Ri, Ro = Ro)   # Instantiating model with the gemasolar geometry

	Tf = np.linspace(548.15, 873.15, 14)
	v_flow = 1.5*np.ones(14)
	hf = model.htfs(v_flow, Tf)
	print(hf/1e6)
	print(Tf)

if __name__=='__main__':
	gen_htfs()
