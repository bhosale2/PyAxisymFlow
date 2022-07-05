black:
	@black --version
	@black examples kernels utils set_sim_params.py

all:black
