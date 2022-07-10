black:
	@black --version
	@black examples kernels pyst_kernels utils set_sim_params.py

all:black
