import numpy as np
from sympy import exp, I, conjugate, diff, lambdify, symbols, re, im, besselj, bessely, simplify
 
def theory_axisymmetric_rigid_slab_spatial(omega,nu,R_tube):
	y, t = symbols('y, t', real=True)
	k1 = omega / nu 
	lam1 = np.sqrt(1j * k1)
	A = 0.5/besselj(0, lam1 * R_tube)
	func_soln = A*besselj(0, lam1 * y)
	soln_lam = lambdify(y, func_soln)
	y_range=  np.linspace(0,R_tube,100)
	Soln = soln_lam(y_range)
	return Soln

def theory_axisymmetric_rigid_slab_temporal(U_0, Soln, omega,t):
	T_soln = U_0 *np.real(2*Soln*np.exp(-1j*omega*(t))) 
	return T_soln