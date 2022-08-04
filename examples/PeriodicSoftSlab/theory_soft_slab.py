import numpy as np
from sympy import exp, I, conjugate, diff, lambdify, symbols, re, im, besselj, bessely, simplify
from scipy import optimize
from sympy.solvers.solveset import linsolve

def theory_axisymmetric_soft_slab_spatial(L_f,L_s,Re,shear_rate,omega,nu,G,V_wall):

	#Theoretical Solution
	y, t1 = symbols('y, t1', real=True)

	# define params
	L_s = 0.1
	rho_f = 1
	rho = 1
	rho_s = rho * rho_f

	nu_f = shear_rate * omega * L_f ** 2 / Re
	mu_f = nu_f * rho_f
	nu_s = 0.0
	mu_s = nu_s * rho_s
	#c1 = mu_f * shear_rate * omega / 2 / Er

	lam1 = np.sqrt(1j * omega / nu_f)
	lam2 = omega / np.sqrt(-1j * omega * nu_s +  G / rho_s)

	# solution form
	A, B, C = symbols('A, B, C')
	vel_f = (A * besselj(0, lam1 * y) + B * bessely(0, lam1 * y)) * exp(-I * omega * t1)
	u_s = (C * besselj(0, lam2 * y)) * exp(-I * omega * t1)
	vel_s = diff(u_s, t1)

	# solve for coeffs
	k1, k2, k3, k4, k5, k6, k7, k8 = symbols('k1, k2, k3, k4, k5, k6, k7, k8')
	eq1 = A * k1 + B * k2 - V_wall/2
	eq2 = A * k3 + B * k4 - C * k5
	eq3 = A * k6 + B * k7 - C * k8
	ans, = linsolve([eq1, eq2, eq3], (A, B, C))

	from scipy.special import jv, yv

	ans = ans.subs(k1, jv(0, lam1 * (L_s + L_f)))
	ans = ans.subs(k2, yv(0, lam1 * (L_s + L_f)))
	ans = ans.subs(k3, jv(0, lam1 * L_s))
	ans = ans.subs(k4, yv(0, lam1 * L_s))
	ans = ans.subs(k5, -1j * omega * jv(0, lam2 * L_s))
	ans = ans.subs(k6, jv(1, lam1 * L_s))
	ans = ans.subs(k7, yv(1, lam1 * L_s))
	ans = ans.subs(k8, lam2 * jv(1, lam2 * L_s) / (mu_f * lam1) * (G - mu_s * 1j * omega))
	vel_f = vel_f.subs(A, ans[0].simplify())
	vel_f = vel_f.subs(B, ans[1].simplify())
	vel_s = vel_s.subs(C, ans[2].simplify())

	vel_f += conjugate(vel_f)
	vel_s += conjugate(vel_s)

	vel_fl = lambdify([y, t1], vel_f)
	vel_sl = lambdify([y, t1], vel_s)

	res_y = 30
	res_t = 21
	eps = 1e-20
	Y = np.linspace(eps, (L_s + L_f), res_y)
	offset = np.pi / 2
	return Y,vel_sl,vel_fl

def theory_axisymmetric_soft_slab_temporal(Y,t,L_s,vel_sl,vel_fl):
	vel_comb = ((Y < L_s) * np.real(vel_sl(Y, t * np.ones_like(Y))) 
	               + (Y >= L_s) * np.real(vel_fl(Y, t * np.ones_like(Y))))
	return vel_comb