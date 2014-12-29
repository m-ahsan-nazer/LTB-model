###############################################################################
# Purpose is to check the code against the results obtained by 
# J. Grande and L. Perivolaropoulos 
# Phys. Rev. D. 84:023514, 2011 
#
# The density profiles for matter and dark energy, and other model parameters 
# are given here to avoid 
# clutter in the main program arXiv_1103_4143.py
#
###############################################################################
from LTB_housekeeping import c, Mpc, Gpc, ageMpc
from LTB_housekeeping import Integrate

import numpy as np
class GP_MODEL():
	"""
	A class for the LTB models used in J. Grande and L. Perivolaropoulos 
	Phys. Rev. D. 84:023514, 2011. 
	"""
	def __init__(self,
	             OmegaM_in=0.301,OmegaM_out=1., 
	             OmegaX_in = 0.699,OmegaX_out=0.,
	             r0=3.37*Gpc,delta_r=0.35*Gpc,age=13.7*ageMpc):
		
		self.set_params( OmegaM_in, OmegaM_out, 
	                     OmegaX_in, OmegaX_out,
	                     r0, delta_r,age)
		
		from functools import wraps
		self.Integrand_H0overc = Integrate(self.Integrand_H0overc)
		self.Integrand_H0overc.set_options(epsabs=1.49e-16,epsrel=1.49e-12)
		self.Integrand_H0overc.set_limits(0.,1.)
		self.H0overc = np.vectorize(self.H0overc)
	
	def set_params(self,
	               OmegaM_in, OmegaM_out, 
	               OmegaX_in, OmegaX_out,
	               r0, delta_r,age):
		"""
		Set the required parameters for the GP models
		"""
		self.OmegaM_in  = OmegaM_in
		self.OmegaM_out = OmegaM_out 
		self.OmegaX_in  = OmegaX_in
		self.OmegaX_out = OmegaX_out
		#self.OmegaC = 1. - Omega
		
		self.r0 =r0  #2.5*Gpc #3.5 #0.33 #0.3-4.5 units Gpc
		self.delta_r = delta_r #0.2*r0 # 0.1r0-0.9r0
		self.t0 = age
		
	def OmegaM(self,r):
		"""
		Eq. (2.27) in http://arxiv.org/abs/0802.1523
		Notation here is 2M(r) \equiv F(r)
		Omega_M(r) fixes M(r) via M(r) = H0(r)**2 Omega_M(r) r**3
		"""
		return_me = self.OmegaM_out+(self.OmegaM_in-self.OmegaM_out)*(1. - 
		np.tanh(0.5*(r-self.r0)/self.delta_r))/(1.+np.tanh(0.5*self.r0/self.delta_r))
		return return_me

	def d_OmegaM_dr(self,r):
		"""
		http://arxiv.org/abs/0802.1523
		Notation here is 2M(r) \equiv F(r)
		Omega_M(r) fixes M(r) via M(r) = H0(r)**2 Omega_M(r) r**3
		evaluates partial derivative of Omega_M(r) w.r.t r
		[d_Omega_M_dr]=Mpc^-1
		"""
		return_me = -(1./2.)*(self.OmegaM_in-self.OmegaM_out)*(1. - 
		np.tanh((1./2.)*(r-self.r0)/self.delta_r)**2)/(self.delta_r*(1. + 
		np.tanh((1./2.)*self.r0/self.delta_r)))
		return return_me

	def OmegaX(self,r):
		"""
		Eq. (2.28) in http://arxiv.org/abs/1103.4143
		"""
		return_me = 1. - self.OmegaM(r)
		#return_me = self.OmegaX_out+(self.OmegaX_in-self.OmegaX_out)*(1. - 
		#np.tanh(0.5*(r-self.r0)/self.delta_r))/(1.+np.tanh(0.5*self.r0/self.delta_r))
		return return_me

	def d_OmegaX_dr(self,r):
		"""
		Evaluates partial derivative of OmegaX(r) w.r.t r
		[d_Omega_M_dr]=Mpc^-1
		"""
		return_me = -self.d_OmegaM_dr(r)
		#return_me = -(1./2.)*(self.OmegaX_in-self.OmegaX_out)*(1. - 
		#np.tanh((1./2.)*(r-self.r0)/self.delta_r)**2)/(self.delta_r*(1. + 
		#np.tanh((1./2.)*self.r0/self.delta_r)))
		return return_me
	
	#def Integrand_H0overc(self,RoverR0,OmegaM,OmegaX,OmegaC):
	def Integrand_H0overc(self,RoverR0,r):
		"""
		Eq. (2.29) in http://arxiv.org/abs/1103.4143
		[H0overc]=Mpc^-1
		"""
		OmegaM = self.OmegaM(r)
		OmegaX = self.OmegaX(r)
		OmegaC = 1. - OmegaM - OmegaX
		OmegaC = 0.
		#print "OmegaC is ", OmegaC
		return np.sqrt(RoverR0) / np.sqrt(OmegaM + OmegaX*RoverR0**3 + OmegaC*RoverR0)

	#def H0overc(self,OmegaM,OmegaX,OmegaC):
	def H0overc(self,r):
		"""
		Eq. (2.29) in http://arxiv.org/abs/1103.4143
		[H0overc]=Mpc^-1
		"""
		return_me = self.Integrand_H0overc.integral(r)/self.t0
		return return_me

	def M(self,r):
		"""
		[LTB_M] = Mpc
		F(r) = H0(r)^2 OmegaM(r) A0(r)^3 = 2M
		"""
		return_me = self.H0overc(r)**2*self.OmegaM(r)*r**3 / 2.
		return return_me


	def E(self,r):
		return 0.

	def d_E_dr(self,r):
		return 0.
	
	def p(self,r):
		"""
		OmegaX = (-8piG / 3) p/H0^2
		return: Redefined p \equiv -pi G p
		[-pi G p ] = Mpc^-2
		"""
		return self.H0overc(r)**2*self.OmegaX(r)*3./8.
	

def get_GP_angles(ell,bee,ell_d = 264.14, bee_d = 48.26):
	"""
	For a fixed choice of coordinates of centre of the universe sets the 
	angles along which the geodesics will be solved. Centre direction corresponds 
	to the dipole axis hence d subscript for its declination and right ascension. 
	The following relationships between right ascention, declination and 
	theta, phi coordinates of the LTB model hold:
	theta = pi/2- bee where -pi/2 <= bee <= pi/2
	phi   = ell       where 0 <= ell < 2pi
	theta:
		is in degrees. And in physics convention 0 <= theta <= pi . 
	phi:
		is in degrees. And in physics convention 0 <= phi <= 2pi . 
	gammas:
	       the angle between the tangent vector to the geodesic and a unit 
	       vector pointing from the observer to the void centre.
	ell_d, bee_d:
	      are the right ascension and declination of the void center i.e dipole
	returns:
	        dec, ras, gamma in radians
	"""
	#convert from galactic coordinates to spherical polar coordinates in radians
	theta   = (90. - bee)*np.pi/180.
	phi     = ell*np.pi/180.
	theta_d = (90. - bee_d)*np.pi/180.
	phi_d   = ell_d*np.pi/180.
	#a general point is p = ( sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
	#and the p_d = ( sin(theta_d)cos(phi_d), sin(theta_d)sin(phi_d), cos(theta_d))
	#cos(gamma) = p_d dot p
	#Let p = (px, py, pz), p_d = (px_d, py_d, pz_d)
	px = np.sin(theta)*np.cos(phi); px_d = np.sin(theta_d)*np.cos(phi_d)
	py = np.sin(theta)*np.sin(phi); py_d = np.sin(theta_d)*np.sin(phi_d)
	pz = np.cos(theta)            ; pz_d = np.cos(theta_d)
	
	gamma = np.arccos(px*px_d + py*py_d + pz*pz_d)
	
	return  phi, theta, gamma 

