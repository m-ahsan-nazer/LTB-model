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
from LTB_housekeeping import Integrate

import numpy as np
class GBH_MODEL():
	def __init__(self,H_in=0.6,H_out=0.7,H_not=0.7,Lambda=0.,Omega_in=0.33,
	                  r0=2e3,delta_r=0.3):
		self.c = 299792458. #ms^-1
		self.Mpc = 1.
		self.Gpc = 1e3*self.Mpc
		self.H_in = H_in #0.5-0.85 units km s^-1 Mpc^-1
		self.Hoverc_in = H_in*1e5/self.c #units of Mpc^-1
		self.H_out = H_out = 0.9 #0.3-0.7 units km s^-1 Mpc^-1
		self.Hoverc_out = H_out*1e5/self.c #units of Mpc^-1
		self.H_not = H_not #0.5-0.95 units km s^-1 Mpc^-1
		self.Hoverc_not = H_not*1e5/self.c #units of Mpc^-1
		self.OmegaM_in = Omega_in #0.05-0.35
		#if Lambda is nonzero check equations for correct units. [Lambda]=[R]^-2
		self.Lambda = Lambda  #0.7
		self.OmegaM_out = 0.99999 - self.Lambda
		self.r0 =r0  #2.5*Gpc #3.5 #0.33 #0.3-4.5 units Gpc
		self.delta_r = delta_r #0.2*r0 # 0.1r0-0.9r0
		
		self.set_H0overc()
		set_H0overc.set_options(epsabs=1.49e-16,epsrel=1.49e-12)
		set_H0overc.set_limits(0.,1.)

	def OmegaM(self,r):
		"""
		Eq. (2.27) in http://arxiv.org/abs/0802.1523
		Notation here is 2M(r) \equiv F(r)
		Omega_M(r) fixes M(r) via M(r) = H0(r)**2 Omega_M(r) r**3
		"""
		return_me = self.OmegaM_out+(self.OmegaM_in-self.OmegaM_out)*(1. - 
		np.tanh((1/2.)*(r-self.r0)/self.delta_r))/(1.+np.tanh((1/2.)*self.r0/self.delta_r))
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
		return_me = self.OmegaX_out+(self.OmegaX_in-self.OmegaX_out)*(1. - 
		np.tanh((1/2.)*(r-self.r0)/self.delta_r))/(1.+np.tanh((1/2.)*self.r0/self.delta_r))
		return return_me

	def d_OmegaX_dr(self,r):
		"""
		Evaluates partial derivative of OmegaX(r) w.r.t r
		[d_Omega_M_dr]=Mpc^-1
		"""
		return_me = -(1./2.)*(self.OmegaX_in-self.OmegaX_out)*(1. - 
		np.tanh((1./2.)*(r-self.r0)/self.delta_r)**2)/(self.delta_r*(1. + 
		np.tanh((1./2.)*self.r0/self.delta_r)))
		return return_me
	
	@Integrate
	def set_H0overc(self,RoverR0,OmegaM,OmegaX,OmegaC):
		return 1./t0 / np.sqrt(OmegaM/RoverR0 + OmegaX/RoverR0**2 + OmegaC)

	def H0overc(self,r):
		"""
		Eq. (2.29) in http://arxiv.org/abs/1103.4143
		[H0overc]=Mpc^-1
		"""
		### General GBH model
		#return_me = self.Hoverc_out+(self.Hoverc_in-self.Hoverc_out)*(1. - \
		#np.tanh((1./2.)*(r-self.r0)/self.delta_r))/(1.+np.tanh((1./2.)*self.r0/self.delta_r))
		#constrainted GBH model
		return_me = self.Hoverc_not*( 1./(1.-self.Omega_M(r)) - self.Omega_M(r)/(
		1.-self.Omega_M(r))**1.5*np.arcsinh(np.sqrt(1./self.Omega_M(r)-1.)) ) 
		return return_me

	def d_H0overc_dr(self,r):
		"""
		Evaluates partial derivative of H0overc(r) w.r.t r
		[d_H0overc_dr]=Mpc^-2
		"""
		### General GBH model
		#return_me = -(1./2.)*(self.Hoverc_in-self.Hoverc_out)*(1.-np.tanh((1./2.)* \
		#(r-self.r0)/self.delta_r)**2)/(self.delta_r*(1.+np.tanh((1./2.)*self.r0/self.delta_r)))
		#constrainted GBH model
		return_me = 0.5*self.d_Omega_M_dr(r)/(1.-self.Omega_M(r))/self.Omega_M(r) \
		* (self.Omega_M(r)*self.H0overc(r)+2.*self.H0overc(r)-2*self.Hoverc_not)
		return return_me


	def LTB_M(self,r):
		"""
		[LTB_M] = Mpc
		"""
		return_me = self.H0overc(r)**2*self.Omega_M(r)*r**3 / 2.
		return return_me

	def dLTB_M_dr(self,r):
		"""
		[dLTB_M_dr] is dimensionless
		"""
		return_me = self.H0overc(r)*self.d_H0overc_dr(r)*self.Omega_M(r)*r**3 + \
	            self.H0overc(r)**2*self.d_Omega_M_dr(r)*r**3 /2. + \
	            3./2.*self.H0overc(r)**2*self.Omega_M(r)*r**2
		return return_me

	def LTB_E(self,r):
		"""
		E(r) in Eq. (2.1) of "Structures in the Universe by Exact Methods"
		2E(r) \equiv -k(r) in http://arxiv.org/abs/0802.1523
		[LTB_E] is dimensionless
		"""
		# Since a gauge condition is used i.e. R(t0,r) =r the expression 
		#below is always true 
		#return_me = r**2.*( H0overc(r)**2 - 2.*LTB_M(r)/r**3 - Lambda/3. )/2.
		#the above should produce the same result as the expression used for 
		# k(r) in the paper given below. uncomment and use either one.
		return_me = -0.5*self.H0overc(r)**2*(self.Omega_M(r)-1.)*r**2
		return return_me

	def dLTB_E_dr(self,r):
		"""
		[dLTB_E_dr]=Mpc^-1
		Note:
		     See LTB_E(r) for the two choices given below
		"""
		#return_me = 2.*LTB_E(r)/r + r**2 * (H0overc(r)*d_H0overc_dr(r) - dLTB_M_dr(r)/r**3 + 3.*LTB_M(r)/r**4)
		return_me = -self.d_H0overc_dr(r)*self.H0overc(r)*(self.Omega_M(r)-1.)*r**2 \
	            -0.5*self.H0overc(r)**2*self.d_Omega_M_dr(r)*r**2 \
	            -self.H0overc(r)**2*(self.Omega_M(r)-1.)*r
		return return_me


