####################################################
# The constrained GBH model, pg 9 of arXiv:0802.1523
# v1 to be used to check that units are correct, and 
# known results are reproduced.
#
#

from __future__ import division
import numpy as np
from LTB_Sclass_v2 import LTB_ScaleFactor
from LTB_Sclass_v2 import LTB_geodesics
from LTB_housekeeping import *

c = 299792458. #ms^-1
Mpc = 1.
Gpc = 1e3*Mpc
H_in = 0.73 #0.5-0.85 units km s^-1 Mpc^-1
Hoverc_in = H_in*1e5/c #units of Mpc^-1
H_out = 0.9 #0.3-0.7 units km s^-1 Mpc^-1
Hoverc_out = H_out*1e5/c #units of Mpc^-1
H_not = 0.7 #0.5-0.95 units km s^-1 Mpc^-1
Hoverc_not = H_not*1e5/c #units of Mpc^-1
Omega_in = 0.33 #0.05-0.35
#if Lambda is nonzero check equations for correct units. [Lambda]=[R]^-2
Lambda = 0. #0.7
Omega_out = 0.99999 - Lambda
r0 = 50./H_out #2.5*Gpc #3.5 #0.33 #0.3-4.5 units Gpc
delta_r = 15. #0.2*r0 # 0.1r0-0.9r0
# r shall be in units of Mpc
# As a point of reference in class the conformal age of 13894.100411 Mpc 
# corresponds to age = 13.461693 Gyr
# age = 15. billion years
#     = 15. *10**9*(365.25*24.*60.*60.) s
#     = 15. *10**9*(365.25*24.*60.*60.)* 299792458. m
#     = 15. *10**9*(365.25*24.*60.*60.)* 299792458. *3.24077929*10**(-23) Mpc
#     = 15. * 306.60139383811764 Mpc

###################
#wiltshire notation
# r0 = R = 20 to 60  (h^{-1] Mpc)
delta_w = -0.95 #-0.95 to -1
#delta_r =  w = 2.5  , 5.,  10.,  15.
# add subscript w to all quantities wiltshire
###################

ageMpc = 306.60139383811764
def Omega_M(r):
	"""
	http://arxiv.org/abs/0802.1523
	Notation here is 2M(r) \equiv F(r)
	Omega_M(r) fixes M(r) via M(r) = H0(r)**2 Omega_M(r) r**3
	"""
	return_me = Omega_out+(Omega_in-Omega_out)*(1.-np.tanh((1/2.)*(r-r0)/delta_r))/(1.+np.tanh((1/2.)*r0/delta_r))
	return return_me

def d_Omega_M_dr(r):
	"""
	http://arxiv.org/abs/0802.1523
	Notation here is 2M(r) \equiv F(r)
	Omega_M(r) fixes M(r) via M(r) = H0(r)**2 Omega_M(r) r**3
	evaluates partial derivative of Omega_M(r) w.r.t r
	[d_Omega_M_dr]=Mpc^-1
	"""
	return_me = -(1./2.)*(Omega_in-Omega_out)*(1.-np.tanh((1./2.)*(r-r0)/delta_r)**2)/(delta_r*(1.+np.tanh((1./2.)*r0/delta_r)))
	return return_me

def H0overc(r):
	"""
	http://arxiv.org/abs/0802.1523
	Notation here is 2M(r) \equiv F(r)
	H(r) fixes M(r) via M(r) = H0(r)**2 Omega_M(r) r**3
	[H0overc]=Mpc^-1
	"""
	### General GBH model
	#return_me = Hoverc_out+(Hoverc_in-Hoverc_out)*(1.-np.tanh((1./2.)*(r-r0)/delta_r))/(1.+np.tanh((1./2.)*r0/delta_r))
	#constrainted GBH model
	return_me = Hoverc_not*( 1./(1.-Omega_M(r)) - Omega_M(r)/(1.-Omega_M(r))**1.5*np.arcsinh(np.sqrt(1./Omega_M(r)-1.)) ) 
	return return_me

def d_H0overc_dr(r):
	"""papers
	http://arxiv.org/abs/0802.1523
	Notation here is 2M(r) \equiv F(r)
	H(r) fixes M(r) via M(r) = H0(r)**2 Omega_M(r) r**3
	evaluates partial derivative of H(r) w.r.t r
	[d_H0overc_dr]=Mpc^-2
	"""
	### General GBH model
	#return_me = -(1./2.)*(Hoverc_in-Hoverc_out)*(1.-np.tanh((1./2.)*(r-r0)/delta_r)**2)/(delta_r*(1.+np.tanh((1./2.)*r0/delta_r)))
	#constrainted GBH model
	return_me = 0.5*d_Omega_M_dr(r)/(1.-Omega_M(r))/Omega_M(r) * (Omega_M(r)*H0overc(r)+2.*H0overc(r)-2*Hoverc_not)
	return return_me


def LTB_M(r):
	"""
	[LTB_M] = Mpc
	"""
	return_me = H0overc(r)**2*Omega_M(r)*r**3 / 2.
	return return_me

def dLTB_M_dr(r):
	"""
	[dLTB_M_dr] is dimensionless
	"""
	return_me = H0overc(r)*d_H0overc_dr(r)*Omega_M(r)*r**3 + \
	            H0overc(r)**2*d_Omega_M_dr(r)*r**3 /2. + \
	            3./2.*H0overc(r)**2*Omega_M(r)*r**2
	return return_me

def LTB_E(r):
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
	return_me = -0.5*H0overc(r)**2*(Omega_M(r)-1.)*r**2
	return return_me

def dLTB_E_dr(r):
	"""
	[dLTB_E_dr]=Mpc^-1
	Note:
	     See LTB_E(r) for the two choices given below
	"""
	#return_me = 2.*LTB_E(r)/r + r**2 * (H0overc(r)*d_H0overc_dr(r) - dLTB_M_dr(r)/r**3 + 3.*LTB_M(r)/r**4)
	return_me = -d_H0overc_dr(r)*H0overc(r)*(Omega_M(r)-1.)*r**2 \
	            -0.5*H0overc(r)**2*d_Omega_M_dr(r)*r**2 \
	            -H0overc(r)**2*(Omega_M(r)-1.)*r
	return return_me

def dLTBw_M_dr(r):
	"""
	[LTB_M] = Mpc
	"""
	return_me = 0.75*Omega_out*Hoverc_out**2*r**2 * (2.+delta_w - delta_w* 
	                                                 np.tanh((r-r0)/delta_r))
	return return_me

#fist make a spline and use it to calcuate the integral than make a second spline
# so that it is computationally less expensive
from scipy.interpolate import UnivariateSpline as sp1d
#rw = np.concatenate((np.logspace(np.log10(1e-10),np.log10(1.),num=500,endpoint=False),
#                       np.linspace(1.,20.*Gpc,num=500,endpoint=True)))

rw = np.concatenate((np.logspace(np.log10(1e-10),np.log10(1.),num=500,endpoint=False),
                       np.linspace(1.,300.,num=500,endpoint=False)))

rw = np.concatenate((rw,np.linspace(300.,20.*Gpc,num=300,endpoint=True)))


spdLTBw_M_dr = sp1d(rw, dLTBw_M_dr(rw), s=0) #dLTBw_M_dr(rw), s=0)
spdLTBw_M_dr_int = spdLTBw_M_dr.antiderivative()
Mw = spdLTBw_M_dr_int(rw) #- spdLTBw_M_dr_int(rw[0])

spMw = sp1d(rw,Mw,s=0)
print "spMw(6) ", spMw(6.)
def LTBw_M(r):
	"""
	[LTB_M] = Mpc
	"""
	return spMw(r)

from matplotlib import pylab as plt
fig = plt.figure()
plt.plot(rw, dLTBw_M_dr(rw),label="dM_dr")
#plt.xscale('log')
#plt.yscale('symlog')
plt.legend(loc='best')
fig = plt.figure()
#plt.plot(rw,LTBw_M(rw)/LTB_M(rw),label="M(r)")
plt.plot(rw,LTBw_M(rw),label="M(r)")
plt.legend(loc='best')
plt.yscale('log')
plt.xscale('log')
fig = plt.figure()
plt.plot(rw,dLTBw_M_dr(rw)*2/rw**2,label="rho(r)")
plt.xscale('log')
plt.legend(loc='best')
plt.show()

LTB_model0 =  LTB_ScaleFactor(Lambda=Lambda,LTB_E=LTB_E, LTB_Edash=dLTB_E_dr,\
                              LTB_M=LTB_M, LTB_Mdash=dLTB_M_dr)



#one, two, three, four, five, six = LTB_model0(r_loc=0.00668343917569,num_pt=6000)

#print one, two, three, four, five, six
#for i in range(len(one)):
#	print one[i], two[i], three[i], four[i], five[i], six[i]


#r_vector = np.concatenate((np.logspace(np.log10(1e-3),np.log10(1.),num=40,endpoint=False),
#                       np.linspace(1.,50.,num=60,endpoint=True)))
r_vector = np.concatenate((np.logspace(np.log10(1e-4),np.log10(1.),num=30,endpoint=False),
                       np.linspace(1.,20.*Gpc,num=90,endpoint=True)))
num_pt = 1000 #6000

#global r_vec, t_vec, R_vec, Rdot_vec, Rdash_vec, Rdotdot_vec, Rdashdot_vec

r_vec = np.zeros((len(r_vector),num_pt))
t_vec = np.zeros((len(r_vector),num_pt))
R_vec = np.zeros((len(r_vector),num_pt))
Rdot_vec = np.zeros((len(r_vector),num_pt))
Rdash_vec = np.zeros((len(r_vector),num_pt))
Rdotdot_vec = np.zeros((len(r_vector),num_pt)) 
Rdashdot_vec = np.zeros((len(r_vector),num_pt))

#for i, r_loc in zip(range(len(r_vector)),r_vector):
#	print r_loc
#	t_vec[i,:], R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdotdot_vec[i,:], \
#	Rdashdot_vec[i,:] = LTB_model0(r_loc=r_loc,num_pt=num_pt)
#	r_vec[i,:] = r_vec[i,:] + r_loc

def r_loop(i,r_loc):
	print "i, r_loc", i, r_loc
	#t_vec[i,:], R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdotdot_vec[i,:], \
	#Rdashdot_vec[i,:] = LTB_model0(r_loc=r_loc,num_pt=num_pt)
	#r_vec[i,:] = r_vec[i,:] + r_loc
	#print "t_vec ", t_vec[i,0], t_vec[i,-1], R_vec[i,0],R_vec[i,-1]
	return LTB_model0(r_loc=r_loc,num_pt=num_pt)



from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import multiprocessing as mp
num_cores = mp.cpu_count()-1
#Parallel(n_jobs=6)(delayed(r_loop)(i, r_loc) for i, r_loc in zip(range(len(r_vector)),r_vector))	
r = Parallel(n_jobs=num_cores,verbose=0)(delayed(r_loop)(i, r_loc) for i, r_loc in zip(range(len(r_vector)),r_vector))
#r = Parallel(n_jobs=num_cores,verbose=0)(delayed(LTB_model0)(r_loc=r_loc,num_pt=num_pt) for r_loc in r_vector)

i = 0
for tup in r:
	t_vec[i,:], R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdotdot_vec[i,:], \
	Rdashdot_vec[i,:] = tup
	i = i + 1

for i, r_loc in zip(range(len(r_vector)),r_vector):
	r_vec[i,:] = r_vec[i,:]+r_loc

#print "tyep of r0", type(r[0]), len(r[0])
#import sys
#sys.exit("done all loops")

#from matplotlib import pylab as plt
#plt.plot(t_vec[0,:],t_vec[1,:])
#print "final ", t_vec[0,0], t_vec[0,-1], R_vec[0,0],R_vec[0,-1]
#plt.plot(R_vec[0,:],R_vec[1,:])
#plt.show()

from matplotlib import pylab as plt
from scipy import interpolate as sciI
from mpl_toolkits.mplot3d import Axes3D

sp = sciI.RectBivariateSpline(r_vector,t_vec[0,:],R_vec,s=0)
spdr = sciI.RectBivariateSpline(r_vector,t_vec[0,:],Rdash_vec,s=0)
tis = t_vec[0,33]
ris = r_vector[44]
Ris = R_vec[44,33]
Rdashis = Rdash_vec[44,33]

print tis, ris, Ris, sp.ev(ris,tis), Rdashis, spdr.ev(ris,tis), 'and ', sp(ris,tis,dx=1),'dog',sp.ev(ris,tis,dx=1),'dog'
 
spR = sciI.RectBivariateSpline(r_vector,t_vec[0,:],R_vec,s=0)
spRdot = sciI.RectBivariateSpline(r_vector,t_vec[0,:],Rdot_vec,s=0)
spRdash = sciI.RectBivariateSpline(r_vector,t_vec[0,:],Rdash_vec,s=0)
spRdashdot = sciI.RectBivariateSpline(r_vector,t_vec[0,:],Rdashdot_vec,s=0)

t0 = t_vec[33,-1]
print "t0 ", t_vec[33,-1],t_vec[97,-1], spR.ev(0.01,t0), spR.ev(23.,t0), spR.ev(2*Gpc,t0)
print "hubble ", spRdot(0.01,t0)/spR.ev(0.01,t0),spRdot(23.,t0)/spR.ev(23.,t0),spRdot(2*Gpc,t0)/spR.ev(2*Gpc,t0)
print "hubble ", H0overc(0.01), H0overc(23.),H0overc(2*Gpc)

LTB_geodesics_model0 =  LTB_geodesics(R_spline=spR,Rdot_spline=spRdot,Rdash_spline=spRdash,Rdashdot_spline=spRdashdot,LTB_E=LTB_E, LTB_Edash=dLTB_E_dr)

def fsolve_LTB_age(t,r): #fsolve_LTB_age(r,t): #
	return spR.ev(r,t)-r# spRdot.ev(r,t)/spR(r,t)-H0overc(r) 
model_age = 0.
#print "checking that age is the same "
for r_val in r_vector:
	from scipy.optimize import brentq
	age, junk = brentq(f=fsolve_LTB_age,a=1e-4,b=30.*ageMpc,args=(r_val,),disp=True,full_output=True) 
	model_age = age
	###print "model age = ", age, "in giga years ", age/ageMpc, "r_val ", r_val, " junk ", junk.converged
	###print "hubble and ratio ", spRdot.ev(r_val,age)/spR.ev(r_val,age), H0overc(r_val)


print "now checking the quad decorator"


@Integrate
def LTB_t(RoverR0,twoE,twoM,Lambda_over3):
	#return 1./np.sqrt(twoE + twoM/RoverR0 + Lambda_over3 * RoverR0**2)
	return np.sqrt(RoverR0)/np.sqrt(twoE*RoverR0 + twoM + Lambda_over3 * RoverR0**3)

LTB_t.set_options(epsabs=1.49e-16,epsrel=1.49e-12)
LTB_t.set_limits(0.,1.)
@Findroot
def LTB_2E_Eq(twoE_over_r3,twoM_over_r3,Lambda_over3):
	return model_age*1e-3 - LTB_t.integral(twoE_over_r3,twoM_over_r3,Lambda_over3) #*1.e-3

r = 0.01 #2*ageMpc
t = 1.*ageMpc
#print "integrand ", LTB_t(spR.ev(r,t)/r,2.*LTB_E(r)/r**2,2.*LTB_M(r)/r**3,0.)
print "integral ", LTB_t.integral(2.*LTB_E(r)/r**2*1e6,2.*LTB_M(r)/r**3*1e6,0.), LTB_t.abserr, model_age
print "checking LTB_E_Eq ", LTB_2E_Eq(2.*LTB_E(r)/r**2*1e6,2.*LTB_M(r)/r**3*1e6,0.) 

LTB_2E_Eq.set_options(xtol=4.4408920985006262e-16,rtol=4.4408920985006262e-15)
LTB_2E_Eq.set_bounds(0.,2.)
print "analytic E ", 2.*LTB_E(r)/r**2*1e6
print "E from integral ", LTB_2E_Eq.root(2.*LTB_M(r)/r**3*1e6,0.), 'and converged ', LTB_2E_Eq.converged
print "% error ", (LTB_2E_Eq.root(2.*LTB_M(r)/r**3*1e6,0.)/(2.*LTB_E(r)/r**2*1e6)-1.)*100
#print "% abserr ", LTB_2E_Eq.abserr

E = np.zeros(len(r_vector))
#serial loop
#i = 0
#for r in r_vector:
#	E[i] = LTB_2E_Eq.root(2.*LTB_M(r)/r**3,0.)
#	i = i + 1

def E_loop(r,Lambda):
	return LTB_2E_Eq.root(2.*LTB_M(r)/r**3*1e6,Lambda)


E_vec = Parallel(n_jobs=num_cores,verbose=0)(delayed(E_loop)(r,0.) for r in r_vector)
E_vec = np.asarray(E_vec)

i = 0
for r in r_vector:
	print E_vec[i]/1e6, 2.*LTB_E(r)/r**2, (E_vec[i]/1e6)/( 2.*LTB_E(r)/r**2)
	i = i + 1
print "all done"



