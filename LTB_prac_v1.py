####################################################
# The constrained GBH model, pg 9 of arXiv:0802.1523
# v1 to be used to check that units are correct, and 
# known results are reproduced.
#
#

from __future__ import division
import numpy as np
from LTB_Sclass_v1 import LTB_ScaleFactor
from LTB_Sclass_v1 import LTB_geodesics

c = 299792458. #ms^-1
Mpc = 1.
Gpc = 1e3*Mpc
H_in = 0.73 #0.5-0.85 units km s^-1 Mpc^-1
Hoverc_in = H_in*1e5/c #units of Mpc^-1
H_out = 0.6 #0.3-0.7 units km s^-1 Mpc^-1
Hoverc_out = H_out*1e5/c #units of Mpc^-1
H_not = 0.64 #0.5-0.95 units km s^-1 Mpc^-1
Hoverc_not = H_not*1e5/c #units of Mpc^-1
Omega_in = 0.13 #0.05-0.35
#if Lambda is nonzero check equations for correct units. [Lambda]=[R]^-2
Lambda = 0. #0.7
Omega_out = 1. - Lambda
r0 = 2.5*Gpc #3.5 #0.33 #0.3-4.5 units Gpc
delta_r = 0.64*r0 # 0.1r0-0.9r0
# r shall be in units of Mpc
# As a point of reference in class the conformal age of 13894.100411 Mpc 
# corresponds to age = 13.461693 Gyr
# age = 15. billion years
#     = 15. *10**9*(365.25*24.*60.*60.) s
#     = 15. *10**9*(365.25*24.*60.*60.)* 299792458. m
#     = 15. *10**9*(365.25*24.*60.*60.)* 299792458. *3.24077929*10**(-23) Mpc
#     = 15. * 306.60139383811764 Mpc
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
	#return_me = H_out+(H_in-H_out)*(1.-np.tanh((1./2.)*(r-r0)/delta_r))/(1.+np.tanh((1./2.)*r0/delta_r))
	#constrainted GBH model
	return_me = Hoverc_not*( 1./(1.-Omega_M(r)) - Omega_M(r)/(1.-Omega_M(r))**1.5*np.arcsinh(1./Omega_M(r)-1.) ) 
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
	#return_me = -(1./2.)*(H_in-H_out)*(1.-np.tanh((1./2.)*(r-r0)/delta_r)**2)/(delta_r*(1.+np.tanh((1./2.)*r0/delta_r)))
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
	return_me = r**2.*( H0overc(r)**2 - 2.*LTB_M(r)/r**3 - Lambda/3. )/2.
	#the above should produce the same result as the expression used for 
	# k(r) in the paper given below. uncomment and use either one.
	#return_me = -0.5*H0overc(r)**2*(Omega_M(r)-1.)*r**2
	return return_me

def dLTB_E_dr(r):
	"""
	[dLTB_E_dr]=Mpc^-1
	Note:
	     See LTB_E(r) for the two choices given below
	"""
	return_me = 2.*LTB_E(r)/r + r**2 * (H0overc(r)*d_H0overc_dr(r) - dLTB_M_dr(r)/r**3 + 3.*LTB_M(r)/r**4)
	#return_me = -d_H0overc_dr(r)*H0overc(r)*(Omega_M(r)-1.)*r**2 \
	#            -0.5*H0overc(r)**2*d_Omega_M_dr(r)*r**2 \
	#            -H0overc(r)**2*(Omega_M(r)-1.)*r
	return return_me



LTB_model0 =  LTB_ScaleFactor(Lambda=Lambda,LTB_E=LTB_E, LTB_Edash=dLTB_E_dr,\
                              LTB_M=LTB_M, LTB_Mdash=dLTB_M_dr)



#one, two, three, four, five, six = LTB_model0(r_loc=0.00668343917569,num_pt=6000)

#print one, two, three, four, five, six
#for i in range(len(one)):
#	print one[i], two[i], three[i], four[i], five[i], six[i]


#r_vector = np.concatenate((np.logspace(np.log10(1e-3),np.log10(1.),num=40,endpoint=False),
#                       np.linspace(1.,50.,num=60,endpoint=True)))
r_vector = np.concatenate((np.logspace(np.log10(1e-3),np.log10(1.),num=10,endpoint=False),
                       np.linspace(1.,10.*Gpc,num=90,endpoint=True)))
num_pt = 1000 #6000
r_vec = np.zeros((len(r_vector),num_pt))
t_vec = np.zeros((len(r_vector),num_pt))
R_vec = np.zeros((len(r_vector),num_pt))
Rdot_vec = np.zeros((len(r_vector),num_pt))
Rdash_vec = np.zeros((len(r_vector),num_pt))
Rdotdot_vec = np.zeros((len(r_vector),num_pt)) 
Rdashdot_vec = np.zeros((len(r_vector),num_pt))
for i, r_loc in zip(range(len(r_vector)),r_vector):
	print r_loc
	t_vec[i,:], R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdotdot_vec[i,:], \
	Rdashdot_vec[i,:] = LTB_model0(r_loc=r_loc,num_pt=num_pt)
	r_vec[i,:] = r_vec[i,:] + r_loc


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
LTB_geodesics_model0 =  LTB_geodesics(R_spline=spR,Rdot_spline=spRdot,Rdash_spline=spRdash,Rdashdot_spline=spRdashdot)

def fsolve_LTB_age(t,r):
	return spR.ev(r,t)-r

print "checking that age is the same "
for r_val in r_vector:
	from scipy.optimize import brentq
	age, junk = brentq(f=fsolve_LTB_age,a=1e-4,b=30.*ageMpc,args=(r_val,),disp=True,full_output=True) 
	print " age = ", age, " r_val = ", r_val, " junk ", junk.converged
print "so what now"
#factor = 0.5
#LTB_geodesics_model0(rp=1.,tp=17.*ageMpc,ktp=-1.,alpha=np.pi*factor)
#LTB_geodesics_model0 =  LTB_geodesics(R_spline=spR,Rdot_spline=spRdot,Rdash_spline=spRdash,Rdashdot_spline=spRdashdot)
for factor in np.linspace(0.,2,21,endpoint=False):
	#LTB_geodesics_model0 =  LTB_geodesics(R_spline=spR,Rdot_spline=spRdot,Rdash_spline=spRdash,Rdashdot_spline=spRdashdot)
	from scipy.optimize import brentq
	loc = 2.5*Gpc
	age, junk = brentq(f=fsolve_LTB_age,a=1e-4,b=30.*ageMpc,args=(loc,),disp=True,full_output=True)
	print "factor and age ", factor, age, loc, junk.converged
	LTB_geodesics_model0(rp=loc,tp=age,ktp=1.,alpha=np.pi*factor)

#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt
#xx , yy = np.meshgrid(r_vector,t_vec[0,:])
#zz = np.zeros((len(r_vector),len(t_vec)))
#fig = plt.figure() # 
#print 'here'
#for rs, ts in zip(r_vec,t_vec):
#	zz[sp.ev(rs,ts)]
#print 'and over'
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface( t_vec[0,:],r_vector,R_vec)#, rstride=10, cstride=10)
#plt.show()

#t_vector = np.logspace(np.log10(1e-5),np.log10(2.0),num=1000,endpoint=True)
#r_grid, t_grid = np.meshgrid(r_vector,t_vector)

#R_grid = sciI.griddata((r_vec, t_vec), R_vec, (r_grid,t_grid), method='cubic')
#ax = plt.axes(projection='3d')
#for i in range(len(r_vector)):
#	#ax.zaxis.set_scale('log') #set_zscale('log')
#	ax.plot(np.zeros(num_pt)+r_vector[i],t_vec[i,:],np.log10(dLTB_M_dr(r_vector[i])*2./R_vec[i,:]**2/Rdash_vec[i,:]),'-')
#	#ax.scatter(np.zeros(num_pt)+r_vector[i],t_vec[i,:],dLTB_M_dr(r_vector[i])*2./R_vec[i,:]**2/Rdash_vec[i,:],c=R_vec[i,:])
#plt.show()
#f = sciI.interp2d(r_vec,t_vec, R_vec,kind='cubic')
#plt.plot(r_vector,t_vec)
#plt.show()
#np.savetxt("zshit.out",t_vec,fmt='%1.10e')





