####################################################
# Good plots and numbers for comparison are found here
#http://arxiv.org/abs/astro-ph/0607334
# "CMB anisotropies seen by an off-center observer in a spherically symmetric inhomogeneous universe"
#The models are properly explained in a previous paper here
#http://arxiv.org/abs/astro-ph/0512006v2
# "An inhomogeneous alternative to dark energy?"
# To be used to check that units are correct, and 
# known results are reproduced.
#
#

from __future__ import division
import numpy as np
from LTB_Sclass_v2 import LTB_ScaleFactor
from LTB_Sclass_v2 import LTB_geodesics

c = 299792458. #ms^-1
Mpc = 1.
Gpc = 1e3*Mpc

hout = 0.51 #0.5-0.95 units km s^-1 Mpc^-1
H0overc = hout*1e5/c #units of Mpc^-1

r0 = 1.34*Gpc #3.5 #0.33 #0.3-4.5 units Gpc
delta_r = 0.4*r0 # 0.1r0-0.9r0
delta_alpha = 0.9
alpha_0 = 1.
Lambda = 0.
# r shall be in units of Mpc
# As a point of reference in class the conformal age of 13894.100411 Mpc 
# corresponds to age = 13.461693 Gyr
# age = 15. billion years
#     = 15. *10**9*(365.25*24.*60.*60.) s
#     = 15. *10**9*(365.25*24.*60.*60.)* 299792458. m
#     = 15. *10**9*(365.25*24.*60.*60.)* 299792458. *3.24077929*10**(-23) Mpc
#     = 15. * 306.60139383811764 Mpc
ageMpc = 306.60139383811764


def LTB_M(r):
	"""
	[LTB_M] = Mpc
	alpha(r) = 2*M(r)
	"""
	return_me = H0overc**2*r**3*(alpha_0-delta_alpha*(0.5-0.5*np.tanh((r-r0)/2./delta_r))) / 2.
	return return_me

def dLTB_M_dr(r):
	"""
	[dLTB_M_dr] is dimensionless
	"""
	alpha_r = 2.*LTB_M(r)
	return_me = (0.25*H0overc**2*delta_alpha/delta_r*(1.-np.tanh((r-r0)/2./delta_r)**2) + 3.*alpha_r/r**4) * r**3 / 2.
	return return_me

def LTB_E(r):
	"""
	E(r) in Eq. (2.1) of "Structures in the Universe by Exact Methods"
	2E(r) \equiv -k(r) in http://arxiv.org/abs/0802.1523
	[LTB_E] is dimensionless
	beta(r) = 2*E(r)
	"""
	# Since a gauge condition is used i.e. R(t0,r) =r the expression 
	#below is always true 
	#return_me = r**2.*( H0overc(r)**2 - 2.*LTB_M(r)/r**3 - Lambda/3. )/2.
	#the above should produce the same result as the expression used for 
	
	return_me = H0overc**2*r**2*(1.-alpha_0 +delta_alpha*(0.5-0.5*np.tanh((r-r0)/2./delta_r))) / 2.
	return return_me

def dLTB_E_dr(r):
	"""
	[dLTB_E_dr]=Mpc^-1
	Note:
	     See LTB_E(r) for the two choices given below
	"""
	beta_r = 2.*LTB_E(r)
	#return_me = 2.*LTB_E(r)/r + r**2 * (H0overc(r)*d_H0overc_dr(r) - dLTB_M_dr(r)/r**3 + 3.*LTB_M(r)/r**4)
	return_me = (-0.25*H0overc**2*delta_alpha/delta_r*(1.-np.tanh((r-r0)/2./delta_r)**2) + 2.*beta_r/r**3) * r**2 / 2.
	return return_me



LTB_model0 =  LTB_ScaleFactor(Lambda=Lambda,LTB_E=LTB_E, LTB_Edash=dLTB_E_dr,\
                              LTB_M=LTB_M, LTB_Mdash=dLTB_M_dr)



#one, two, three, four, five, six = LTB_model0(r_loc=0.00668343917569,num_pt=6000)

#print one, two, three, four, five, six
#for i in range(len(one)):
#	print one[i], two[i], three[i], four[i], five[i], six[i]


#r_vector = np.concatenate((np.logspace(np.log10(1e-3),np.log10(1.),num=40,endpoint=False),
#                       np.linspace(1.,50.,num=60,endpoint=True)))
r_vector = np.concatenate((np.logspace(np.log10(1e-4),np.log10(1.),num=30,endpoint=False),
                       np.linspace(1.,20.*Gpc,num=90,endpoint=True)))#90
num_pt = 2000 #6000

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
num_cores = mp.cpu_count()
#Parallel(n_jobs=6)(delayed(r_loop)(i, r_loc) for i, r_loc in zip(range(len(r_vector)),r_vector))	
#r = Parallel(n_jobs=num_cores,verbose=0)(delayed(r_loop)(i, r_loc) for i, r_loc in zip(range(len(r_vector)),r_vector))
r = Parallel(n_jobs=num_cores,verbose=0)(delayed(LTB_model0)(r_loc=r_loc,num_pt=num_pt) for r_loc in r_vector)

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
print "hubble ", H0overc #(0.01), H0overc(23.),H0overc(2*Gpc)

LTB_geodesics_model0 =  LTB_geodesics(R_spline=spR,Rdot_spline=spRdot,Rdash_spline=spRdash,Rdashdot_spline=spRdashdot,LTB_E=LTB_E, LTB_Edash=dLTB_E_dr,num_pt=1770)

def fsolve_LTB_age(t,r): #fsolve_LTB_age(r,t): #
	return spR.ev(r,t)-r# spRdot.ev(r,t)/spR(r,t)-H0overc(r) 
model_age = 0.
#print "checking that age is the same "
for r_val in r_vector:
	from scipy.optimize import brentq
	age, junk = brentq(f=fsolve_LTB_age,a=1e-4,b=30.*ageMpc,args=(r_val,),disp=True,full_output=True) 
	model_age = age
	print "model age = ", age, "in giga years ", age/ageMpc, "r_val ", r_val, " junk ", junk.converged
	print "hubble and ratio ", spRdot.ev(r_val,age)/spR.ev(r_val,age), H0overc #(r_val)

def get_angles():
	"""
	For a fixed choice of coordinates of centre of the universe sets the 
	angles along which the geodesics will be solved. Centre direction corresponds 
	to the dipole axis hence d subscript for its declination and right ascension. 
	The following relationships between right ascention, declination and 
	theta, phi coordinates of the LTB model hold:
	theta = pi/2- dec where -pi/2 <= dec <= pi/2
	phi   = ras       where 0 <= ras < 2pi
	gammas:
	       the angle between the tangent vector to the geodesic and a unit 
	       vector pointing from the observer to the void centre.
	returns:
	        dec, ras, gammas
	"""
	pi = np.pi
	dec_d = 29.3*pi/180.
	ras_d = pi+96.4*pi/180.
	
	num_pt = 1 #3 #123
	dec = np.linspace(-pi/2.,pi/2.,num_pt,endpoint=True)
	ras = np.linspace(0.,2.*pi,num_pt,endpoint=False)
	
	dec, ras = np.meshgrid(dec,ras)
	gammas = np.arccos( np.sin(dec)*np.sin(dec_d) + 
	                    np.cos(dec)*np.cos(dec_d)*np.cos(ras-ras_d))
	return dec.flatten(), ras.flatten(), gammas.flatten()
declination, Rascension, gammas  = get_angles()

#for factor in np.linspace(0.,2,21,endpoint=False):
#print "gamma, dec, ras, z, affine_parameter, t, r, p, theta"
#for dec, ras, gamma in zip(declination,Rascension,gammas):
#	loc = 0.07 #1.*Gpc #200 #1.3*Gpc #r_vector[0] #1. #2.5*Gpc
#	a,b,c,d,e,f = LTB_geodesics_model0(rp=loc,tp=13.7*ageMpc,alpha=1.01*np.pi)#gamma)
#	#print "%12.4f  %12.4f  %12.4f  %12.10f  %12.10f  %20.10f  %12.10f  %12.10f %12.10f" %(gamma, dec, ras, 
#	#ans[-1][0],ans[-1][1],ans[-1][2],ans[-1][3],ans[-1][4],ans[-1][5])
#
#	for i in xrange(len(a)):
#		print a[i],b[i],c[i],d[i],e[i],f[i]

num_angles = 10 #200
angles = np.concatenate( (np.linspace(0.,0.99*np.pi,num=100,endpoint=True), 
                        np.linspace(1.01*np.pi,2.*np.pi,num=100,endpoint=False)))

num_z_points = LTB_geodesics_model0.num_pt 
geo_z_vec = LTB_geodesics_model0.z_vec

geo_affine_vec = np.zeros((num_angles,num_z_points))
geo_t_vec = np.zeros((num_angles,num_z_points))
geo_r_vec = np.zeros((num_angles,num_z_points))
geo_p_vec = np.zeros((num_angles,num_z_points))
geo_theta_vec = np.zeros((num_angles,num_z_points))


loc = 200
model_age = 13.7*ageMpc
#serial version
#for i, angle in zip(xrange(num_angles),angles):
#	geo_affine_vec[i,:], geo_t_vec[i,:], geo_r_vec[i,:], geo_p_vec[i,:], \
#	geo_theta_vec[i,:] = LTB_geodesics_model0(rp=loc,tp=model_age,alpha=angle)

#parallel version 1
#num_cores=7
#geos = Parallel(n_jobs=num_cores,verbose=1)(
#delayed(LTB_geodesics_model0)(rp=loc,tp=model_age,alpha=angle) for angle in angles[0:10])

#parallel version 2
def geo_loop(angle):
	return LTB_geodesics_model0(rp=loc,tp=model_age,alpha=angle)
num_cores=7
geos = Parallel(n_jobs=num_cores,verbose=1)(
delayed(geo_loop)(angle=angle) for angle in angles[0:10])

i = 0
for geo_tuple in geos:
	geo_affine_vec[i,:], geo_t_vec[i,:], geo_r_vec[i,:], geo_p_vec[i,:], \
	geo_theta_vec[i,:] = geo_tuple
	i = i + 1

#finally make the 2d splines
sp_affine = sciI.RectBivariateSpline(angles[0:10],geo_z_vec,geo_affine_vec,s=0)
sp_t_vec = sciI.RectBivariateSpline(angles[0:10],geo_z_vec,geo_t_vec,s=0)
sp_r_vec = sciI.RectBivariateSpline(angles[0:10],geo_z_vec,geo_r_vec,s=0)
sp_p_vec = sciI.RectBivariateSpline(angles[0:10],geo_z_vec,geo_p_vec,s=0)
sp_theta_vec = sciI.RectBivariateSpline(angles[0:10],geo_z_vec,geo_theta_vec,s=0)

print "geodesics splines are working"
print sp_r_vec.ev(0.1,1100.),sp_r_vec.ev(0.1,1100.,dx=1)
print sp_theta_vec.ev(0.1,1100.),sp_theta_vec.ev(0.1,1100.,dx=1)

