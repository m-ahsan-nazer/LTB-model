################################################################################
## Purpose is to check the code against the results obtained by 
## J. Grande and L. Perivolaropoulos 
## Phys. Rev. D. 84:023514, 2011 
################################################################################ 

from __future__ import division
import numpy as np
from LTB_Sclass_v2 import LTB_ScaleFactor
from LTB_Sclass_v2 import LTB_geodesics, sample_radial_coord
from LTB_housekeeping import c, Mpc, Gpc, ageMpc
from LTB_housekeeping import Integrate
from GP_profiles import GP_MODEL

from scipy.interpolate import UnivariateSpline as spline_1d
from scipy.interpolate import RectBivariateSpline as spline_2d
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import multiprocessing as mp

OmegaX_in = 0.699
OmegaM_in = 1. - OmegaX_in
test_GP = GP_MODEL(OmegaM_in=OmegaM_in,OmegaM_out=1., 
	               OmegaX_in = OmegaX_in,OmegaX_out=0.,
	               r0=3.37*Gpc,delta_r=0.35*Gpc,age=13.7*ageMpc)
print test_GP.__doc__

test_r_vals = np.array([0.,0.3,0.9,1.1,10.,2.e3,3.36e3,3.37e3,3.38e3,1e4])
print test_GP.OmegaM(test_r_vals)
print "d_OmegaM_dr"
print test_GP.d_OmegaM_dr(test_r_vals)
print "OmegaX"
print test_GP.OmegaX(test_r_vals)
print "d_OmegaX_dr"
print test_GP.d_OmegaX_dr(test_r_vals)

print "H0overc"
#print test_GP.H0overc(0.2,0.8,0.)
print test_GP.H0overc(0.)
print "d_H0overc_dr"
print test_GP.d_H0overc_dr(3.37*Gpc)

#Now make splines for M(r), dM_dr(r), H(r), dH_dr(r). Splines will be much faster 
#than the direct computation done in GP_profiles. Note that since E(r), dE_dr(r) 
#are zero only M(r), dM_dr(r), H(r), dH_dr(r) and Lambda are needed for solving
#the background and geodesic equations.
# r_vec is used to make splines
# r_vector is used for making the t-r grid on which the background LTB equations 
# are solved.
# r_vec has many more points than r_vector 
r_vec = sample_radial_coord(r0=test_GP.r0,delta_r=test_GP.delta_r,r_init=1e-10,
                            r_max=20*1e3,num_pt1=1000,num_pt2=1000)
#r_vector = sample_radial_coord(r0=r0,delta_r=delta_r,r_init=1e-4,r_max=20*1e3,num_pt1=100,num_pt2=100)
size_r_vec = r_vec.size
r_vector = sample_radial_coord(r0=test_GP.r0,delta_r=test_GP.delta_r,r_init=1e-4,
                              r_max=20*1e3,num_pt1=100,num_pt2=100)
size_r_vector = r_vector.size

M_GP    = test_GP.M(r_vec)
#dMdr_GP = test_GP.d_M_dr(r_vec)
H_GP    = test_GP.H0overc(r_vec)
#dHdr_GP = test_GP.d_H0overc_dr(r_vec)
sp_M = spline_1d(r_vec, M_GP, s=0)
sp_dMdr = sp_M.derivative(1)
sp_dMdr = spline_1d(r_vec,sp_dMdr(r_vec))
sp_H = spline_1d(r_vec, H_GP, s=0)
sp_dHdr = sp_H.derivative(1) 
sp_dHdr = spline_1d(r_vec,sp_dHdr(r_vec))

def sp_E(r):
	return 0.
def sp_dEdr(r):
	return 0.

plt.figure()
plt.plot(r_vec,sp_M(r_vec))
plt.figure()
plt.plot(r_vec,sp_dMdr(r_vec))
plt.figure()
plt.plot(r_vec,sp_H(r_vec))
plt.figure()
plt.plot(r_vec,sp_dHdr(r_vec))
plt.show()

Lambda = test_GP.OmegaX(0.)*3.*test_GP.H0overc(0.)**2
model_age = test_GP.t0
print "Lambda is ", Lambda, test_GP.OmegaX(0.), test_GP.OmegaX(300.)*3.*test_GP.H0overc(300.)**2, test_GP.OmegaX(3*Gpc)*3.*test_GP.H0overc(3*Gpc)**2
model =  LTB_ScaleFactor(Lambda=Lambda,LTB_E=sp_E, LTB_Edash=sp_dEdr,\
                              LTB_M=sp_M, LTB_Mdash=sp_dMdr)

t_num_pt = 1000 #6000
r_vec, t_vec, R_vec, Rdot_vec, Rdash_vec, Rdotdot_vec, Rdashdot_vec, = \
              [np.zeros((size_r_vector,t_num_pt)) for i in xrange(7)]

##serial 
##for i, r_loc in zip(range(len(r_vector)),r_vector):
##	print r_loc
##	t_vec[i,:], R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdotdot_vec[i,:], \
##	Rdashdot_vec[i,:] = LTB_model0(r_loc=r_loc,num_pt=num_pt)
##	r_vec[i,:] = r_vec[i,:] + r_loc

def r_loop(r_loc):
	return model(r_loc=r_loc,t_max=model_age,num_pt=t_num_pt)


num_cores = mp.cpu_count()-3	
r = Parallel(n_jobs=num_cores,verbose=0)(delayed(r_loop)(r_loc) for r_loc in r_vector)

i = 0
for tup in r:
	t_vec[i,:], R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdotdot_vec[i,:], \
	Rdashdot_vec[i,:] = tup
	i = i + 1

t_vector = t_vec[0,:]
spR = spline_2d(r_vector,t_vector,R_vec,s=0)
spRdot = spline_2d(r_vector,t_vector,Rdot_vec,s=0)
spRdash = spline_2d(r_vector,t_vector,Rdash_vec,s=0)
spRdashdot = spline_2d(r_vector,t_vector,Rdashdot_vec,s=0)

print "A basic check on the Hubble expansion "
for r_val in r_vector:
	print "H(r,t0) ", r_val, sp_H(r_val), spRdot.ev(r_val,model_age)/spR.ev(r_val,model_age)
	print "dHdr(r,t0) ", sp_dHdr(r_val), \
	spRdashdot.ev(r_val,model_age)/spR.ev(r_val,model_age) - spRdot.ev(r_val,model_age)*spRdash.ev(r_val,model_age)/spR.ev(r_val,model_age)**2
	print "****"

################################################################################
##******************************************************************************
#model_geodesics =  LTB_geodesics(R_spline=spR,Rdot_spline=spRdot,
#Rdash_spline=spRdash,Rdashdot_spline=spRdashdot,LTB_E=LTBw_E, LTB_Edash=dLTBw_E_dr)

#num_angles = 100 #20. #200 #200
#angles = np.linspace(0.,0.995*np.pi,num=100,endpoint=True)
##angles = np.concatenate( (np.linspace(0.,0.995*np.pi,num=10,endpoint=True), 
##                        np.linspace(1.01*np.pi,2.*np.pi,num=10,endpoint=False)))

#num_z_points = model_geodesics.num_pt 
#geo_z_vec = model_geodesics.z_vec

#geo_affine_vec, geo_t_vec, geo_r_vec, geo_p_vec, geo_theta_vec = \
                        #[np.zeros((num_angles,num_z_points)) for i in xrange(5)] 

#loc = 20

##First for an on center observer calculate the time when redshift is 1100.
#center_affine, center_t_vec, center_r_vec, \
#center_p_vec, center_theta_vec = model_geodesics(rp=r_vector[0],tp=model_age,alpha=angles[-1])
#sp_center_t = spline_1d(geo_z_vec,center_t_vec,s=0)
#print "age at t(z=1100) for central observer is ", sp_center_t(1100.)

##serial version
##for i, angle in zip(xrange(num_angles),angles):
##	geo_affine_vec[i,:], geo_t_vec[i,:], geo_r_vec[i,:], geo_p_vec[i,:], \
##	geo_theta_vec[i,:] = LTB_geodesics_model0(rp=loc,tp=model_age,alpha=angle)

##parallel version 2
#def geo_loop(angle):
	#return model_geodesics(rp=loc,tp=model_age,alpha=angle)

#num_cores=7
#geos = Parallel(n_jobs=num_cores,verbose=5)(
#delayed(geo_loop)(angle=angle) for angle in angles)

#i = 0
#for geo_tuple in geos:
	#geo_affine_vec[i,:], geo_t_vec[i,:], geo_r_vec[i,:], geo_p_vec[i,:], \
	#geo_theta_vec[i,:] = geo_tuple
	#i = i + 1

##finally make the 2d splines
#sp_affine = spline_2d(angles,geo_z_vec,geo_affine_vec,s=0) 
#sp_t_vec = spline_2d(angles,geo_z_vec,geo_t_vec,s=0)
#sp_r_vec = spline_2d(angles,geo_z_vec,geo_r_vec,s=0)
#sp_p_vec = spline_2d(angles,geo_z_vec,geo_p_vec,s=0)
#sp_theta_vec = spline_2d(angles,geo_z_vec,geo_theta_vec,s=0)


#ras, dec = np.loadtxt("pixel_center_galactic_coord_12288.dat",unpack=True)
#Rascension, declination, gammas = get_angles(ras, dec)

#z_of_gamma = np.empty_like(gammas)
#z_star = 1100.
##age_central = sp_center_t(z_star)
#age_central = sp_t_vec.ev(0.,z_star)
#@Findroot
#def z_at_tdec(z,gamma):
	#return sp_t_vec.ev(gamma,z)/age_central - 1.

#z_at_tdec.set_options(xtol=1e-8,rtol=1e-8)
#z_at_tdec.set_bounds(z_star-20.,z_star+20.) 
##Because Job lib can not pickle z_at_tdec.root directly
#def z_at_tdec_root(gamma):
	#return z_at_tdec.root(gamma)

#z_of_angles = Parallel(n_jobs=num_cores,verbose=0)(delayed(z_at_tdec_root)(gamma) for gamma in angles)
#z_of_angles = np.asarray(z_of_angles)

#z_of_angles_sp = spline_1d(angles,z_of_angles)
#z_of_gamma = z_of_angles_sp(gammas) 
##z_of_gamma = 1100. - (age_central-sp_t_vec.ev(gammas,1100.))/sp_t_vec.ev(gammas,1100.,dy=1)
#np.savetxt("zw_of_gamma_1000.dat",z_of_gamma)


#def lum_and_Ang_dist(gamma,z):
	#"""
	#The luminosity distance for an off center observer obtained from the angular 
	#diameter distance. Assumes the splines have already been defined and are accessible.
	#The gamma is the fixed angle.
	#"""
	#t = sp_t_vec.ev(gamma,z)
	#r = sp_r_vec.ev(gamma,z)
	#theta = sp_theta_vec.ev(gamma,z)
	
	#dr_dgamma = sp_r_vec.ev(gamma,z,dx=1)
	#dtheta_dgamma = sp_theta_vec.ev(gamma,z,dx=1)
	
	#R = spR.ev(r,t)
	#Rdash = spRdash.ev(r,t)
	#E = LTBw_E(r)
	
	#DL4 = (1.+z)**8*R**4*np.sin(theta)**2/np.sin(gamma)**2 * ( 
	      #Rdash**2/R**2/(1.+2.*E)*dr_dgamma**2 + dtheta_dgamma**2)
	
	#DL = DL4**0.25
	#DA = DL/(1.+z)**2
	#return DL4**0.25, DA

#def lum_dist(z,gamma,comp_dist):
	#"""
	#The luminosity distance for an off center observer obtained from the angular 
	#diameter distance. Assumes the splines have already been defined and are accessible.
	#The gamma is the fixed angle.
	#"""
	#t = sp_t_vec.ev(gamma,z)
	
	#r = sp_r_vec.ev(gamma,z)
	#theta = sp_theta_vec.ev(gamma,z)
	
	#dr_dgamma = sp_r_vec.ev(gamma,z,dx=1)
	#dtheta_dgamma = sp_theta_vec.ev(gamma,z,dx=1)
	
	#R = spR.ev(r,t)
	#Rdash = spRdash.ev(r,t)
	#E = LTBw_E(r)
	
	#DL4 = (1.+z)**8*R**4*np.sin(theta)**2/np.sin(gamma)**2 * ( 
	      #Rdash**2/R**2/(1.+2.*E)*dr_dgamma**2 + dtheta_dgamma**2)
	
	#DL = DL4**0.25
	
	#return DL4**0.25 - comp_dist

#******************************************************************************
#******************************************************************************

#fig = plt.figure()
#plt.plot(rw, dLTBw_M_dr(rw),label="dM_dr")
##plt.xscale('log')
##plt.yscale('symlog')
#plt.legend(loc='best')
#fig = plt.figure()
##plt.plot(rw,LTBw_M(rw)/LTB_M(rw),label="M(r)")
#plt.plot(rw,LTBw_M(rw),label="M(r)")
#plt.legend(loc='best')
#plt.yscale('log')
#plt.xscale('log')
#fig = plt.figure()
#plt.plot(rw,dLTBw_M_dr(rw)*2/rw**2,label="rho(r)")
#plt.xscale('log')
#plt.legend(loc='best')
#plt.show()
#
#
#print "all done"
#fig = plt.figure()
#plt.plot(r_vector,E_vec,'g-')
#fig = plt.figure()
#plt.plot(r_vector,E_vec*r_vector**2,'r-')
#plt.show()
#
#fig = plt.figure()
##plt.plot(r_vector,dLTBw_E_dr(r_vector))
#plt.plot(rw,dLTBw_E_dr(rw))
#plt.show()
