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
test_GP = GP_MODEL(H_in=0.6,H_out=0.7,H_not=0.7,Lambda=0.,
	               OmegaM_in=OmegaM_in,OmegaM_out=1., 
	               OmegaX_in = OmegaX_in,OmegaX_out=0.,
	               r0=3.37*Gpc,delta_r=0.35,age=13.7*ageMpc)
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

#c = 299792458. #ms^-1
#Mpc = 1.
#Gpc = 1e3*Mpc
#H_in = 0.73 #0.5-0.85 units km s^-1 Mpc^-1
#Hoverc_in = H_in*1e5/c #units of Mpc^-1
#H_out = 0.6 #0.7 #0.3-0.7 units km s^-1 Mpc^-1
#Hoverc_out = H_out*1e5/c #units of Mpc^-1
#H_not = 0.6 #0.7 #0.5-0.95 units km s^-1 Mpc^-1
#Hoverc_not = H_not*1e5/c #units of Mpc^-1
#Omega_in = 0.33 #0.05-0.35
##if Lambda is nonzero check equations for correct units. [Lambda]=[R]^-2

#Omega_Lambda = 0.7 #0.9
#Lambda = Omega_Lambda * 3. * Hoverc_not**2
#Omega_out = .98 - Omega_Lambda
#r0 = 60./H_out #3.*Gpc  #2.5*Gpc #3.5 #0.33 #0.3-4.5 units Gpc
#delta_r = 15. #2.5 #0.2*r0 #5. #0.2*r0 # 0.1r0-0.9r0
## r shall be in units of Mpc
## As a point of reference in class the conformal age of 13894.100411 Mpc 
## corresponds to age = 13.461693 Gyr
## age = 15. billion years
##     = 15. *10**9*(365.25*24.*60.*60.) s
##     = 15. *10**9*(365.25*24.*60.*60.)* 299792458. m
##     = 15. *10**9*(365.25*24.*60.*60.)* 299792458. *3.24077929*10**(-23) Mpc
##     = 15. * 306.60139383811764 Mpc
#ageMpc = 306.60139383811764
####################
##wiltshire notation
## r0 = R = 20 to 60  (h^{-1] Mpc)
#delta_w = -0.95 #-0.95 to -1
##delta_r =  w = 2.5  , 5.,  10.,  15.
## add subscript w to all quantities wiltshire
####################
#print "Omega_Lambda", Omega_Lambda, "Lambda ", Lambda
#print "Omega_out ", Omega_out, "H0 ", H_out

##from GBH import GBH_MODEL
##gbh =  GBH_MODEL(H_in=H_in,H_out=H_out,H_not=H_not,Lambda=Lambda,Omega_in=Omega_in,
##	                  r0=r0,delta_r=delta_r)  

##Omega_M      = gbh.Omega_M
##d_Omega_M_dr = gbh.d_Omega_M_dr
##H0overc      = gbh.H0overc
##d_H0overc_dr = gbh.d_H0overc_dr 
##LTB_M        = gbh.LTB_M
##dLTB_M_dr    = gbh.dLTB_M_dr 
##LTB_E        = gbh.LTB_E
##dLTB_E_dr    = gbh.dLTB_E_dr

#def dLTBw_M_dr(r):
	#"""
	#[LTB_M] = Mpc
	#"""
	#return_me = 0.75*Omega_out*Hoverc_out**2*r**2 * (2.+delta_w - delta_w* 
	                                                 #np.tanh((r-r0)/delta_r))
	#return return_me

##fist make a spline and use it to calcuate the integral than make a second spline
## so that it is computationally less expensive

##generously sample M(r) as it is not expensive
##rw = np.concatenate((np.logspace(np.log10(1e-10),np.log10(1.),num=500,endpoint=False),
##                       np.linspace(1.,r0+4.*delta_r,num=500,endpoint=False)))

##rw = np.concatenate((rw,np.linspace(r0+4.*delta_r,20.*Gpc,num=300,endpoint=True)))

#rw = sample_radial_coord(r0=r0,delta_r=delta_r,r_init=1e-10,r_max=20*1e3,num_pt1=1000,num_pt2=1000)

#r_vector = sample_radial_coord(r0=r0,delta_r=delta_r,r_init=1e-4,r_max=20*1e3,num_pt1=100,num_pt2=100)
#size_r_vector = 200

#spdLTBw_M_dr = spline_1d(rw, dLTBw_M_dr(rw), s=0) #dLTBw_M_dr(rw), s=0)
#spdLTBw_M_dr_int = spdLTBw_M_dr.antiderivative()
#Mw = spdLTBw_M_dr_int(rw) #- spdLTBw_M_dr_int(rw[0])
#model_age = 4282.74963782
#spMw = spline_1d(rw,Mw,s=0)

#def LTBw_M(r):
	#"""
	#[LTB_M] = Mpc
	#"""
	#return spMw(r)




#@Integrate
#def LTB_t(RoverR0,twoE,twoM,Lambda_over3):
	##return 1./np.sqrt(twoE + twoM/RoverR0 + Lambda_over3 * RoverR0**2)
	#return np.sqrt(RoverR0)/np.sqrt(twoE*RoverR0 + twoM + Lambda_over3 * RoverR0**3)

#LTB_t.set_options(epsabs=1.49e-16,epsrel=1.49e-12)
#LTB_t.set_limits(0.,1.)
#@Findroot
#def LTB_2E_Eq(twoE_over_r3,twoM_over_r3,Lambda_over3):
	#return model_age - LTB_t.integral(twoE_over_r3,twoM_over_r3,Lambda_over3) #*1.e-3

 

#LTB_2E_Eq.set_options(xtol=4.4408920985006262e-16,rtol=4.4408920985006262e-15)
#LTB_2E_Eq.set_bounds(0.,1e-6) #(0,2.)

#E = np.zeros(len(r_vector))
##serial loop
##i = 0
##for r in r_vector:
##	E[i] = LTB_2E_Eq.root(2.*LTB_M(r)/r**3,0.)
##	i = i + 1

#def E_loop(r,Lambda_over3):
	#return LTB_2E_Eq.root(2.*LTBw_M(r)/r**3,Lambda_over3)


#num_cores = mp.cpu_count()-1

#E_vec = Parallel(n_jobs=num_cores,verbose=0)(delayed(E_loop)(r,Lambda/3.) for r in r_vector)
#E_vec = np.asarray(E_vec)/2.

#i = 0
#for r in r_vector:
	#print E_vec[i],  r
	#i = i + 1


#E_vec = E_vec

#spLTBw_E = spline_1d(r_vector, E_vec, s=0) 
#dE_vec_dr = spLTBw_E(r_vector,nu=1) #compute the first derivative
#spdLTBw_E_dr = spline_1d(r_vector,dE_vec_dr,s=0)

#def LTBw_E(r):
	#"""
	#returns the spline for r^2 * E(r) as E was E/r^2
	#"""
	#return r**2*spLTBw_E(r)

#def dLTBw_E_dr(r):
	#"""
	#Returns the spline for diff(E(r),r)
	#"""
	#return 2*r*spLTBw_E(r) + r**2*spdLTBw_E_dr(r)



#model =  LTB_ScaleFactor(Lambda=Lambda,LTB_E=LTBw_E, LTB_Edash=dLTBw_E_dr,\
                              #LTB_M=LTBw_M, LTB_Mdash=dLTBw_M_dr)

#num_pt = 1000 #6000
#r_vec, t_vec, R_vec, Rdot_vec, Rdash_vec, Rdotdot_vec, Rdashdot_vec, = \
              #[np.zeros((size_r_vector,num_pt)) for i in xrange(7)]

##serial 
##for i, r_loc in zip(range(len(r_vector)),r_vector):
##	print r_loc
##	t_vec[i,:], R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdotdot_vec[i,:], \
##	Rdashdot_vec[i,:] = LTB_model0(r_loc=r_loc,num_pt=num_pt)
##	r_vec[i,:] = r_vec[i,:] + r_loc

#def r_loop(r_loc):
	#return model(r_loc=r_loc,t_max=model_age,num_pt=num_pt)


#num_cores = mp.cpu_count()-1	
#r = Parallel(n_jobs=num_cores,verbose=0)(delayed(r_loop)(r_loc) for r_loc in r_vector)
##r = Parallel(n_jobs=num_cores,verbose=0)(delayed(LTB_model0)(r_loc=r_loc,num_pt=num_pt) for r_loc in r_vector)

#i = 0
#for tup in r:
	#t_vec[i,:], R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdotdot_vec[i,:], \
	#Rdashdot_vec[i,:] = tup
	#i = i + 1

#t_vector = t_vec[0,:]
#sp = spline_2d(r_vector,t_vector,R_vec,s=0)
#spdr = spline_2d(r_vector,t_vector,Rdash_vec,s=0)
#spR = spline_2d(r_vector,t_vector,R_vec,s=0)
#spRdot = spline_2d(r_vector,t_vector,Rdot_vec,s=0)
#spRdash = spline_2d(r_vector,t_vector,Rdash_vec,s=0)
#spRdashdot = spline_2d(r_vector,t_vector,Rdashdot_vec,s=0)

#print "checking that age is the same "
#for r_val in r_vector:
	#print "model age = ", model_age/ageMpc, "sp(r,age) ", sp.ev(r_val,model_age), "r ", r_val
	#print "H(r,t0) ", spRdot.ev(r_val,model_age)/spR.ev(r_val,model_age)

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
