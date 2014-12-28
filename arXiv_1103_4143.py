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
from LTB_housekeeping import Findroot, Integrate
from GP_profiles import GP_MODEL, get_GP_angles

from scipy.interpolate import UnivariateSpline as spline_1d
from scipy.interpolate import RectBivariateSpline as spline_2d
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import multiprocessing as mp

import healpy as hp

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

#Now make splines for M(r), dM_dr(r), H(r), dH_dr(r), p(r), dp_dr(r). 
#Splines will be much faster 
#than the direct computation done in GP_profiles. Note that since E(r), dE_dr(r) 
#are zero only M(r), dM_dr(r), H(r), dH_dr(r) and Lambda are needed for solving
#the background and geodesic equations.
# r_vec is used to make splines
# r_vector is used for making the t-r grid on which the background LTB equations 
# are solved.
# r_vec has many more points than r_vector 
r_vec = sample_radial_coord(r0=test_GP.r0,delta_r=test_GP.delta_r,r_init=1e-10,
                            r_max=15.*1e3,num_pt1=1000,num_pt2=1000)
#r_vector = sample_radial_coord(r0=r0,delta_r=delta_r,r_init=1e-4,r_max=20*1e3,num_pt1=100,num_pt2=100)
size_r_vec = r_vec.size
r_vector = sample_radial_coord(r0=test_GP.r0,delta_r=test_GP.delta_r,r_init=1e-4,
                              r_max=15.*1e3,num_pt1=100,num_pt2=100)#r_init=1e-4
size_r_vector = r_vector.size

M_GP    = test_GP.M(r_vec)
H_GP    = test_GP.H0overc(r_vec)
p_GP    = test_GP.p(r_vec)
sp_M    = spline_1d(r_vec, M_GP, s=0)
sp_dMdr = sp_M.derivative(1)
sp_dMdr = spline_1d(r_vec,sp_dMdr(r_vec))
sp_H    = spline_1d(r_vec, H_GP, s=0)
sp_dHdr = sp_H.derivative(1) 
sp_dHdr = spline_1d(r_vec,sp_dHdr(r_vec))
sp_p    = spline_1d(r_vec,p_GP, s=0)
sp_dpdr = sp_p.derivative(1)
sp_dpdr = spline_1d(r_vec,sp_dpdr(r_vec))

def sp_E(r):
	return 0.
def sp_dEdr(r):
	return 0.

#plt.figure()
#plt.plot(r_vec,sp_M(r_vec))
#plt.figure()
#plt.plot(r_vec,sp_dMdr(r_vec))
#plt.figure()
#plt.plot(r_vec,sp_H(r_vec))
#plt.figure()
#plt.plot(r_vec,sp_dHdr(r_vec))
#plt.figure()
#plt.plot(r_vec,sp_p(r_vec))
#plt.show()

if (sp_p(r_vec).any() < 0.):
	print "o yeah a negative number "

Lambda = test_GP.OmegaX(0.)*3.*test_GP.H0overc(0.)**2
model_age = test_GP.t0
print "Lambda is ", Lambda, test_GP.OmegaX(0.), test_GP.OmegaX(300.)*3.*test_GP.H0overc(300.)**2, test_GP.OmegaX(3*Gpc)*3.*test_GP.H0overc(3*Gpc)**2

def model_background(t,r):
	sinh_t = np.sinh(np.sqrt(6.*sp_p(r)) * t)
	if ( sp_p(r) < 0.):
		print "is nan", r, sp_p(r)
	cosh_t = np.cosh(np.sqrt(6.*sp_p(r)) * t) 
	# R(t,r)
	R = 3./4.*sp_M(r)* sinh_t**2 / sp_p(r)
	R = R**(1./3.)
	# diff( R(t,r), t)
	Rdot = 6**(5./6.)/3. * ( sp_M(r)/sp_p(r) )**(1./3.) * np.sqrt( sp_p(r) ) \
	       *cosh_t / sinh_t**(1./3.)
	
	# diff( R(t,r), r)
	Rdash = -sp_dpdr(r)* sp_M(r)**(1./3.) * ( 
	        6**(-2./3.)* sinh_t**(2./3.) /sp_p(r)**(4./3.) - \
	        6**(-1./6.)* cosh_t * t / sinh_t**(1./3.) / sp_p(r)**(5./6.) \
	        ) + \
	        6**(-2./3.)*sinh_t**(2./3.)* sp_dMdr(r)/ (sp_M(r)**2 * sp_p(r))**(1./3.) 
	
	# diff( R(t,r), t,r)
	Rdashdot = 1./3./Rdot/R**2 *( ( 3.*R*Rdot**2 - 9.*sp_M(r) )*Rdash + \
	                              4.*R**4*sp_dpdr(r) + \
	                              3.*sp_dMdr(r)*R \
	                            )
	return R, Rdot, Rdash, Rdashdot

t_num_pt = 1000 #6000
t_vec = np.logspace(np.log10(1e-6),np.log10(model_age),num=t_num_pt,endpoint=True)
R_vec, Rdot_vec, Rdash_vec, Rdashdot_vec, = \
              [np.zeros((size_r_vector,t_num_pt)) for i in xrange(4)]

##serial 
##for i, r_loc in zip(range(len(r_vector)),r_vector):
##	print r_loc
##	t_vec[i,:], R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdotdot_vec[i,:], \
##	Rdashdot_vec[i,:] = LTB_model0(r_loc=r_loc,num_pt=num_pt)
##	r_vec[i,:] = r_vec[i,:] + r_loc

def r_loop(r_loc):
	return model_background(t_vec,r_loc)


num_cores = mp.cpu_count()-1	
r = Parallel(n_jobs=num_cores,verbose=0)(delayed(r_loop)(r_loc) for r_loc in r_vector)

i = 0
for tup in r:
	R_vec[i,:], Rdot_vec[i,:], Rdash_vec[i,:], Rdashdot_vec[i,:] = tup
	i = i + 1

t_vector = t_vec
spR = spline_2d(r_vector,t_vector,R_vec,s=0)
spRdot = spline_2d(r_vector,t_vector,Rdot_vec,s=0)
spRdash = spline_2d(r_vector,t_vector,Rdash_vec,s=0)
spRdashdot = spline_2d(r_vector,t_vector,Rdashdot_vec,s=0)

print "A basic check on the Hubble expansion "
for r_val in r_vector:
	print "H(r,t0) ", r_val, sp_H(r_val), spRdot.ev(r_val,model_age)/spR.ev(r_val,model_age)

################################################################################
##******************************************************************************
model_geodesics =  LTB_geodesics(R_spline=spR,Rdot_spline=spRdot,
Rdash_spline=spRdash,Rdashdot_spline=spRdashdot,LTB_E=sp_E, LTB_Edash=sp_dEdr)

num_angles = 100 #20. #200 #200
angles = np.linspace(0.,0.996*np.pi,num=100,endpoint=True)
#angles = np.concatenate( (np.linspace(0.,0.995*np.pi,num=50,endpoint=True), 
#                        np.linspace(1.01*np.pi,2.*np.pi,num=50,endpoint=False)))

num_z_points = model_geodesics.num_pt 
geo_z_vec = model_geodesics.z_vec

geo_affine_vec, geo_t_vec, geo_r_vec, geo_p_vec, geo_theta_vec = \
                        [np.zeros((num_angles,num_z_points)) for i in xrange(5)] 

loc = 30.*Mpc

##First for an on center observer calculate the time when redshift is 1100.
center_affine, center_t_vec, center_r_vec, \
center_p_vec, center_theta_vec = model_geodesics(rp=r_vector[0],tp=model_age,alpha=np.pi)#angles[-1]
sp_center_t = spline_1d(geo_z_vec,center_t_vec,s=0)
print "age at t(z=1100) for central observer is ", sp_center_t(1100.)

##serial version
##for i, angle in zip(xrange(num_angles),angles):
##	geo_affine_vec[i,:], geo_t_vec[i,:], geo_r_vec[i,:], geo_p_vec[i,:], \
##	geo_theta_vec[i,:] = LTB_geodesics_model0(rp=loc,tp=model_age,alpha=angle)

##parallel version 2
def geo_loop(angle):
	return model_geodesics(rp=loc,tp=model_age,alpha=angle)

num_cores=7
geos = Parallel(n_jobs=num_cores,verbose=5)(
delayed(geo_loop)(angle=angle) for angle in angles)

i = 0
for geo_tuple in geos:
	geo_affine_vec[i,:], geo_t_vec[i,:], geo_r_vec[i,:], geo_p_vec[i,:], \
	geo_theta_vec[i,:] = geo_tuple
	i = i + 1

##finally make the 2d splines
sp_affine = spline_2d(angles,geo_z_vec,geo_affine_vec,s=0) 
sp_t_vec = spline_2d(angles,geo_z_vec,geo_t_vec,s=0)
sp_r_vec = spline_2d(angles,geo_z_vec,geo_r_vec,s=0)
sp_p_vec = spline_2d(angles,geo_z_vec,geo_p_vec,s=0)
sp_theta_vec = spline_2d(angles,geo_z_vec,geo_theta_vec,s=0)


ras, dec = np.loadtxt("pixel_center_galactic_coord_3072.dat",unpack=True)
Rascension, declination, gammas = get_GP_angles(ell=ras, bee=dec,ell_d = 51.7, bee_d = -24.9)

#z_of_gamma = np.empty_like(gammas)
z_star = 1100.
##age_central = sp_center_t(z_star)
age_central = sp_t_vec.ev(0.,z_star)
@Findroot
def z_at_tdec(z,gamma):
	return sp_t_vec.ev(gamma,z)/age_central - 1.

z_at_tdec.set_options(xtol=1e-8,rtol=1e-8)
z_at_tdec.set_bounds(z_star-20.,z_star+20.) 
##Because Job lib can not pickle z_at_tdec.root directly
def z_at_tdec_root(gamma):
	return z_at_tdec.root(gamma)

z_of_angles = Parallel(n_jobs=num_cores,verbose=0)(delayed(z_at_tdec_root)(gamma) for gamma in angles)
z_of_angles = np.asarray(z_of_angles)

z_of_angles_sp = spline_1d(angles,z_of_angles)
z_of_gamma = z_of_angles_sp(gammas) 
#z_of_gamma = 1100. - (age_central-sp_t_vec.ev(gammas,1100.))/sp_t_vec.ev(gammas,1100.,dy=1)
#np.savetxt("zw_of_gamma_1000.dat",z_of_gamma)

#z_of_gamma = 2.725*z_of_gamma
my_map = (z_of_gamma.mean() - z_of_gamma)/(1.+z_of_gamma)
#my_map = (hp.pixelfunc.fit_monopole(z_of_gamma) - z_of_gamma)/(1.+z_of_gamma)
flip = 'astro' # 'geo'
hp.mollview(map = my_map, title = "temp_map" ,
flip = flip,format='%.4e') 
hp.mollview(map = my_map, title = "temp_map" ,
flip = flip, remove_mono=True,format='%.4e') 
hp.mollview(map = my_map, title = "temp_map_no_dipole" ,
flip = flip, remove_dip=True,format='%.4e')
plt.show()

print " alms for l 0 to 10"
alms = hp.map2alm(my_map,mmax=10,lmax=10).real
for i, alm in zip(xrange(alms.size),alms):
	print "ell %i  alm %5e" %(i, alm)

cls = hp.sphtfunc.anafast(my_map,lmax=10,pol=False)
print "checking with C_l = abs(al0)^2/(2l+1)"
ells = np.arange(cls.size)
print np.sqrt(np.abs(cls) * (2.*ells +1.))


