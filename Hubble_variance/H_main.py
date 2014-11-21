#!/usr/bin/env python2.7
import numpy as np
import healpy as hp

from H_variance import *
from H_parameters import *

#The columns in the COMPOSITE sample are as follows:
#v = cz (km/sec)   d (/h Mpc)  v_pec (km/s)   sigma_v   l (degrees) b (degrees)
#The redshifts are given in the reference frame of the sun

cz_comp, dist_comp, vpec_comp, sigma_comp, ell_comp, bee_comp = \
np.loadtxt("COMPOSITEn-survey-dsrt.dat",unpack=True)

#Convert from distance to H uncertainty
sigma_comp = sigma_comp / 100.
#Angles are always in radians in this code
ell_comp = ell_comp*np.pi/180.
bee_comp = bee_comp*np.pi/180.

# 12 edges to make 11 shells
binning_1 = np.array([ 2.25, 12.50,25.00,37.50,50.00,62.50,75.00,87.50,100.00,112.50,156.25,417.44])
binning_2 = np.array([ 6.25,18.75,31.25 ,43.75,56.25,68.75 ,81.25,93.75 ,106.25,118.75, 156.25,417.44])

indices_1 = np.array([np.where(dist_comp <= r_val)[0][-1] for r_val in binning_1])
indices_2 = np.array([np.where(dist_comp <= r_val)[0][-1] for r_val in binning_2])

Hs_cmb, sigma_s_cmb, bar_rs_cmb = get2_Hs_sigmas_rs(indices=indices_1,binning_type=binning_1,
                                       cz=cz_comp,r=dist_comp,sigma=sigma_comp)



#boost to local group frame
cz_sun = boost(cz=cz_comp,v=-v_cmb,ell=ell_comp,bee=bee_comp,l=l_cmb,b=b_cmb)
cz_lg  = boost(cz=cz_sun,v=v_lg,ell=ell_comp,bee=bee_comp,l=l_lg,b=b_lg)




from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import multiprocessing as mp

ell_hp, bee_hp = get_healpix_coords()

def wrap_smear(cz, sigma,shell_index, ell_hp, bee_hp, inner=True):
	"""
	In the smearing function only cz, sigma changed as reference frames are 
	changed the other quantities are simply those from the composite sample
	"""
	Hs = None
	sigma_s = None
	if ( inner ):
		Hs, sigma_s =  smear(cz[:shell_index],dist_comp[:shell_index],
		       sigma[:shell_index],ell_comp[:shell_index],bee_comp[:shell_index],
	           ell_hp, bee_hp,
	           sigma_theta=25.*np.pi/180.,weight=False)
	else:
		Hs, sigma_s =  smear(cz[shell_index:],dist_comp[shell_index:],
	            sigma[shell_index:],ell_comp[shell_index:],bee_comp[shell_index:],
	            ell_hp, bee_hp,
	            sigma_theta=25.*np.pi/180.,weight=False)
	
	return Hs, sigma_s

radii = np.array([12.5, 15., 20., 30., 40., 50., 60., 70., 80., 90., 100.])


num_cores = mp.cpu_count()-1

def smear_loop(i,radius):
	shell_index = np.where(dist_comp < radius)[0][-1]
	cz = cz_lg# #cz_comp
	sigma = sigma_comp
	
	Hs_in, sigma_in, Hs_out, sigma_out = \
	                                  [np.zeros(ell_hp.size) for i in (1,2,3,4)]
	for j in xrange(ell_hp.size):
		Hs_in[j], sigma_in[j] = wrap_smear(cz, sigma,shell_index, ell_hp[j],
		                                       bee_hp[j], inner=True)
		Hs_out[j], sigma_out[j] = wrap_smear(cz, sigma,shell_index, ell_hp[j],
		                                       bee_hp[j], inner=False) 
	return np.asarray([Hs_in, sigma_in, Hs_out, sigma_out])


Hs_sigma = Parallel(n_jobs=num_cores,verbose=5)(delayed(smear_loop)(
                  i,radius) for radius, i in zip(radii,xrange(radii.size)))

Hs_sigma  = np.asarray(Hs_sigma)

Hs_in     = Hs_sigma[:,0]
sigma_in  = Hs_sigma[:,1]
Hs_out    = Hs_sigma[:,2]
sigma_out = Hs_sigma[:,3]

import sys
for radius, i in zip(radii,xrange(radii.size)):
	cls_in = hp.sphtfunc.anafast(Hs_in[i,:],lmax=3,pol=False)
	cls_out = hp.sphtfunc.anafast(Hs_out[i,:],lmax=3,pol=False)
	
	cls_in  = cls_in/cls_in[1]
	cls_out = cls_out/cls_out[1]
	sys.stdout.write("%.3f   %.3f    %.3f  %.3f    %.3f  \n" %(radius, cls_in[2], cls_in[3]
	, cls_out[2], cls_out[3]))



lon_hp = np.unique(ell_hp)
lat_hp = np.pi/2 - np.unique(bee_hp)
nside = hp.npix2nside(bee_hp.size)
H_map_in  = np.zeros((radii.size,lon_hp.size,lat_hp.size))
H_map_out = np.zeros((radii.size,lon_hp.size,lat_hp.size))
#H_map = np.zeros((lon_hp.size,lat_hp.size))
for i, angle in zip(xrange(lat_hp.size),lat_hp):
	pixel_indices = hp.ang2pix(nside,angle,lon_hp)
	for j in xrange(radii.size):
		H_map_in[j,:,i] = Hs_in[j,pixel_indices]
		H_map_out[j,:,i] = Hs_out[j,pixel_indices]



