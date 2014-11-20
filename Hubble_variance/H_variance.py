#!/usr/bin/env python2.7
import numpy as np
import healpy as hp

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

def boost(cz,v,ell,bee,l,b):
	"""
	remark:
	    switching simultaneously ell with l and bee with b makes no difference
	returns:
	        cz' = cz + v cos(phi) 
	        where phi is the angle between the data point and the boost direction
	"""
	return cz+ v*( np.sin(bee)*np.sin(b) + np.cos(bee)*np.cos(b)*np.cos(ell-l) )


def get_Hs_sigmas_rs(cz,r,sigma):
	"""
	Hs:
	   Hubble constant in sth shell
	sigma_s:
	   uncertainty in Hs in the sth shell found from
	   sigma_s = sqrt( sigma_0s**2 + sigma_1s**2) 
	rs:
	   weighted mean distance assigned to sth shell
	returns:
	   Hs, sigma_s, rs
	"""
	a = cz**2/sigma**2
	b = cz*r /sigma**2
	
	Hs = a.sum() / b.sum()
	sigma_1s = a.sum()**1.5 / b.sum()**2
	
	rs = np.sum(r/sigma**2) / np.sum(1/sigma**2)
	sigma_not = 0.201 #in units of h^-1 Mpc 
	sigma_0s = Hs * sigma_not / rs
	
	sigma_s = np.sqrt(sigma_0s**2 + sigma_1s**2)
	
	return Hs, sigma_s, rs

def get_healpix_coords():
	"""
	returns:
	  the angular positions of healpix pixels in galactic coordinates
	  in radians
	"""
	ell, bee = np.loadtxt("pixel_center_galactic_coord_3072.dat",unpack=True)
	#ell, bee = np.loadtxt("pixel_center_galactic_coord_12288.dat",unpack=True)
	
	return ell*np.pi/180., bee*np.pi/180.

def smear(cz,r,sigma,ell,bee,ell_hp, bee_hp,sigma_theta=25.*np.pi/180.,weight=False):
	"""
	All angles are in radians
	ell_hp, bee_hp:
	    The healpix coordinates where H_alpha is calculated
	H_alpha:
	    Hubble constant at the pixel location obtained from the smearing procedure
	bar_sigma_alpha:
	    uncertainty associated with H_alpha
	returns:
	    H_alpha, bar_sigma_alpha
	"""
	pi = np.pi
	
	theta = np.arccos( np.sin(bee)*np.sin(bee_hp) + 
	                  np.cos(bee)*np.cos(bee_hp)*np.cos(ell-ell_hp) )
	
	W_alpha = 1./np.sqrt(2.*pi)/sigma_theta * np.exp(-theta**2/ (2.*sigma_theta**2))
	
	sigma_H_inv = sigma/cz 
	if (weight):
		W_alpha = W_alpha / sigma_H_inv**2 
	
	sigma2_H_alpha_inv = np.sum(W_alpha**2 * sigma_H_inv**2) / np.sum(W_alpha)**2 
	
	H_alpha_inv = np.sum(W_alpha*r / cz) / np.sum(W_alpha)
	H_alpha = 1./H_alpha_inv
	bar_sigma_alpha = np.sqrt(sigma2_H_alpha_inv) * H_alpha**2
	
	return H_alpha, bar_sigma_alpha 


def get2_Hs_sigmas_rs(indices,binning_type,cz,r,sigma):
	"""
	Uses function get_Hs_sigmas_rs for a choice of binning
	returns:
	        Hs, sigma_s, bar_rs
	"""
	Hs, sigma_s, bar_rs = [ np.zeros(binning_type.size-1) for i in (1,2,3) ]
	
	a = indices[0]
	b = indices[1]
	
	Hs[0], sigma_s[0], bar_rs[0] = get_Hs_sigmas_rs(
	                                      cz=cz[a:b], r=r[a:b],sigma=sigma[a:b])

	for i in xrange(1,binning_type.size-1):
		a = indices[i]+1
		b = indices[i+1]+1
		Hs[i], sigma_s[i], bar_rs[i] = get_Hs_sigmas_rs(
	                      cz=cz[a:b],
	                      r=r[a:b],
	                      sigma=sigma[a:b]
	                      )
	return  Hs, sigma_s, bar_rs

# 12 edges to make 11 shells
binning_1 = np.array([ 2.25, 12.50,25.00,37.50,50.00,62.50,75.00,87.50,100.00,112.50,156.25,417.44])
binning_2 = np.array([ 6.25,18.75,31.25 ,43.75,56.25,68.75 ,81.25,93.75 ,106.25,118.75, 156.25,417.44])

indices_1 = np.array([np.where(dist_comp <= r_val)[0][-1] for r_val in binning_1])
indices_2 = np.array([np.where(dist_comp <= r_val)[0][-1] for r_val in binning_2])

Hs_cmb, sigma_s_cmb, bar_rs_cmb = get2_Hs_sigmas_rs(indices=indices_1,binning_type=binning_1,
                                       cz=cz_comp,r=dist_comp,sigma=sigma_comp)

print "cmb frame, binning one"
print bar_rs_cmb,"\n", Hs_cmb,"\n", sigma_s_cmb

from H_parameters import *
#boost to local group frame
cz_sun = boost(cz=cz_comp,v=-v_cmb,ell=ell_comp,bee=bee_comp,l=l_cmb,b=b_cmb)
cz_lg  = boost(cz=cz_sun,v=v_lg,ell=ell_comp,bee=bee_comp,l=l_lg,b=b_lg)

Hs_lg, sigma_s_lg, bar_rs_lg = get2_Hs_sigmas_rs(indices=indices_1,binning_type=binning_1,
                                       cz=cz_lg,r=dist_comp,sigma=sigma_comp)

print "lg frame, binning one"
print bar_rs_lg,"\n", Hs_lg,"\n", sigma_s_lg


print "cmb frame, binning two"
Hs, sigma_s, bar_rs = get2_Hs_sigmas_rs(indices=indices_2,binning_type=binning_2,
                                       cz=cz_comp,r=dist_comp,sigma=sigma_comp)
print bar_rs,"\n", Hs,"\n", sigma_s

print "lg frame, binning two"
Hs, sigma_s, bar_rs = get2_Hs_sigmas_rs(indices=indices_2,binning_type=binning_2,
                                       cz=cz_lg,r=dist_comp,sigma=sigma_comp)
print bar_rs,"\n", Hs,"\n", sigma_s

from matplotlib import pylab as plt
#fig = plt.figure()
#plt.plot(bar_rs_cmb[1:-1],(Hs_cmb[1:-1]-Hs_cmb[-1])/Hs_cmb[-1],label="deltaH cmb"+
#" "+str((Hs_cmb[0]-Hs_cmb[-1])/Hs_cmb[-1]) )
#plt.legend(loc="best")
#fig = plt.figure()
#plt.plot(bar_rs_lg[1:-1],(Hs_lg[1:-1]-Hs_lg[-1])/Hs_lg[-1],label="deltaH lg"+
#" "+str((Hs_lg[0]-Hs_lg[-1])/Hs_lg[-1]) )
#plt.legend(loc="best")
#plt.show()


from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory
import multiprocessing as mp

ell_hp, bee_hp = get_healpix_coords()

def smear_loop(cz, sigma,shell_index, ell_hp, bee_hp, inner=True):
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

Hs_in     = np.zeros((radii.size,ell_hp.size))
sigma_in  = np.zeros((radii.size,ell_hp.size))
Hs_out    = np.zeros((radii.size,ell_hp.size))
sigma_out = np.zeros((radii.size,ell_hp.size))

num_cores = mp.cpu_count()-7

for radius, i in zip(radii,xrange(radii.size)):
	shell_index = np.where(dist_comp < radius)[0][-1]
	cz = cz_comp
	sigma = sigma_comp
	
	Hs_sigma_in = Parallel(n_jobs=num_cores,verbose=5)(delayed(smear_loop)(
                  cz,sigma,shell_index,coords[0],coords[1],inner=True)
                  for coords in zip(ell_hp,bee_hp))
	Hs_sigma_in = np.asarray(Hs_sigma_in)
	Hs_in[i,:] = Hs_sigma_in[:,0]
	sigma_in[i,:] = Hs_sigma_in[:,1]
    
	Hs_sigma_out = Parallel(n_jobs=num_cores,verbose=5)(delayed(smear_loop)(
                  cz,sigma,shell_index,coords[0],coords[1],inner=False)
                  for coords in zip(ell_hp,bee_hp))
	Hs_sigma_out = np.asarray(Hs_sigma_out)
	Hs_out[i,:] = Hs_sigma_out[:,0]
	sigma_out[i,:] = Hs_sigma_out[:,1]
    
import sys
for radius, i in zip(radii,xrange(radii.size)):
	cls_in = hp.sphtfunc.anafast(Hs_in[i,:],lmax=3,pol=False)
	cls_out = hp.sphtfunc.anafast(Hs_out[i,:],lmax=3,pol=False)
	
	cls_in  = cls_in/cls_in[1]
	cls_out = cls_out/cls_out[1]
	sys.stdout.write("%.3f   %.3f    %.3f  %.3f    %.3f  \n" %(radius, cls_in[2], cls_in[3]
	, cls_out[2], cls_out[3]))
	


