#!/usr/bin/env python2.7
import numpy as np
import healpy as hp

#The columns in the COMPOSITE sample are as follows:
#v = cz (km/sec)   d (/h Mpc)  v_pec (km/s)   sigma_v   l (degrees) b (degrees)
#The redshifts are given in the reference frame of the sun

cz_cmb, d_cmb, vpec_cmb, sigma_cmb, ell_cmb, bee_cmb = \
np.loadtxt("COMPOSITEn-survey-dsrt.dat",unpack=True)

#Convert from distance to H uncertainty
sigma_cmb = sigma_cmb / 100.
#Angles are always in radians in this code
ell_cmb = ell_cmb*np.pi/180.
bee_cmb = bee_cmb*np.pi/180.

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
	ell, bee = np.loadtxt("pixel_center_galactic_coord_12288.dat",unpack=True)
	
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

# 12 edges to make 11 shells
binning_1 = np.array([ 2.25, 12.50,25.00,37.50,50.00,62.50,75.00,87.50,100.00,112.50,156.25,417.44])
binning_2 = np.array([ 6.25,18.75,31.25 ,43.75,56.25,68.75 ,81.25,93.75 ,106.25,118.75, 156.25,417.44])

indices_1 = np.array([np.where(d_cmb <= r_val)[0][-1] for r_val in binning_1])
indices_2 = np.array([np.where(d_cmb <= r_val)[0][-1] for r_val in binning_2])

a = indices_1[0]
b = indices_1[1]
Hs, sigma_s, bar_rs = get_Hs_sigmas_rs( cz_cmb[a:b], d_cmb[a:b],sigma_cmb[a:b])
print b-a, binning_1[0], bar_rs, Hs, sigma_s
for i in xrange(1,binning_1.size-1):
	a = indices_1[i]+1
	b = indices_1[i+1]+1
	Hs, sigma_s, bar_rs = get_Hs_sigmas_rs(
	                      cz_cmb[a:b],
	                      d_cmb[a:b],
	                      sigma_cmb[a:b]
	                      )

	print b-a, binning_1[i], bar_rs, Hs, sigma_s

from H_parameters import *
cz_sun = boost(cz=cz_cmb,v=-v_cmb,ell=ell_cmb,bee=bee_cmb,l=l_cmb,b=b_cmb)
cz_lg  = boost(cz=cz_sun,v=v_lg,ell=ell_cmb,bee=bee_cmb,l=l_lg,b=b_lg)

print "*******************************************************"
a = indices_1[0]
b = indices_1[1]
Hs, sigma_s, bar_rs = get_Hs_sigmas_rs( cz_lg[a:b], d_cmb[a:b],sigma_cmb[a:b])
print b-a, binning_1[0], bar_rs, Hs, sigma_s
for i in xrange(1,binning_1.size-1):
	a = indices_1[i]+1
	b = indices_1[i+1]+1
	Hs, sigma_s, bar_rs = get_Hs_sigmas_rs(
	                      cz_lg[a:b],
	                      d_cmb[a:b],
	                      sigma_cmb[a:b]
	                      )

	print b-a, binning_1[i], bar_rs, Hs, sigma_s

print "******************************************************"

a = indices_2[0]
b = indices_2[1]
Hs, sigma_s, bar_rs = get_Hs_sigmas_rs( cz_cmb[a:b], d_cmb[a:b],sigma_cmb[a:b])
print b-a, binning_2[0], bar_rs, Hs, sigma_s
for i in xrange(1,binning_1.size-1):
	a = indices_2[i]+1
	b = indices_2[i+1]+1
	Hs, sigma_s, bar_rs = get_Hs_sigmas_rs(
	                      cz_cmb[a:b],
	                      d_cmb[a:b],
	                      sigma_cmb[a:b]
	                      )

	print b-a, binning_2[i], bar_rs, Hs, sigma_s

from H_parameters import *
cz_sun = boost(cz=cz_cmb,v=-v_cmb,ell=ell_cmb,bee=bee_cmb,l=l_cmb,b=b_cmb)
cz_lg  = boost(cz=cz_sun,v=v_lg,ell=ell_cmb,bee=bee_cmb,l=l_lg,b=b_lg)

print "*******************************************************"
a = indices_2[0]
b = indices_2[1]
Hs, sigma_s, bar_rs = get_Hs_sigmas_rs( cz_lg[a:b], d_cmb[a:b],sigma_cmb[a:b])
print b-a, binning_2[0], bar_rs, Hs, sigma_s
for i in xrange(1,binning_2.size-1):
	a = indices_2[i]+1
	b = indices_2[i+1]+1
	Hs, sigma_s, bar_rs = get_Hs_sigmas_rs(
	                      cz_lg[a:b],
	                      d_cmb[a:b],
	                      sigma_cmb[a:b]
	                      )

	print b-a, binning_2[i], bar_rs, Hs, sigma_s



