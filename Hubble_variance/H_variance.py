#!/usr/bin/env python2.7
import numpy as np


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
	#ell, bee = np.loadtxt("pixel_center_galactic_coord_49152.dat",unpack=True)
	
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


def Pearson_corr_coeff(H_map, sigma, T_map, T_mean):
	"""
	Not optimized but written to match exactly the corresponding equation in paper.
	
	H_map:
	    The Hubble constant map. It is 1d array and has size = number of pixels,
	    which is chosen according to the desired resolution in healpy corresponding 
	    to nside 16,32 or 64.
	sigma:
	    The uncertainty at each point in H_map. Dimension of sigma is same as H_map
	T_map:
	     Temperature map. It has same dimensions as H_map
	T_mean:
	     Mean temperature
	returns:
	     rho_HT
	     The Pearson correlation coefficient calculated from H_map and T_map.
	     See Eq. (12) in http://arxiv.org/abs/1201.5371v5
	"""
	
	H_mean = np.sum(H_map/sigma**2) / np.sum(1./sigma**2)
	
	Np = np.sqrt(H_map.size)
	Delta_H = H_map - H_mean
	Delta_T = T_map - T_mean
	
	numer_rho_HT = np.sqrt(Np) * np.sum ( Delta_H / sigma**2 * Delta_T )
	
	denom_rho_HT = np.sqrt( np.sum(1./sigma**2) * np.sum( Delta_H**2/sigma**2) * 
	                        np.sum(Delta_T**2) )
	
	rho_HT = numer_rho_HT / denom_rho_HT
	return rho_HT


def boost_T(T0,v,ell,bee,l,b):
	"""
	remark:
	    switching simultaneously ell with l and bee with b makes no difference
	v:
	  boost velocity in km/sec.
	returns:
	        The boosted temperature
	        T' = T0/ gamma / (1 - v * cos(phi) ) 
	        where phi is the angle between the data point and the boost direction
	"""
	#dimensionless velocity 
	c = 299792.458 # km / sec
	v = v/c
	gamma = 1./np.sqrt(1. - v**2)
	
	cos_phi = np.sin(bee)*np.sin(b) + np.cos(bee)*np.cos(b)*np.cos(ell-l)
	return T0/gamma/ (1 + v * cos_phi )





