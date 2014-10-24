# To demonstrate the use of healpy routines
import healpy as hpy
import numpy as np
from matplotlib import pylab as plt

map_name = 'GBH'
#T_cmb, ell, bee = np.loadtxt(map_name+'.dat',unpack=True)
#ls = last scattering
loc, dec, ras, z_ls, affine_parameter, t, r, kt =  \
                                      np.loadtxt(map_name+'.dat',unpack=True)

#http://en.wikipedia.org/wiki/Galactic_coordinate_system
# 0 <= ell <= 360 degrees and
# -90 <= bee <= 90 degrees

#In healpix 0 <= phi <= 2 pi 
# and colatitude 0 <= theta <= pi.   0 at north pole and pi at south pole

# ell = phi in healpix and
# bee = (pi/2 - theta) in healpix 

nside = 16 #32

#convert angles to radians
#ell = ell*(np.pi/180.)
#bee = bee*(np.pi/180.)
#correction for rounding errors, floor so that angles are not outside the allowed limits
ell = ras*0.9999
bee = dec*0.9999
#Here T_cmb is infact \nebla T /T 
T_cmb = (z_ls.mean()-z_ls)/(1.+z_ls)
print z_ls.mean(), T_cmb[0], T_cmb[-1], T_cmb[100], T_cmb[3500], z_ls[0]
print 'z_max ', z_ls.max()
#for Temp in T_cmb:
#	print Temp
def hpy_theta(latitude):
	"""
	All angles are in radians and latitude numpy array.
	Takes the latitude in galactic coordinates i.e bee and returns theta in the
	notation of healpix. theta is the latitude in healpix. In healpix theta = 0
	corresponds to north pole and theta = pi to south pole, where as in galactic 
	coordinate system the north pole is at pi/2 and the south pole at minus pi/2.
	So we apply the formula bee = (pi/2 - theta) or equally here 
	theta = (pi/2 - bee)
	
	See prac_healpix2.py where this convention is verified.
	0 <= theta <= pi 
	pi/2 >= bee >= -pi/2
	"""
	theta = np.pi/2. - latitude 
	#for ang in  theta:
	#	print theta
	return theta

def hpy_phi(longitude):
	"""
	Angles are in radians and longitude a numpy array.
	ell in galactic coordinates equals phi in healpix. i.e ell = phi.
	"""
	phi = longitude
	return phi

ith_pixel = hpy.ang2pix(nside,hpy_theta(bee),hpy_phi(ell))
print bee.shape
num_pixels = hpy.nside2npix(nside)
#np.savetxt(map_name+'pixel_num'+'.dat',pixel_num,fmt="%10d")

# num rows = num pixels and two columns
#secon column stores the number of repetitions in the pixel
map_for_anafast = np.zeros((num_pixels,2))


for pixel, value in zip(ith_pixel,T_cmb):
	map_for_anafast[pixel,0] = map_for_anafast[pixel,0] + value 	
	map_for_anafast[pixel,1] = map_for_anafast[pixel,1] + 1.

#Now average the values for each pixel, and set masked pixels
for pixel in np.arange(num_pixels):
	if (map_for_anafast[pixel,1] > 0):
		map_for_anafast[pixel,0] = map_for_anafast[pixel,0]/map_for_anafast[pixel,1]
	else:
		#set all pixels to bad value.
		map_for_anafast[pixel,0] = hpy.UNSEEN

for pixel in np.arange(num_pixels):
	if (map_for_anafast[pixel,0] > 1.):
		map_for_anafast[pixel,0] = hpy.UNSEEN#print " pixel ", pixel #print hpy.pix2ang(nside,pixel) #map_for_anafast[i,
np.savetxt(map_name+'_for_anafast'+str(num_pixels),map_for_anafast)
print " and now map "
for stuff in map_for_anafast[:,0]:
	print stuff
hpy.mollview(map_for_anafast[:,0],title="GBH map")
plt.show()
#Default regression is True
Cls = hpy.anafast(map_for_anafast[:,0],lmax=20,regression=False) #lmax=50

np.savetxt(map_name+'_Cls'+str(num_pixels)+'.dat',zip(np.arange(Cls.size),Cls))




