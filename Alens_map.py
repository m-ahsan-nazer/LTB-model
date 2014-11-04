from matplotlib import pylab as plt
import numpy as np
import healpy as hp

#Alnes_model_I_200Mpc = np.loadtxt("z_of_gamma_2.dat")
Alnes_model_I_200Mpc = np.loadtxt("z_of_gamma_1000.dat")
Alnes_model_I_200Mpc = (1100. - Alnes_model_I_200Mpc)/(1.+Alnes_model_I_200Mpc)
#Alnes_model_I_200Mpc = (Alnes_model_I_200Mpc.mean() - Alnes_model_I_200Mpc)/(1.+Alnes_model_I_200Mpc)

#monopole_removed = hp.remove_monopole(Alnes_model_I_200Mpc)
#dipole_removed = hp.remove_dipole(Alnes_model_I_200Mpc)
#quadrupole_removed = hp.remove_dipole(dipole_removed)
#octupole_removed = hp.remove_dipole(quadrupole_removed)

#hp.mollview(monopole_removed,"model I 200Mpc")
#hp.mollview(dipole_removed,"model I dipole removed")
#hp.mollview(quadrupole_removed,"model I quadrupole removed")
#hp.mollview(octupole_removed,"model I octupole removed")

print " alms for l 0 to 10"
alms = hp.map2alm(Alnes_model_I_200Mpc,mmax=0,lmax=16*2,regression=False).real
for i, alm in zip(xrange(alms.size),alms):
	print "ell %i  alm %5e" %(i, alm)
hp.mollview(map = Alnes_model_I_200Mpc, title = "Alnes_model_I_200Mpc" ,
flip = 'geo', remove_mono=True,format='%.4e') 
hp.mollview(map = Alnes_model_I_200Mpc, title = "Alnes_model_I_200Mpc" ,
flip = 'geo', remove_dip=True,format='%.4e')
plt.show()

