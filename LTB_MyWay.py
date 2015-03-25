#!/usr/bin/env python2.7
#from __future__ import division
import numpy as np
from scipy.integrate import ode, odeint
from LTB_housekeeping import ageMpc


class LTB_geodesics():
	"""
	Solves the Null geodesics in Szekeres model. The independent variable is 
	redshift and not the affine parameter. Class LTB_geodesics in LTB_Scalss_v2.py 
	are the LTB geodesics commonly used. There phi is set to zero and its differential 
	equation removed, owing to the symmetry inherent in the LTB metric. 
	Here I have not made any simplifications and both theta and phi coordiantes 
	are evolved. 
	"""
	def __init__(self, E, E_r, R,R_r,R_rt,R_t, 
	             num_pt=1600, *args, **kwargs):
		
		self.R    = R;    self.R_r = R_r;
		self.R_rt = R_rt; self.R_t = R_t
		self.E    = E;    self.E_r = E_r
		
		self.args      = args
		self.kwargs    = kwargs
		#setup the vector of redshifts
		self.num_pt = num_pt
		self.z_vec = np.empty(num_pt)
		self._set_z_vec()
	
	def _set_z_vec(self):
		"""
		vector of redshifts at which points the geodesics are saved
		"""
		atleast = 100
		atleast_tot = atleast*5+1100 #atleast_tot = atleast*4+1200
		
		if not isinstance(self.num_pt, int):
			raise AssertionError("num_pt has to be an integer")		
		elif self.num_pt < atleast_tot:
			raise AssertionError("Senor I assume at least 1600 points distributed \
			between z=0 and z=3000")
		
		bonus = self.num_pt - atleast_tot 
		#insert the extra points between z=10 and 3000
		z = np.linspace(0.,1e-6,num=atleast,endpoint=False)
		z = np.concatenate((z,np.linspace(1e-6,0.01,num=atleast,endpoint=False)))
		#z = np.linspace(0.,0.01,num=atleast,endpoint=False)
		z = np.concatenate((z, np.linspace(0.01,0.1,num=atleast,endpoint=False)))
		z = np.concatenate((z, np.linspace(0.1,1.,num=atleast,endpoint=False)))
		z = np.concatenate((z, np.linspace(1.,10.,num=atleast,endpoint=False)))
		z = np.concatenate((z, np.linspace(10.,3000.,
		                    num=atleast_tot-4*atleast+bonus,endpoint=True)))
		#10.,3000.
		self.z_vec = z
		return
	
	def reset_z_vec(self):
		#atleast = 100
		#z = np.linspace(0.,1e-6,num=atleast,endpoint=False)
		#z = np.concatenate((z,np.linspace(1e-6,0.01,num=atleast,endpoint=False)))
		#z = np.concatenate((z, np.linspace(0.01,0.2,num=atleast,endpoint=True)))
		#self.z_vec = z
		self.z_vec = np.log(np.logspace(0.,0.2,25,base=np.exp(1.)))	
		return

	def get_E_R_and_derivs(self,t,r, *args, **kwargs):
		"""
		"""
		r = np.abs(r)
		R    = self.R.ev(r,t)
		R_r  = self.R_r.ev(r,t)
		R_t  = self.R_t.ev(r,t)
		R_rt = self.R_rt.ev(r,t)
		R_rr = self.R_r.ev(r,t,dx=1,dy=0)
		
		E   = self.E(r)#-2.*self.E(r)
		E_r = self.E_r(r)#-2.*self.E_r(r)
		
		return (E, E_r, R, R_r, R_rr, R_rt, R_t)
	
	def get_init_conds(self,P_obs,Dir,*args,**kwargs):
		y_init = np.zeros(8)
		cos = np.cos
		sin = np.sin

		t, r, theta, phi = P_obs
		a, b = Dir
		
		y_init[0]=t; y_init[1]=r; 
		y_init[2]=theta
		y_init[3]=phi 
		#y_init[4]= -1.
		#y_init[4]= sin(b-phi)*sin(a)+cos(a)#*(np.sqrt(1.+2.*self.E(r))/self.R_r.ev(r,t)) ## # (
		#y_init[5]= -(sin(theta)*cos(a)-sin(a)*cos(theta+b-phi))/(r*cos(theta)**2)
		#y_init[6]= sin(a)*sin(b-phi)/ (r*sin(theta))
		#c_theta = -0.95*r**2#require - R^2 <= c_theta <= 0 
		#c_phi = np.sqrt(-0.95*c_theta*np.sin(theta)**2) #reqauire c_phi**2 < -c_theta*sin(theta)**2  
		#y_init[4] = np.sqrt(1.+2.*self.E(r))*np.sqrt(1.+c_theta/r**2) #plus or minus sign choice
		#y_init[5] = np.sqrt(-c_theta/r**4-c_phi**2/(r**4*np.sin(theta)**2)) #plus or minus sign choice
		#y_init[6] = c_phi/(r**2*np.sin(theta)**2) #plus or minus sign choice
		
		
		#y_init[5] = (theta- a)/1e-1/r
		#y_init[6] = (phi - b)/1e-1/(r*np.sin(theta))
		#y_init[4] = np.sqrt(1.+2.*self.E(r))*np.sqrt(1.-r**2*y_init[5]**2-\
		#                               np.sin(theta)**2*r**2*y_init[6]**2)
		
		#delta_r = 1e-3
		#delta_theta = a-theta
		#delta_phi = b-phi
		#L01 = np.sqrt(-1+delta_r**2/(1.+2*self.E(r))+delta_theta**2+delta_phi**2)
		#y_init[4] = -delta_r/L01
		#y_init[5] = delta_theta/L01
		#y_init[6] = delta_phi/L01
		#a = a-theta
		#a = a+theta
		b = b-phi
		sin_a = sin(a)
		cos_a = cos(a)
		sin_b = sin(b)
		cos_b = cos(b)
		#if ( np.abs(sin_a) < 1.5e-16):
		#	sin_a = 0.
		#elif ( np.abs(cos_a) < 1.5e-16):
		#	cos_a = 0.
		#elif (np.abs(sin_b) < 1.5e-16):
		#	sin_b = 0.
		#elif (np.abs(cos_b) < 1.5e-16):
		#	cos_b = 0.
		
		y_init[4] = -sin_a*cos_b*np.sqrt(1.+2.*self.E(r))
		y_init[5] = sin_a*sin_b/r
		y_init[6] = cos_a/(sin(theta)*r)
		
		#y_init[4] = np.sqrt(1.+2.*self.E(r))
		#y_init[5] = 1./r*0.
		#y_init[6] = 1./(r*sin(theta))*0.
		
		
		print "norm ", y_init[4]**2/np.sqrt(1.+2.*self.E(r))+ \
		               r**2*(y_init[5]**2+sin(theta)**2*y_init[6]**2)		
		print "theta_dot, phi_dot, r_dot", y_init[5], y_init[6], y_init[4]
		print "theta, a, phi, b", theta, a, phi, b
		y_init[7]=0.	
		return y_init
	
	def LTB_geodesic_derivs_ode(self,z,y,*arg):#(self,y,t,*arg):
		"""
		"""
		t=y[0]; r=y[1]; theta=y[2]; phi=y[3]
		sin_theta = np.sin(theta); cos_theta = np.cos(theta)
		
		E, E_r, R, R_r, R_rr, R_rt, R_t  = \
		self.get_E_R_and_derivs(t,r)
		
		dt_ds = (1.+z)
		dr_ds = y[4]
		dtheta_ds = y[5]
		dphi_ds = y[6]
		DA = y[7]
		ddt_dss = -R_r*R_rt/(1.+2.*E)*dr_ds**2 -\
		          R*R_t*( dtheta_ds**2 + sin_theta**2*dphi_ds**2 )
		
		ds_dz = 1./ddt_dss
		
		ddr_dss = -(2.*R_rr*E-R_r*E_r+R_rr)/(1.+2.*E)/R_r*dr_ds**2 - \
		            2.*R_rt/R_r*dr_ds*dt_ds + \
		           (1.+2.*E)*R/R_r*(dtheta_ds**2 + sin_theta**2*dphi_ds**2) 
		ddr_dsz = ddr_dss*ds_dz
		
		ddtheta_dss = -2./R*dtheta_ds*(R_r*dr_ds + R_t*dt_ds) + \
		              sin_theta*cos_theta*dphi_ds**2 
		ddtheta_dsz = ddtheta_dss*ds_dz
		
		ddphi_dss = -2./R*dphi_ds*(R_r*dr_ds + R_t*dt_ds) - \
		            2.*cos_theta/sin_theta*dtheta_ds*dphi_ds 
		ddphi_dsz = ddphi_dss*ds_dz
		
		dDA_ds = (R_t*dt_ds + R_r*np.abs(dr_ds))*(DA/R)
		dDA_dz = -dDA_ds*ds_dz
		
		dt_dz = dt_ds*ds_dz
		dr_dz = dr_ds*ds_dz
		dtheta_dz = dtheta_ds*ds_dz
		dphi_dz   = dphi_ds*ds_dz
		
		return [dt_dz,   dr_dz,   dtheta_dz,   dphi_dz,
		        ddr_dsz, ddtheta_dsz, ddphi_dsz, dDA_dz]

	def LTB_geodesic_derivs_odeint(self,y,z,*arg):
		"""
		"""
		t=y[0]; r=y[1]; theta=y[2]; phi=y[3]
		sin_theta = np.sin(theta); cos_theta = np.cos(theta)
		
		E, E_r, R, R_r, R_rr, R_rt, R_t  = \
		self.get_E_R_and_derivs(t,r)
		
		dt_ds = (1.+z)
		dr_ds = y[4]
		dtheta_ds = y[5]
		dphi_ds = y[6]
		DA = y[7]
		ddt_dss = -R_r*R_rt/(1.+2.*E)*dr_ds**2 -\
		          R*R_t*( dtheta_ds**2 + sin_theta**2*dphi_ds**2 )
		
		ds_dz = 1./ddt_dss
		
		ddr_dss = -(2.*R_rr*E-R_r*E_r+R_rr)/(1.+2.*E)/R_r*dr_ds**2 - \
		            2.*R_rt/R_r*dr_ds*dt_ds + \
		           (1.+2.*E)*R/R_r*(dtheta_ds**2 + sin_theta**2*dphi_ds**2) 
		ddr_dsz = ddr_dss*ds_dz
		
		ddtheta_dss = -2./R*dtheta_ds*(R_r*dr_ds + R_t*dt_ds) + \
		              sin_theta*cos_theta*dphi_ds**2 
		ddtheta_dsz = ddtheta_dss*ds_dz
		
		ddphi_dss = -2./R*dphi_ds*(R_r*dr_ds + R_t*dt_ds) - \
		            2.*cos_theta/sin_theta*dtheta_ds*dphi_ds 
		ddphi_dsz = ddphi_dss*ds_dz
		
		dDA_ds = (R_t*dt_ds + R_r*np.abs(dr_ds))#*(100./R)
		dDA_dz = -dDA_ds*ds_dz
		
		dt_dz = dt_ds*ds_dz
		dr_dz = dr_ds*ds_dz
		dtheta_dz = dtheta_ds*ds_dz
		dphi_dz   = dphi_ds*ds_dz
		
		return [dt_dz,   dr_dz,   dtheta_dz,   dphi_dz,
		        ddr_dsz, ddtheta_dsz, ddphi_dsz, dDA_dz]
			
	def __call__(self,P_obs,Dir,atol=1e-12,rtol=1e-10):#atol=1e-12,rtol=1e-10):
		"""
		P_obs: tuple identifying the position of the observer 
		       (t_obs, r_obs, theta_obs, phi_obs) 
		Dir: = tuple of angular direction in which the geodesic propagates.
		       (theta_star , phi_star )
		"""
		
		y_init = self.get_init_conds(P_obs,Dir)
		#print y_init
		
		#r = ode(self.LTB_geodesic_derivs_ode).set_integrator(
		#                              'vode', method='bdf', with_jacobian=False)
		#r.set_initial_value(y_init,0.)
		i = 0
		#print self.z_vec
		#while r.successful() and r.t<self.z_vec[-2]:
		#	r.integrate(r.t+self.z_vec[i+1]-self.z_vec[i])
		#	i = i + 1
		#	print r.t, r.y
		#	print r.y[4]**2*self.R_r.ev(np.abs(r.y[1]),r.y[0])**2/(1.+2.*self.E(np.abs(r.y[1])))+\
		#	      self.R.ev(np.abs(r.y[1]),r.y[0])**2*(r.y[5]**2+np.sin(r.y[2])**2*r.y[6]**2)-(1.+r.t)**2
		#	#print ("%g %9g" %(r.t,r.y))
		odeint_ans = odeint(func=self.LTB_geodesic_derivs_odeint,y0=y_init,
		t=self.z_vec,
		args=(),Dfun=None,full_output=0,rtol=rtol,atol=atol,mxstep=10**5)
		
		return [odeint_ans[:,i] for i in range(8)]




