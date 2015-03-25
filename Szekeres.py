#!/usr/bin/env python2.7
#from __future__ import division
import numpy as np
from scipy.integrate import ode, odeint
from LTB_housekeeping import ageMpc


class Szekeres_geodesics():
	"""
	Solves the Null geodesics in Szekeres model with the metric in (t,r,p,q) 
	coordinates in which the metric is diagonal. The equations are solved w.r.t to 
	redshift. The angular diameter distance is included but is incorrect as of yet.
	"""
	def __init__(self, R,R_r,R_rr,R_rt,R_t,E, E_r, 
	             P,P_r,P_rr,Q,Q_r,Q_rr,S,S_r,S_rr,num_pt=1600, *args, **kwargs):
		
		self.R    = R;    self.R_r = R_r; self.R_rr = R_rr
		self.R_rt = R_rt; self.R_t = R_t
		self.E    = E;    self.E_r = E_r
		self.P = P; self.P_r = P_r; self.P_rr = P_rr
		self.Q = Q; self.Q_r = Q_r; self.Q_rr = Q_rr
		self.S = S; self.S_r = S_r; self.S_rr = S_rr
		
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
		atleast = 400
		atleast_tot = atleast*6+1100 #atleast_tot = atleast*4+1200
		
		if not isinstance(self.num_pt, int):
			raise AssertionError("num_pt has to be an integer")		
		elif self.num_pt < atleast_tot:
			raise AssertionError("Senor I assume at least 1600 points distributed \
			between z=0 and z=3000")
		
		bonus = self.num_pt - atleast_tot 
		#insert the extra points between z=10 and 3000
		z = np.linspace(0.,1e-8,num=atleast,endpoint=False)#1e-6
		z = np.concatenate((z,np.linspace(1e-8,1e-5,num=atleast,endpoint=False)))
		z = np.concatenate((z,np.linspace(1e-5,0.01,num=atleast,endpoint=False)))
		#z = np.concatenate((z,np.linspace(1e-8,0.01,num=atleast,endpoint=False)))
		#z = np.linspace(0.,0.01,num=atleast,endpoint=False)
		z = np.concatenate((z, np.linspace(0.01,0.1,num=atleast,endpoint=False)))
		z = np.concatenate((z, np.linspace(0.1,1.,num=atleast,endpoint=False)))
		z = np.concatenate((z, np.linspace(1.,10.,num=atleast,endpoint=False)))
		z = np.concatenate((z, np.linspace(10.,3000.,
		                    num=atleast_tot-4*atleast+bonus,endpoint=True)))#30000.
		#10.,3000.
		self.z_vec = z
		return

	def get_init_conds(self,P_obs,Dir,*args,**kwargs):
		y_init = np.zeros(8)
		cos = np.cos
		sin = np.sin
		#cot = np.cot
		tan = np.tan
		t, r, theta, phi = P_obs
		a, b = Dir
		
		y_init[0]=t; y_init[1]=r; 
		#y_init[2]=self.P(r)+self.S(r)*tan(theta/2.)*cos(phi)
		#y_init[3]=self.Q(r)+self.S(r)*tan(theta/2.)*sin(phi)
		#y_init[2] = 1e5#0.#77.#3.
		#y_init[3] = 1e5#0.#7.#7.
		y_init[2]=self.P(r)+self.S(r)/tan(theta/2.)*cos(phi)
		y_init[3]=self.Q(r)+self.S(r)/tan(theta/2.)*sin(phi)
		#y_init[2]=self.P(r)+self.S(r)*(cos(theta/2.)/sin(theta/2.))*cos(phi)
		#y_init[3]=self.Q(r)+self.S(r)*(cos(theta/2.)/sin(theta/2.))*sin(phi)
		p = y_init[2]
		q = y_init[3]
		#a = a-theta
		#b = b-phi
		sin_a = sin(a)
		cos_a = cos(a)
		sin_b = sin(b)
		cos_b = cos(b)
		
		if ( np.abs(sin_a) < 1.5e-16):
			sin_a = 0.
		elif ( np.abs(cos_a) < 1.5e-16):
			cos_a = 0.
		elif (np.abs(sin_b) < 1.5e-16):
			sin_b = 0.
		elif (np.abs(cos_b) < 1.5e-16):
			cos_b = 0.
		
		H, H_p, H_q, H_r, H_t,  F, F_p, F_q, F_r, F_t  = \
		self.get_H_F_and_derivs(t,r,p,q)
		
		dtheta_ds = sin_a*sin_b/F
		dphi_ds   = cos_a/F
		y_init[4] = sin_a*cos_b/H
		y_init[5] = 0.5*(-1.-1./tan(theta/2.)**2)*dtheta_ds*cos(phi) \
		            -sin(phi)/tan(theta/2.)*dphi_ds
		y_init[6] = 0.5*(-1.-1./tan(theta/2.)**2)*dtheta_ds*sin(phi) \
		            +cos(phi)/tan(theta/2.)*dphi_ds
		#y_init[4] = sin_a*cos_b/H
		#y_init[5] = sin_a*sin_b/F
		#y_init[6] = cos_a/F
		
		
		print "norm ", y_init[4]**2*H**2+ \
		               F**2*(y_init[5]**2+y_init[6]**2)		
		print "theta_dot, phi_dot, r_dot", y_init[5], y_init[6], y_init[4]
		print "theta, a, phi, b", theta, a, phi, b
		print "y_init ", y_init, self.S(r)
		y_init[7]=0.

		#p0 = np.cos(alpha)*np.sqrt(1.+2*self.E(rp))/self.Rdash.ev(rp,tp)
		#J  = rp*np.sin(alpha)
		# First try the LTB limit. Instead of the usual theta=0. set 
		# theta = Pi/2. Thus for E=1 then S=1/2. Choose P(r)=Q(r)=0.
		
#		y_init[0]=tp; y_init[1]=rp; 
#		y_init[2]=cot(theta/2.)*cos(phi)*self.S(r)-self.P(r) 
#		y_init[3]=cot(theta/2.)*sin(phi)*self.S(r)-self.Q(r) 
#		y_init[4]= 1.
#		y_init[5]= p0
#		y_init[6]=
#		y_init[7]=
#		y_init[8]=0.
		
		return y_init

	def get_H_F_and_derivs(self,t,r,p,q, *args, **kwargs):
		"""
		"""
		r = np.abs(r)
		R    = self.R.ev(r,t)
		R_r  = self.R_r.ev(r,t)
		R_t  = self.R_t.ev(r,t)
		R_rr = self.R_rr.ev(r,t)
		R_rt = self.R_rt.ev(r,t)
		
		k   = -2.*self.E(r)
		k_r = -2.*self.E_r(r)
		
		P = self.P(r); P_r = self.P_r(r); P_rr = self.P_rr(r)
		Q = self.Q(r); Q_r = self.Q_r(r); Q_rr = self.Q_rr(r)
		S = self.S(r); S_r = self.S_r(r); S_rr = self.S_rr(r)
		
		#E = ((p**2 + q**2)/2. - P*p - Q*q + P**2/2. +Q**2/2. + S**2/2.)/S
		E = 0.5*S*((p-P)**2/S**2 + (q-Q)**2/S**2 + 1.)
		E_p  = (p - P)/S
		E_q  = (q - Q)/S
		E_pr = -P_r/S - (p-P)/S**2*S_r
		E_qr = -Q_r/S - (q-Q)/S**2*S_r
		E_r  = -E*S_r/S + (-P_r*p - Q_r*q +P_r*P + Q_r*Q + S_r*S)/S
		E_rr = (-P_rr*p - Q_rr*q + P_r**2 + P*P_rr + Q_r**2 + Q*Q_rr + \
		        S_r**2 + S*S_rr - S_rr*E - 2.*S_r*E_r)/S

		#E = 1.
		#E_p  = 0.
		#E_q  = 0.
		#E_pr = 0.
		#E_qr = 0.
		#E_r  = 0.
		#E_rr = 0.
		
		H   = (R_r - R*E_r/E)/np.sqrt(1.-k)
		H_p = (-R*E_pr/E + R*E_r*E_p/E**2)/np.sqrt(1-k)
		H_q = (-R*E_qr/E + R*E_r*E_q/E**2)/np.sqrt(1-k)
		H_r = (R_rr - R_r*E_r/E - R*E_rr/E + R*(E_r/E)**2)/np.sqrt(1.-k) + \
		      H*k_r/(1-k)/2.
		H_t = (R_rt - R_t*E_r/E)/np.sqrt(1-k)
		
		F   = R/E
		F_p = -R*E_p/E**2
		F_q = -R*E_q/E**2
		F_r = (R_r/E - R*E_r/E**2)
		F_t = R_t/E
		
		return (H, H_p, H_q, H_r, H_t,  F, F_p, F_q, F_r, F_t)
	
	def Szekeres_geodesic_derivs_odeint(self,y,z,J):
		"""
		"""
		t=y[0]; r=y[1]; p=y[2]; q=y[3]
		
		H, H_p, H_q, H_r, H_t,  F, F_p, F_q, F_r, F_t  = \
		self.get_H_F_and_derivs(t,r,p,q)
		
		dt_ds = (1.+z) 
		dr_ds = y[4]
		dp_ds = y[5]
		dq_ds = y[6]
		#dt_ds = np.sqrt(H**2* dr_ds**2 + F**2* ( dp_ds**2 + dq_ds**2 ))
		
		#ddt_dss = -H*H_t * dr_ds**2 - F*F_t* ( dp_ds**2 + dq_ds**2 )
		ddt_dss = -H*H_t * dr_ds**2 - F*F_t*dp_ds**2 - F*F_t* dq_ds**2
		ds_dz = 1./ddt_dss

		ddr_dss = -H_r/H * dr_ds**2 - 2.*H_p/H*dr_ds* (dp_ds + dq_ds) - \
		          2.*H_t/H*dr_ds*dt_ds + F*F_r/H**2 *(dp_ds**2 + dq_ds**2)  
		ddr_dsz = ddr_dss*ds_dz
		
		ddq_dss = H*H_q/F**2*dr_ds**2 - 2.*F_r/F*dr_ds*dq_ds \
		          +F_q/F*dp_ds**2 - 2.*F_p/F*dp_ds*dq_ds - F_q/F*dq_ds**2 \
		          -2.*F_t/F*dq_ds*dt_ds
		ddq_dsz = ddq_dss*ds_dz
		
		ddp_dss = H*H_p/F**2*dr_ds**2 - 2.*F_r/F*dr_ds*dp_ds - F_p/F*dp_ds**2 \
		          -2.*F_q/F*dp_ds*dq_ds - 2.*F_t/F*dp_ds*dt_ds + 2.*F_p/F*dq_ds**2
		ddp_dsz = ddp_dss*ds_dz
		
		dlnDA_ds = H_t/H+2.*F_t/F
		dlnDA_dz = dlnDA_ds*ds_dz

		dt_dz = dt_ds*ds_dz
		dr_dz = dr_ds*ds_dz
		dp_dz = dp_ds*ds_dz
		dq_dz = dq_ds*ds_dz
		
		return (dt_dz,dr_dz,dp_dz,dq_dz,ddr_dsz,ddp_dsz,ddq_dsz,dlnDA_dz) 
			
	def __call__(self,P_obs,Dir,atol=1e-15,rtol=1e-12):#atol=1e-15,rtol=1e-12):
		"""
		P_obs: tuple identifying the position of the observer 
		       (t_obs, r_obs, theta_obs, phi_obs) 
		Dir: = tuple of angular direction in which the geodesic propagates.
		       (theta_star , phi_star )
		"""
		J=0.
		y_init = self.get_init_conds(P_obs,Dir)
		odeint_ans = odeint(func=self.Szekeres_geodesic_derivs_odeint,y0=y_init,
		t=self.z_vec,
		args=(J,),Dfun=None,full_output=0,rtol=rtol,atol=atol,mxstep=10**5)
		
		return [odeint_ans[:,i] for i in range(8)]




