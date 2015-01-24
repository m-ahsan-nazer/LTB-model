#!/usr/bin/env python2.7
#from __future__ import division
import numpy as np
from scipy.integrate import ode, odeint
from LTB_housekeeping import ageMpc


class Szekeres_geodesics():
	"""
	Solves the Null geodesics in Szekeres model. The equations are solved w.r.t to 
	the affine parameter ``s'' in units of Mpc but splined w.r.t to redshift.
	"""
	def __init__(self, R,R_r,R_rr,R_rt,R_t,E, E_r 
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
		self.s_vec = np.empty(num_pt)
		self._set_s_vec()
	
	def _set_s_vec(self):
		"""
		vector of affine parameter in units of Mega parsec at which points the 
		geodesics are saved
		"""
		atleast = 100
		atleast_tot = atleast*4+1200
		
		if not isinstance(self.num_pt, int):
			raise AssertionError("num_pt has to be an integer")		
		elif self.num_pt < atleast_tot:
			raise AssertionError("Senor I assume at least 1600 points distributed \
			between s=0 and s=15.*306 Mega parsecs")
		
		bonus = self.num_pt - atleast_tot 
		#insert the extra points between z=10 and 3000
		s = np.linspace(0.,0.01,num=atleast,endpoint=False)
		s = np.concatenate((s, np.linspace(0.01,0.1,num=atleast,endpoint=False)))
		s = np.concatenate((s, np.linspace(0.1,1.,num=atleast,endpoint=False)))
		s = np.concatenate((s, np.linspace(1.,10.,num=atleast,endpoint=False)))
		s = np.concatenate((s, np.linspace(10.,15.*ageMpc,
		                    num=atleast_tot-4*atleast+bonus,endpoint=True)))
		
		self.s_vec = s
		return

	def get_H_F_and_derivs(self,t,r,p,q, *args, **kwargs):
		"""
		"""
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
		
		E = ((p**2 + q**2)/2. - P*p - Q*q + P**2/2. +Q**2/2. + S**2/2.)/S
		E_p  = (p - P)/S
		E_q  = (q - Q)/S
		E_pr = -P_r/S + P/S**2*S_r
		E_qr = -Q_q/S + Q/S**2*S_r
		E_r  = -E*S_r/S + (-P_r*p - Q_r*q +P_r*P + Q_r*Q + S_r*S)/S
		E_rr = (-P_rr*p - Q_rr*q + P_r**2 + P*P_rr + Q_r**2 + Q*Q_rr + \
		        S_r**2 + S*S_rr - S_rr*E - 2.*S_r*E_r)/S
		
		H   = ((R_r - R)*E_r/E)/np.sqrt(1.-k)
		H_p = (-R*E_pr/E + R*E_r*E_p/E**2)/np.sqrt(1-k)
		H_q = (-R*E_qr/E + R*E_r*E_q/E**2)/np.sqrt(1-k)
		H_r = (R_rr - R_r*E_r/E - R*E_rr/E + R*E_r**2)/np.sqrt(1.-k) + \
		      H*k_r/(1-k)/2.
		H_t = (R_rt - R_t*E_r/E)/np.sqrt(1-k)
		
		F   = R/E
		F_p = -R*E_p/E**2
		F_q = -R*E_q/E**2
		F_r = (R_r/E - R*E_r/E**2)
		F_t = R_t/E
		
		return (H, H_p, H_q, H_r, H_t,  F, F_p, F_q, F_r, F_t)
	
	def Szekeres_geodesic_derivs_odeint(self,y,t,J):
		"""
		"""
		t=y[0]; r=y[1]; p=y[2]; q=y[3]
		
		H, H_p, H_q, H_r, H_t,  F, F_p, F_q, F_r, F_t  = \
		self.get_H_F_and_derivs(t,r,p,q)
		 
		dt_ds = y[4]
		dr_ds = y[5]
		dp_ds = y[6]
		dq_ds = y[7]
		
		ddt_dss = H*H_t * dr_ds**2 + F*F_t* ( dp_ds**2 + dq_ds**2 )
		ddt_dss = -ddt_dss

		ddr_dss = H_r/H * dr_ds**2 + 2.*H_p/H*dr_ds* (dp_ds + dq_ds) + \
		          2.*H_t/H*dr_ds*dt_ds - F*F_r/H**2 *(dp_ds**2 + dq_ds**2)  
		ddr_dss = -ddr_dss
		
		ddq_dss = -H*H_q/F**2*dr_ds**2 + 2.*F_r/F*dr_ds*dq_ds \
		          -F_q/F*dp_ds**2 + 2.*F_p/F*dp_ds*dq_ds + F_q/F*dq_ds**2 \
		          +2.*F_t/F*dq_ds*dt_ds
		ddq_dss = -ddq_dss
		
		ddp_dss = -H*H_p/F**2*dr_ds**2 + 2.*F_r/F*dr_ds*dp_ds + F_p/F*dp_ds**2 \
		          +2.*F_q/F*dp_ds*dq_ds + 2.*F_t/F*dp_ds*dt_ds - 2.*F_p/F*dq_ds**2
		ddp_dss = -ddp_dss
		
		dlnDA_ds = H_t/H+2.*F_t/F
		
		return (dt_ds,dr_ds,dp_ds,dq_ds,ddt_dss,ddr_dss,ddp_dss,ddq_dss,dlnDA_ds) 
			
	def __call__(self,r=45.,t=0.92,theta=np.pi/6.,phi=np.pi/6.,
	             theta_s=np.pi/6.,phi_s=np.pi/6.,atol=1e-12,rtol=1e-10):
		"""
		"""
		y_init = np.zeros(9)
		cos = np.cos
		sin = np.sin
		cot = np.cot
		#p0 = np.cos(alpha)*np.sqrt(1.+2*self.E(rp))/self.Rdash.ev(rp,tp)
		#J  = rp*np.sin(alpha)
		# First try the LTB limit. Instead of the usual theta=0. set 
		# theta = Pi/2. Thus for E=1 then S=1/2. Choose P(r)=Q(r)=0.
		p = cot(theta/2.)*cos(phi)*self.S(r)-self.P(r) 
		q = cot(theta/2.)*sin(phi)*self.S(r)-self.Q(r) 
		H, H_p, H_q, H_r, H_t,  F, F_p, F_q, F_r, F_t  = \
		self.get_H_F_and_derivs(t,r,p,q)
		
		S = self.S(r); S_r = self.S_r(r);
		P = self.P(r); P_r = self.P_r(r);
		Q = self.Q(r); Q_r = self.Q_r(r);
		
		dt_ds = 1.
		
		A = H**2+F**2*((S_r*cot(0.5*theta)*cos(phi)+P_r)**2)+F**2*((S_r*cot(0.5*theta)*sin(phi)+Q_r)**2)
		
		B = 2.*F**2*(0.5*(S*(-1-cot(0.5*theta)**2)*theta_s*cos(phi))-S*cot(0.5*theta)*sin(phi)*phi_s)*(S_r*cot(0.5*theta)*cos(phi)+P_r)+2*F**2*(0.5*(S*(-1-cot(0.5*theta)**2)*theta_s*sin(phi))+S*cot(0.5*theta)*cos(phi)*phi_s)*(S_r*cot(0.5*theta)*sin(phi)+Q_r)
		
		C = -dt_ds**2+F**2*((0.5*(S*(-1-cot(0.5*theta)**2)*theta_s*cos(phi))-S*cot(0.5*theta)*sin(phi)*phi_s)**2)+F**2*((0.5*(S*(-1-cot(0.5*theta)**2)*theta_s*sin(phi))+S*cot(0.5*theta)*cos(phi)*phi_s)**2)
		
		dr_ds = (-B+np.sqrt(B**2 - 4.*A*C))/(2.*A)
		
		dp_ds = (S_r*cot(0.5*theta)*cos(phi)+P_r)*dr_ds+0.5*(S*(-1.-cot(0.5*theta)**2)*theta_s*cos(phi))-S*cot((1/2)*theta)*sin(phi)*phi_s
		
		dq_ds = (S_r*cot(0.5*theta)*sin(phi)+Q_r)*dr_ds+0.5*(S*(-1-cot(0.5*theta)**2)*theta_s*sin(phi))+S*cot((1/2)*theta)*cos(phi)*phi_s
		
		y_init[0]=t; y_init[1]=r; 
		y_init[2]= p
		y_init[3]= q
		y_init[4]= dt_ds
		y_init[5]= dr_ds
		y_init[6]= dp_ds
		y_init[7]= dq_ds
		y_init[8]=0.		
		
		odeint_ans = odeint(func=self.Szekeresgeodesic_derivs_odeint,y0=y_init,t=s_vec,
		args=(J,),Dfun=None,full_output=0,rtol=rtol,atol=atol)
		
		return [odeint_ans[:,i] for i in range(9)]




