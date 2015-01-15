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
	def __init__(self, R_spline,Rdot_spline,Rdash_spline,Rdashdot_spline,LTB_E, LTB_Edash,num_pt=1600, *args, **kwargs):
		self.R         = R_spline
		self.Rdot      = Rdot_spline
		self.Rdash     = Rdash_spline
		self.Rdashdot  = Rdashdot_spline
		self.args      = args
		self.kwargs    = kwargs
		self.E         = LTB_E
		self.Edash     = LTB_Edash
		# i stands for integer index
		self.i_lambda = 0; self.i_t = 1; self.i_r = 2; self.i_p = 3; self.i_theta = 4
		#self.i_z = 0; self.i_t = 1; self.i_r = 2; self.i_p = 3; self.i_theta = 4
		#setup the vector of redshifts
		self.num_pt = num_pt
		self.z_vec = np.empty(num_pt)
		self._set_z_vec()
	
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
		
		P = self.P(r); P_r = self.P(r,nu=1); P_rr = self.P(r,nu=2)
		Q = self.Q(r); Q_r = self.Q(r,nu=1); Q_rr = self.Q(r,nu=2)
		S = self.S(r); S_r = self.S(r,nu=1); S_rr = self.S(r,nu=2)
		
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
		
		return [dt_ds,dr_ds,dp_ds,dq_ds,ddt_dss,ddr_dss,ddp_dss,ddq_dss,dlnDA_ds]  
			
	def __call__(self,rp=45.,tp=0.92,alpha=np.pi/6.,atol=1e-12,rtol=1e-10):
		y_init = np.zeros(5)
		p0 = np.cos(alpha)*np.sqrt(1.+2*self.E(rp))/self.Rdash.ev(rp,tp); J = rp*np.sin(alpha)
		y_init[0]=0.; y_init[1]=tp; y_init[2]=rp; y_init[3]=p0; y_init[4] = 0.
		
		print "R(rp,tp) = ", self.R.ev(rp,tp), rp, tp, alpha, self.Rdash.ev(rp,tp)
		#print "alpha, sign ", alpha, sign
		z_init = 0.
		#evolve_LTB_geodesic = ode(self.LTB_geodesic_derivs).set_integrator('vode', method='adams', with_jacobian=False,atol=atol, rtol=rtol)
		#evolve_LTB_geodesic.set_initial_value(y_init, z_init).set_f_params(J)
		print 'init_conds ', y_init, z_init, self.Rdash.ev(rp,tp), self.R.ev(rp,tp), 'loc ', rp
		#print "and derivs ", self.LTB_geodesic_derivs(t=0.,y=y_init,J=J)
		print "E, Edash, J ", self.E(rp), self.Edash(rp), J
	
		z_vec = self.z_vec # could just use self.z_vec itself
		print "ding "#, z_vec
		
		odeint_ans = odeint(func=self.LTB_geodesic_derivs_odeint,y0=y_init,t=z_vec,
		args=(J,),Dfun=None,full_output=0,rtol=rtol,atol=atol)
		#use odeint_ans, myfull_out  when setting full_output to True
		print  odeint_ans[-1,self.i_lambda],odeint_ans[-1,self.i_t], \
		odeint_ans[-1,self.i_r], odeint_ans[-1,self.i_p], odeint_ans[-1,self.i_theta]
		#self.i_lambda = 0; self.i_t = 1; self.i_r = 2; self.i_p = 3; self.i_theta = 4
		return odeint_ans[:,self.i_lambda],odeint_ans[:,self.i_t], \
		odeint_ans[:,self.i_r], odeint_ans[:,self.i_p], odeint_ans[:,self.i_theta]




