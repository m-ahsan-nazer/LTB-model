#!/usr/bin/env python2.7
#from __future__ import division
import numpy as np
from scipy.integrate import ode, odeint
from LTB_housekeeping import ageMpc


class Szekeres_geodesics():
	"""
	Solves the Null geodesics in Szekeres model with the equations in spherical 
	polar coordinates. The equations are solved w.r.t to 
	redshift. The angular diameter distance differential equation is yet to be implemented.
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
		atleast = 100
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
		z = np.concatenate((z, np.linspace(0.01,0.1,num=atleast,endpoint=False)))
		z = np.concatenate((z, np.linspace(0.1,1.,num=atleast,endpoint=False)))
		z = np.concatenate((z, np.linspace(1.,10.,num=atleast,endpoint=False)))
		#z = np.linspace(0.,10.,num=atleast,endpoint=False)
		z = np.concatenate((z, np.linspace(10.,1200.,
		                    num=self.num_pt-6*atleast,endpoint=True)))#30000.
		#10.,3000.
		self.z_vec = z

#		atleast = 100
#		atleast_tot = atleast*5+1100 #atleast_tot = atleast*4+1200
#		
#		if not isinstance(self.num_pt, int):
#			raise AssertionError("num_pt has to be an integer")		
#		elif self.num_pt < atleast_tot:
#			raise AssertionError("Senor I assume at least 1600 points distributed \
#			between z=0 and z=3000")
#		
#		bonus = self.num_pt - atleast_tot 
#		#insert the extra points between z=10 and 3000
#		z = np.linspace(0.,1e-6,num=atleast,endpoint=False)
#		z = np.concatenate((z,np.linspace(1e-6,0.01,num=atleast,endpoint=False)))
#		#z = np.linspace(0.,0.01,num=atleast,endpoint=False)
#		z = np.concatenate((z, np.linspace(0.01,0.1,num=atleast,endpoint=False)))
#		z = np.concatenate((z, np.linspace(0.1,1.,num=atleast,endpoint=False)))
#		z = np.concatenate((z, np.linspace(1.,10.,num=atleast,endpoint=False)))
#		z = np.concatenate((z, np.linspace(10.,1200.,
#		                    num=atleast_tot-4*atleast+bonus,endpoint=True)))#3000
#		#10.,3000.
#		self.z_vec = z
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
		y_init[2]=theta
		y_init[3]=phi
		b = b-phi
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

#		if ( np.abs(sin_a) > 0.9999999):
#			sin_a = 1.
#		elif ( np.abs(cos_a)  > 0.9999999):
#			cos_a = 1.
#		elif (np.abs(sin_b)  > 0.9999999):
#			sin_b = 1.
#		elif (np.abs(cos_b)  > 0.9999999):
#			cos_b = 1.
		
		A, A_t, A_r, A_theta,A_phi,\
		B, B_t, B_r, B_theta,B_phi,\
		C, C_t, C_r, C_theta,C_phi,\
		F, F_t, F_r,\
		G, G_t, G_r, G_theta =\
		self.get_A_B_C_F_G_and_derivs(t,r,theta,phi)
		
		dtheta_ds = sin_a*sin_b/np.sqrt(F)
		dphi_ds   = cos_a/np.sqrt(G)
		#dtheta_ds = cos_a*sin_b/np.sqrt(F)
		#dphi_ds   = sin_a*sin_b/np.sqrt(G)
		#dr_ds = (-2.*(B*dphi_ds + C*dtheta_ds)+\
		#          np.sqrt((2.*B*dphi_ds+2.*C*dphi_ds)**2\
		#         -4.*A*(F*dtheta_ds**2 + G*dphi_ds**2 - 1.)))/(2.*A)
		#dr_ds = -np.abs(dr_ds)
		#dr_ds = -1./A*(B*dphi_ds + C*dtheta_ds - np.sqrt(-A*F*dtheta_ds**2\
		#        -A*G*dphi_ds**2 + B**2*dphi_ds**2 + 2.*B*C*dphi_ds*dtheta_ds\
		#        +C**2*dtheta_ds**2 + A))
		#dr_ds = -np.abs(dr_ds)
		#dr_ds = (-2.*(B*dphi_ds + C*dtheta_ds)+\
		#          np.sqrt((2.*B*dphi_ds+2.*C*dphi_ds)**2\
		#         +4.*A*sin_a**2*cos_b**2))/(2.*A)
		dr_ds = -sin_a*cos_b/np.sqrt(A)
		if (dtheta_ds == 0. and dphi_ds == 0.):
			dr_ds = -1.
		if (np.isnan(dr_ds)):
			print "isnan"
			dr_ds = -1e-10    		
		y_init[4] = dr_ds
		y_init[5] = dtheta_ds
		y_init[6] = dphi_ds
		
		
		print "F, G, dr_ds", F, G, dr_ds
		print "A, B, C, F, G", A, B, C, F, G
		print "sqrt ", (2.*B*dphi_ds+2.*C*dphi_ds)**2,-4.*A*(F*dtheta_ds**2 + G*dphi_ds**2 - 1.)
		print "dtheta_ds, dphi_ds", dtheta_ds , dphi_ds
		
		print "norm ", A*y_init[4]**2 + y_init[4]*(B*y_init[6]+C*y_init[5])\
		              +F*y_init[5]**2 + G*y_init[6]**2 
		               
		print "theta_dot, phi_dot, r_dot", y_init[5], y_init[6], y_init[4]
		print "theta, a, phi, b", theta, a, phi, b
		print "y_init ", y_init, self.S(r)
		y_init[7]=0.
		
		return y_init

	def get_A_B_C_F_G_and_derivs(self,t,r,theta,phi, *args, **kwargs):
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
		
		sin_theta = np.sin(theta); sin_htheta = np.sin(theta/2.)
		cos_theta = np.cos(theta); cos_htheta = np.cos(theta/2.)
		
		sin_phi = np.sin(phi); cos_phi = np.cos(phi)
#		if ( np.abs(theta) < 1e-13):
#			sin_theta = theta- theta**3/6.+theta**5/120.
#		elif (np.abs(phi) < 1e-13):
#			sin_phi = phi- phi**3/6.+phi**5/120.
		
		cot_theta = cos_theta/sin_theta; cot_htheta = cos_htheta/sin_htheta
		
		
		E        = 0.5*S/sin_htheta**2
		E_r      = 0.5*S_r/sin_htheta**2
		E_theta  = -E*cot_htheta
		E_rr      = 0.5*S_rr/sin_htheta**2
		E_rtheta = -0.5*S_r*cos_htheta/sin_htheta**3
		
		#print "E_r and ", E_r, E*S_r/S,-E/S*(S_r*cos_theta+sin_theta*(P_r*cos_phi+Q_r*sin_phi)),cos_theta
		#print "Esss ", E, E_r, E_theta, E_rr, E_rtheta
#		E        = 0.5*S/sin_htheta**2
#		E_r      = -E/S*(S_r*cos_theta+sin_theta*(P_r*cos_phi+Q_r*sin_phi))
#		E_theta  = -E*cot_htheta
#		E_rr      = 0.5*S_rr/sin_htheta**2
#		E_rtheta = -0.5*S_r*cos_htheta/sin_htheta**3
		
		Aeq = S_r**2*cot_htheta**2 + 2.*S_r*cot_htheta*(Q_r*sin_phi+P_r*cos_phi)\
		       +P_r**2 + Q_r**2
		Aeq_r = 2.*S_r*S_rr*cot_htheta**2\
		       +2.*S_rr*cot_htheta*(Q_r*sin_phi+P_r*cos_phi)\
		       +2.*S_r*cot_htheta*(Q_rr*sin_phi+P_rr*cos_phi)\
		       +2.*P*P_rr + 2.*Q*Q_rr
		Aeq_theta = S_r*(-1.-cot_htheta**2)*(S_r*cot_htheta + Q_r*sin_phi + P_r*cos_phi)
		Aeq_phi = 2.*S_r*cot_htheta*(Q_r*cos_phi - P_r*sin_phi)
		
		A = (R_r - R*E_r/E)**2/(1.-k) + (R/E)**2*Aeq
		A_t = 2.*(R_r - R*E_r/E)*(R_rt - R_t*E_r/E)/(1.-k) + 2.*R*R_t*Aeq/E**2
		A_r = 2.*(R_r - R*E_r/E)*(R_rr - R_r*E_r/E - R*E_rr/E + R*E_r**2/E**2)/(1-k)\
		     +(R_r-R*E_r/E)**2*k_r/(1.-k)**2 + 2.*R*R_r*Aeq/E**2 + R**2*Aeq_r/E**2\
		     -2.*R**2*Aeq*E_r/E**3
		A_theta = 2.*(R-R*E_r/E)*(-R*E_rtheta/E + R*E_r*E_theta/E**2)/(1.-k)\
		         +R**2*Aeq_theta/E**2 - 2.*R**2*Aeq*E_theta/E**3
		A_phi = (R/E)**2*Aeq_phi
		
		Beq = cot_htheta*(Q_r*cos_phi - P_r*sin_phi)
		Beq_r = cot_htheta*(Q_rr*cos_phi - P_rr*sin_phi)
		Beq_theta = -0.5*(1.+cot_htheta**2)*(Q_r*cos_phi - P_r*sin_phi)
		Beq_phi = cot_htheta*(-Q_r*sin_phi - P_r*cos_phi)
		
		B = (R/E)**2*S*Beq
		B_t = 2.*R*R_t*S/E**2*Beq
		#B_r = 2.*(R/E)**2*(S_r*Beq + S*Beq_r ) +  4.*R*R_r/E**2*S*Beq\
		#     -4.*R**2*E_r/E**3*S*Beq
		#compact form for B_r
		B_r = (R/E)**2*(S_r*Beq + S*Beq_r +2.*R_r/R*S*Beq - 2.*E_r/E*S*Beq)
		B_theta = S*(R/E)**2*(-2.*Beq*E_theta/E + Beq_theta) 
		B_phi = (R/E)**2*S*Beq_phi
		
		Ceq = Q_r*sin_phi + P_r*cos_phi + S_r*cot_htheta
		Ceq_r = Q_rr*sin_phi + P_rr*cos_phi + S_rr*cot_htheta
		Ceq_theta = -0.5*S_r*(1.+cot_htheta**2)
		Ceq_phi   = S_r*cos_phi - P_r*sin_phi
		
		C = -R**2/E*Ceq
		C_t = -2.*R*R_t/E*Ceq
		C_r = -2.*R*R_r/E*Ceq + (R/E)**2*E_r*Ceq -R**2/E*Ceq_r
		C_theta = (R/E)**2*E_theta*Ceq -R**2/E*Ceq_theta
		C_phi   = -R**2/E*Ceq_phi
		
		G = (R*sin_theta)**2
		G_t = 2.*R*R_t*sin_theta**2
		G_r = 2.*R_r*R*sin_theta**2
		G_theta = 2.*cos_theta*(R**2*sin_theta)
		
		
		A_B_C_F_G_and_derivs_tuple =(A, A_t, A_r, A_theta,A_phi,\
		                             B, B_t, B_r, B_theta,B_phi,\
		                             C, C_t, C_r, C_theta,C_phi,\
		                             R**2, 2.*R*R_t, 2.*R*R_r,\
		                             G, G_t, G_r, G_theta)
		return A_B_C_F_G_and_derivs_tuple
	
	def Szekeres_geodesic_derivs_odeint(self,y,z,J):
		"""
		Notation: The metric in spherical polar coordinates as in Eq. (22) of 
		http://arxiv.org/abs/1501.01413. The Equations here are equivalent to 
		Eq. (23--26) when the following substitutions are made: 
		R(t,r,theta,phi)    --> A(t,r,theta,phi)
		Phi(t,r,theta,phi)  --> B(t,r,theta,phi)
		Theta(t,r,theta,phi)--> C(t,r,theta,phi)
		P(t,r,theta)        --> G(t,r,theta)
		"""
		t=y[0]; r=y[1]; theta=y[2]; phi=y[3]
		
		A, A_t, A_r, A_theta,A_phi,\
		B, B_t, B_r, B_theta,B_phi,\
		C, C_t, C_r, C_theta,C_phi,\
		F, F_t, F_r,\
		G, G_t, G_r, G_theta =\
		self.get_A_B_C_F_G_and_derivs(t,r,theta,phi)
		
		dt_ds     = (1.+z) 
		dr_ds     = y[4]
		dtheta_ds = y[5]
		dphi_ds   = y[6]
		#dt_ds = np.sqrt(A*dr_ds**2 + 2.*B*dr_ds*dphi_ds + 2.*C*dr_ds*dtheta_ds\
		#                 +F*dtheta_ds**2 + G*dphi_ds**2)
		
		ddt_dss = -A_t*dr_ds**2/2. - C_t*dr_ds*dtheta_ds - B_t*dr_ds*dphi_ds \
		          -F_t*dtheta_ds**2/2. - G_t*dphi_ds**2/2.#
		ds_dz = 1./ddt_dss
		
		theta_eq =  - C_t*dt_ds*dr_ds - F_t*dt_ds*dtheta_ds \
		            -(C_r-A_theta/2.)*dr_ds**2 - (C_phi -B_theta)*dr_ds*dphi_ds\
		            -F_r*dr_ds*dtheta_ds + G_theta*dphi_ds**2/2.
		
		phi_eq = - B_t*dt_ds*dr_ds - G_t*dt_ds*dphi_ds \
		            -(B_r-A_phi/2.)*dr_ds**2 - (B_theta -C_phi)*dr_ds*dtheta_ds\
		            -G_r*dr_ds*dphi_ds - G_theta*dtheta_ds*dphi_ds
		
		r_eq = -A_t*dt_ds*dr_ds - C_t*dt_ds*dtheta_ds - B_t*dt_ds*dphi_ds\
		       -A_r*dr_ds**2/2. - A_theta*dr_ds*dtheta_ds - A_phi*dr_ds*dphi_ds\
		       -(C_theta - F_r/2.)*dtheta_ds**2 - (C_phi + B_theta)*dtheta_ds*dphi_ds\
		       -(B_phi - G_r/2.)*dphi_ds**2
		
		ddr_dss = (r_eq - C/F*theta_eq - B/G*phi_eq)/(A - C**2/F - B**2/G)
		ddr_dsz = ddr_dss*ds_dz
		
		ddtheta_dss = -C*ddr_dss/F + theta_eq/F
		ddtheta_dsz = ddtheta_dss*ds_dz
		
		ddphi_dss = -B*ddr_dss/G +phi_eq/G
		ddphi_dsz = ddphi_dss*ds_dz
		
		dlnDA_ds = 1.
		dlnDA_dz = dlnDA_ds*ds_dz

		dt_dz = dt_ds*ds_dz
		dr_dz = dr_ds*ds_dz
		dtheta_dz = dtheta_ds*ds_dz
		dphi_dz = dphi_ds*ds_dz
		
		return (dt_dz,dr_dz,dtheta_dz,dphi_dz,ddr_dsz,ddtheta_dsz,ddphi_dsz,dlnDA_dz) 
			
	def __call__(self,P_obs,Dir,atol=1e-14,rtol=1e-8):#atol=1e-15,rtol=1e-12):
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




