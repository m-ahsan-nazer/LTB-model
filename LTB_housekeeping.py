#To be used for various chores that are better left outside of the natural flow
#in the code
import numpy as np

class Findroot(object):
	"""
	Quite possibly an unnecessary class decorator for the scipy brentq root 
	finder. The idea is that one can simply access the root method 
	within the method defined for the Equation 
	to be solved.
	"""
	def __init__(self,f):
		self.f = f
		from scipy.optimize import brentq
		self._brentq = brentq
		self.set_options()
	
	def set_bounds(self,a=None,b=None):
		self._a=a
		self._b=b
	
	def root(self,*args):
		if self._a is None or self._b is None:
			raise AssertionError("Set bounds before trying to find root.")

		if  not args:
			self._root, r = self._brentq(f=self.f,a=self._a,b=self._b,
		                       args=self._args , xtol=self._xtol,
		                       rtol=self._rtol,maxiter=self._maxiter,
		                       disp=True,full_output=True)
		else:
			self._args = args
			self._root, r = self._brentq(f=self.f,a=self._a,b=self._b,
		                       args=self._args , xtol=self._xtol,
		                       rtol=self._rtol,maxiter=self._maxiter,
		                       disp=True,full_output=True)		
		self._converged = r.converged 
		return self._root
	
	@property
	def converged(self):
		return self._converged
	
	def set_options(self,xtol=1e-12,rtol=4.4408920985006262e-16,
	                maxiter=100, full_output=False, disp=True):#xtol=1e-12
		self._xtol = xtol
		self._rtol = rtol
		self._maxiter = maxiter
		self._full_output = full_output
		self._disp = disp
		return
	
	def __call__(self,x,*args):
		self._args = args

		return self.f(x,*args)




class Integrate(object):
	"""
	Given M(r) and \Lambda integrate the LTB equation w.r.t R as 
	the independent variable. 
	If decorator is used for generic functions then the integration 
	limits should be first set. 
	"""
	def __init__(self,f):
		self.f = f
		from scipy.integrate import quad
		self._quad = quad
		self.set_limits()
		self.set_options()
	
	def set_limits(self,a=0.,b=1.):
		self._a=a
		self._b=b
	
	def integral(self,*args):
		if args: self._args = args

		self._integral, self._abserr = self._quad(func=self.f,a=self._a,
		b=self._b,args=self._args ,epsabs=self._epsabs, epsrel=self._epsrel)	

		return self._integral
	
	@property
	def abserr(self):
		return self._abserr
	
	def set_options(self,epsabs=1.49e-8,epsrel=1.49e-8): #1.49e-8
		self._epsabs = epsabs
		self._epsrel = epsrel
		return
	
	def __call__(self,x,*args):
		self._args = args

		return self.f(x,*args)
		

def get_angles(ras,dec,ras_d = np.pi+96.4*np.pi/180., dec_d = 29.3*np.pi/180.):
	"""
	For a fixed choice of coordinates of centre of the universe sets the 
	angles along which the geodesics will be solved. Centre direction corresponds 
	to the dipole axis hence d subscript for its declination and right ascension. 
	The following relationships between right ascention, declination and 
	theta, phi coordinates of the LTB model hold:
	theta = pi/2- dec where -pi/2 <= dec <= pi/2
	phi   = ras       where 0 <= ras < 2pi
	ras:
		is in degrees
	dec:
		is in degrees
	gammas:
	       the angle between the tangent vector to the geodesic and a unit 
	       vector pointing from the observer to the void centre.
	ras_d, dec_d:
	      are the right ascension and declination of the void center i.e dipole
	returns:
	        dec, ras, gammas
	        dec = bee , ras = ell (in radians)
	"""
	#pi = np.pi
	#dec_d = 29.3*pi/180.
	#ras_d = pi+96.4*pi/180.
	
	#num_pt = 1 #3 #123
	#dec = np.linspace(-pi/2.,pi/2.,num_pt,endpoint=True)
	#ras = np.linspace(0.,2.*pi,num_pt,endpoint=False)
	#ras, dec = np.loadtxt("pixel_center_galactic_coord_12288.dat",unpack=True)
	#convert to radians
	ras = np.pi*ras/180.
	dec = np.pi*dec/180.
	#dec, ras = np.meshgrid(dec,ras)
	gammas = np.arccos( np.sin(dec)*np.sin(dec_d) + 
	                    np.cos(dec)*np.cos(dec_d)*np.cos(ras-ras_d))
	return  ras, dec, gammas #dec.flatten(), ras.flatten(), gammas.flatten()

