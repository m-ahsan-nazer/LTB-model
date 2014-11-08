#To be used for various chores that are better left outside of the natural flow
#in the code

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
	                maxiter=100, full_output=False, disp=True):
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
	
	def set_limits(self,a=1e-10,b=1.):
		self._a=a
		self._b=b
	
	def integral(self,*args):

		if  not args:
			self._integral, self._abserr = self._quad(func=self.f,a=self._a,b=self._b,
		                       args=self._args ,epsabs=self._epsabs,
		                       epsrel=self._epsrel)
		else:
			self._args = args
			self._integral, self._abserr = self._quad(func=self.f,a=self._a,b=self._b,
		                       args=self._args ,epsabs=self._epsabs,
		                       epsrel=self._epsrel)	
		
		return self._integral
	
	@property
	def abserr(self):
		return self._abserr
	
	def set_options(self,epsabs=1.49e-8,epsrel=1.49e-8):
		self._epsabs = epsabs
		self._epsrel = epsrel
		return
	
	def __call__(self,x,*args):
		self._args = args

		return self.f(x,*args)
		




