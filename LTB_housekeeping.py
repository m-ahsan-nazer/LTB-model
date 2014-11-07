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
		


