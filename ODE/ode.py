# Automated check that you use the correct Python version #
from sys import version_info
if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################

import numpy as np
from typing import Callable
from scipy.optimize import fsolve

def forward_difference_quotient(f: Callable[[float],float], x: float, h:float) -> float:
    """
        This method shall implement the forward difference quotient
    Input:
        f: Callable -> Function of which the derivative shall be approximated
        x: float    -> Position at which the derivative shall be approximated
        h: float    -> Stepwidth
    Output:
        fdq: float  -> Approximation of f'(x)
    """
    fdq = 0
    return fdq

def backward_difference_quotient(f: Callable[[float],float], x: float, h:float) -> float:
    """
        This method shall implement the backward difference quotient
    Input:
        f: Callable -> Function of which the derivative shall be approximated
        x: float    -> Position at which the derivative shall be approximated
        h: float    -> Stepwidth
    Output:
        fdq: float  -> Approximation of f'(x)
    """
    bdq = 0
    return bdq

def central_difference_quotient(f: Callable[[float],float], x: float, h:float) -> float:
    """
        This method shall implement the central difference quotient
    Input:
        f: Callable -> Function of which the derivative shall be approximated
        x: float    -> Position at which the derivative shall be approximated
        h: float    -> Stepwidth
    Output:
        cdq: float  -> Approximation of f'(x)
    """
    cdq = 0
    return cdq

def explicit_euler(rhs: Callable[[float,float], float], y0: float, t0: float, T: float, N: int) -> [np.ndarray[float], np.ndarray[float]]:
    """
        This method shall implement the explicit Euler method.
    Input:
        rhs : Callable -> Right-hand side of the ODE y'(t) = rhs(t, y)
        y0 : float     -> Initial value y(t0)
        t0 : float     -> Initial moment in time
        T : float      -> Final moment in time
        N : int        -> Number of time steps
    Output:
        t : np.ndarray -> Array of moments in time
        y : np.ndarray -> Array of the solution
    """
    t = []
    y = []
    return t, y


def implicit_euler(rhs: Callable[[float,float], float], y0: float, t0: float, T: float, N: int) -> [np.ndarray[float], np.ndarray[float]]:
    """
        This method shall implement the implicit Euler method.
    Input:
        rhs : Callable -> Right-hand side of the ODE y'(t) = rhs(t, y)
        y0 : float     -> Initial value y(t0)
        t0 : float     -> Initial moment in time
        T : float      -> Final moment in time
        N : int        -> Number of time steps
    Output:
        t : np.ndarray -> Array of moments in time
        y : np.ndarray -> Array of the solution
    """
    t = []
    y = []
    return t, y
