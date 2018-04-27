# coding: utf-8
import numpy as np
from scipy.optimize import bisect
from scipy.integrate import odeint

def shoot_for_eigenvalues(rhs, args, ic, get_interval_endpoint, n=3):
    """Find the energies allowed by the TISE for a particular potential.
    
    Parameters:
    -----------
    rhs : function
          RHS of the eigenproblem we're solving
    args : arguments for rhs (including, e.g., a potential function)
           does not include the eigenvalue
    ic : initial conditions to use
    get_interval_endpoint : function
                            generates the endpoint of the integration interval
    n : integer, optional
        how many energies to find; defaults to 3
        
    Returns:
    --------
    eigenvalues : list
                  eigenvalues
    """
    
    if not callable(rhs):
        raise TypeError("{} is not callable".format(rhs))
    eigenvalues = []
    bracket_found = False
    starting_value = 0.01
    e_step = 0.3
    
    # prime the pump
    prev_eigenvalue = starting_value
    endpoint = get_interval_endpoint(args, prev_eigenvalue)
    prev_bc = bc(prev_eigenvalue, rhs, args, ic, endpoint)
    
    for idx in range(n):
        # try values until we find two that bracket a correct eigenvalue
        while not bracket_found:
            current_eigenvalue = prev_eigenvalue + e_step
            endpoint = get_interval_endpoint(args, current_eigenvalue)
            current_bc = bc(current_eigenvalue, rhs, args, ic, endpoint)
            if np.sign(current_bc) != np.sign(prev_bc):
                bracket_found = True
            else:
                prev_eigenvalue = current_eigenvalue
                prev_bc = current_bc
        # use bisect to get the actual value
        eigenvalues.append(bisect(bc, prev_eigenvalue, current_eigenvalue,
                               args=(rhs, args, ic, endpoint)))
        
        # set up for the next eigenvalue
        # populate prev_eigenvalue
        prev_eigenvalue = current_eigenvalue
        prev_bc = current_bc
        # we no longer have a bracket
        bracket_found = False
    return eigenvalues

def bc(eigenvalue, rhs, args, ic, endpoint=5):
    """Find the boundary value for an eigenvalue problem.
    
    Parameters:
    -----------
    eigenvalue: number
                eigenvalue to test
    rhs : function
          evaluate the RHS of the eigenproblem we are solving
    args : iterable
           contains arguments for rhs
           (examples: the eigenvalue, a potential function, etc.)
    ic : array
         initial conditions for the integration
    endpoint : number, optional
               the endpoint for the integration interval.
    Returns:
    --------
    y : array
        The solution to the equation defined by rhs. 
    """
    domain = np.linspace(-endpoint,endpoint,1000)
        
    result = odeint(rhs, ic, domain, args=(eigenvalue, ) + args)
    return result[-1,0]
