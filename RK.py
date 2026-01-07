import numpy as np
import matplotlib.pyplot as plt

def RKmethod(A, c, b, f, y0, T, delta_t):
    """runge-kutta method implementation.
    
    inputs:
        A: s x s lower triangular matrix
        c: vector in R^s with c[0] = 0
        b: vector in R^s
        f: function handle f(y, t)
        y0: initial condition
        T: final time
        delta_t: time step size
        
    outputs:
        t: time steps
        Y: approximate solutions
    """
    N = int(T / delta_t)  # number of steps
    t = np.linspace(0, T, N + 1)  # time steps
    Y = np.zeros(N + 1)  # solution array
    Y[0] = y0  # initial condition

    s = len(b)  # number of stages

    for n in range(N):
        k = np.zeros(s)  # stage values
        for i in range(s):
            ti = t[n] + c[i] * delta_t  # current time
            yi = Y[n] + delta_t * sum(A[i, j] * k[j] for j in range(i))  # intermediate y-value
            k[i] = f(yi, ti)  # compute stage
        Y[n + 1] = Y[n] + delta_t * sum(b[i] * k[i] for i in range(s))  # next y-value
    
    return t, Y  # return time steps and solutions

def f(y, t):
    """function for the ivp.
    
    inputs:
        y: current value
        t: current time
        
    outputs:
        derivative at (y, t)
    """
    return y * np.sin(t)  # define the function

def exact_solution(t):
    """exact solution for error calculation.
    
    inputs:
        t: time steps
        
    outputs:
        y: exact solution values
    """
    return -np.exp(1 - np.cos(t))  # exact solution

# butcher tableau for rk4
A_RK4 = np.array([[0, 0, 0, 0],
                  [1 / 2, 0, 0, 0],
                  [0, 1 / 2, 0, 0],
                  [0, 0, 1, 0]])
c_RK4 = np.array([0, 1 / 2, 1 / 2, 1])
b_RK4 = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])

# butcher tableau for chebyshev method
A_Cheb = np.array([[0, 0, 0],
                   [1 / 27, 0, 0],
                   [4 / 27, 0, 4 / 27]])
c_Cheb = np.array([0, 1 / 27, 4 / 27])
b_Cheb = np.array([0, 1, 1])

# initial condition and time
y0 = -1
T = 10
delta_ts = [1/4, 1/8, 1/16, 1/32, 1/64]

# store errors
errors = []

# apply rkmethod to the ivp for each delta_t
for delta_t in delta_ts:
    t, Y = RKmethod(A_RK4, c_RK4, b_RK4, f, y0, T, delta_t)  # call rkmethod
    exact_y = exact_solution(t[-1])  # exact solution at T = 10
    relative_error = np.abs((Y[-1] - exact_y) / exact_y)  # compute relative error
    errors.append(relative_error)  # store final relative error
    print(f"relative error for Δt={delta_t}: {relative_error}")

# log-log plot of the relative errors
plt.figure()
plt.loglog(delta_ts, errors, marker='o')
plt.title("relative error vs. time step size")
plt.xlabel("time step size (Δt)")
plt.ylabel("relative error")
plt.grid(True)
plt.show()

# compute rate of convergence
rates = []
for i in range(1, len(errors)):
    rate = np.log(errors[i-1] / errors[i]) / np.log(delta_ts[i-1] / delta_ts[i])
    rates.append(rate)
    print(f"rate of convergence between Δt={delta_ts[i-1]} and Δt={delta_ts[i]}: {rate}")

# amplification factor function for rk4
def amplification_factor_RK4(lam, delta_t):
    """compute amplification factor for rk4.
    
    inputs:
        lam: complex number
        delta_t: time step size
        
    outputs:
        G: computed amplification factor
    """
    return 1 + (delta_t * lam) / 6 + (delta_t * lam)**2 / 12 + (delta_t * lam)**3 / 24 + (delta_t * lam)**4 / 120

# stability plot for rk4
def stability_plot_RK4(limits, N):
    """plot stability region for rk4.
    
    inputs:
        limits: [xmin, xmax, ymin, ymax]
        N: number of grid points
        
    outputs: none
    """
    real_vals = np.linspace(limits[0], limits[1], N)  # real part range
    imag_vals = np.linspace(limits[2], limits[3], N)  # imaginary part range
    X, Y = np.meshgrid(real_vals, imag_vals)
    
    Z = X + 1j * Y  # complex grid
    G = amplification_factor_RK4(Z, 1)  # compute G
    
    stability_region = np.abs(G) <= 1  # stability condition
    
    plt.figure()
    plt.contourf(X, Y, stability_region, levels=[0, 1], colors=["blue", "white"])
    plt.title('stability region for RK4')
    plt.xlabel('real(lambda * delta_t)')
    plt.ylabel('imag(lambda * delta_t)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# amplification factor function for chebyshev
def amplification_factor_chebyshev(lam, delta_t):
    """compute amplification factor for chebyshev.
    
    inputs:
        lam: complex number
        delta_t: time step size
        
    outputs:
        G: computed amplification factor
    """
    return 1 + delta_t * lam + (delta_t * lam)**2 / 4

# stability plot for chebyshev
def stability_plot_chebyshev(limits, N):
    """plot stability region for chebyshev.
    
    inputs:
        limits: [xmin, xmax, ymin, ymax]
        N: number of grid points
        
    outputs: none
    """
    real_vals = np.linspace(limits[0], limits[1], N)  # real part range
    imag_vals = np.linspace(limits[2], limits[3], N)  # imaginary part range
    X, Y = np.meshgrid(real_vals, imag_vals)
    
    Z = X + 1j * Y  # complex grid
    G = amplification_factor_chebyshev(Z, 1)  # compute G
    
    stability_region = np.abs(G) <= 1  # stability condition
    
    plt.figure()
    plt.contourf(X, Y, stability_region, levels=[0, 1], colors=["green", "white"])
    plt.title('stability region for chebyshev')
    plt.xlabel('real(lambda * delta_t)')
    plt.ylabel('imag(lambda * delta_t)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# stability plots
limits_rk4 = [-4, 4, -4, 4]  # limits for rk4
limits_chebyshev = [-10, 1, -4, 4]  # limits for chebyshev
N = 800  # number of points

# stability plot for rk4
stability_plot_RK4(limits_rk4, N)

# stability plot for chebyshev
stability_plot_chebyshev(limits_chebyshev, N)

# apply both methods to the new ivp
def f_new_ivp(y, t):
    """function for the new ivp."""
    return -y  # define the function for the new ivp

# compare rk4 and chebyshev methods
def compare_methods():
    """compare rk4 and chebyshev methods."""
    T = 100  # total time
    y0 = 1  # initial condition
    delta_t = 10  # large step size
    
    # solve using rk4
    t_RK4, Y_RK4 = RKmethod(A_RK4, c_RK4, b_RK4, f_new_ivp, y0, T, delta_t)
    
    # solve using chebyshev
    t_Cheb, Y_Cheb = RKmethod(A_Cheb, c_Cheb, b_Cheb, f_new_ivp, y0, T, delta_t)
    
    # plot results
    plt.figure()
    plt.plot(t_RK4, Y_RK4, label="RK4")  # plot RK4 results
    plt.plot(t_Cheb, Y_Cheb, label="Chebyshev")  # plot Chebyshev results
    plt.legend()
    plt.title(f"solutions with large Δt={delta_t}")
    plt.xlabel("time t")
    plt.ylabel("solution y(t)")
    plt.grid(True)
    plt.show()

# compare the two methods
compare_methods()
