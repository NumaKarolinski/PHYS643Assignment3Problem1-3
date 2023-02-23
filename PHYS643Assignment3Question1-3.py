import numpy as np
import matplotlib.pyplot as pl

# This is the implicit solution which solves the diffusion portion of the ODE
def solve_diffusion(beta, Ngrid, sigma):
    A = np.eye(Ngrid) + (1.0 + 2.0 * beta)+ np.eye(Ngrid, k=1) * -beta + np.eye(Ngrid, k=-1) * -beta
    sigma = np.linalg.solve(A, sigma)
    return sigma
# This generates the initial "sharp" gaussian of the surface density at t=0
def get_sigma_initial(radius):
    mean = 1.
    standard_deviation = 0.05
    return np.exp(np.divide(np.multiply(-1., np.power(np.subtract(radius, mean), 2.)), (2 * np.power(standard_deviation, 2.))))

# Set up the grid, advection parameters, and diffusion parameters
dx = 0.01
Ngrid = int(1./dx)
Nsteps = 1000
kinematic_viscosity = 0.01

# definition of diffusion coefficient based on assignment 1-3
diffusion_coefficient = 3. * kinematic_viscosity

# Radial distance (really this is x, where x = r / R_0)
radius = np.multiply(np.arange(0, 1., dx), 2.)

# definition of radial velocity based on assignment 1-3
velocity = np.divide(-3. * diffusion_coefficient / 2., radius[1:])

# Boundary conditions for velocity (guaranteed negative on the left, at [0],
# and guaranteed positive on the right, at [-1])
velocity = np.append([0], velocity)
velocity[0] = -np.abs(velocity[1])
velocity[-1] = np.abs(velocity[-2])

# Use “Courant” condition to decide dt
# dt should be based on the largest magnitude velocity
# thus np.min(abs(dx/velocity)) is the smallest possible value
dt = np.min(np.abs(np.divide(dx, velocity)))

# Definition of alpha and beta based on notes of numerical methods
alpha = np.multiply(velocity, dt / (2. * dx))
beta = diffusion_coefficient * dt / (dx ** 2.)

# In our case, surface density (sigma) is the fluid quantity that is
# being solved for in the numerical notes: 'f'
sigma = get_sigma_initial(radius)

# Set up plotting of surface density with radius
pl.ion()
fig, ax = pl.subplots(1, 1)

ax.plot(radius, sigma, 'k-', label = 'Initial Surface Density')

plt, = ax.plot(radius, sigma, 'ro', label = 'Surface Density')

ax.set_title('Surface density ratio plotted against radius ratio, animated over time')
ax.set_xlabel('Radius ratio, x = r/R_0')
ax.set_ylabel('Surface Density ratio (1)')
ax.legend()

fig.canvas.draw()

# Evolution
for ct in range(Nsteps):
    
    solve_diffusion(beta, Ngrid, sigma)

    # solve for advection evolution based on Lax-Friedrich method,
    # as done in class except that alpha needs to be selected over [1:(Ngrid - 1)]
    sigma[1:(Ngrid - 1)] = 0.5 * (sigma[2:] + sigma[:(Ngrid - 2)]) - alpha[1:(Ngrid - 1)] * (sigma[2:] - sigma[:(Ngrid - 2)])

    sigma[0] = sigma[1]
    sigma[-1] = sigma[-2]

    plt.set_ydata(sigma)

    ax.set_xlim([0., 2.])
    ax.set_ylim([0., 1.])

    fig.canvas.draw()
    pl.pause(0.0001)
