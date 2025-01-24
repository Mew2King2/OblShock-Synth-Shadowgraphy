"""PARTICLE TRACKER
BASED ON: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.61.895

SOLVES: 
$ \frac{d\vec{v}}{dt} = -\nabla \left( \frac{c^2}{2} \frac{n_e}{n_c} \right) $

$ \frac{d\vec{x}}{dt} = \vec{v} $

CODE BY: Aidan CRILLY
REFACTORING: Jack HARE

EXAMPLES:
#############################
#NULL TEST: no deflection
import particle_tracker as pt

N_V = 100
M_V = 2*N_V+1
ne_extent = 5.0e-3
ne_x = np.linspace(-ne_extent,ne_extent,M_V)
ne_y = np.linspace(-ne_extent,ne_extent,M_V)
ne_z = np.linspace(-ne_extent,ne_extent,M_V)

null = pt.ElectronCube(ne_x,ne_y,ne_z,ne_extent)
null.test_null()
null.calc_dndr()

### Initialise rays
s0 = pt.init_beam(Np = 100000, beam_size=5e-3, divergence = 0.5e-3, ne_extent = ne_extent)
### solve
null.solve(s0)
rf = null.rf

### Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
nbins = 201

_,_,_,im1 = ax1.hist2d(rf[0]*1e3, rf[2]*1e3, bins=(nbins, nbins), cmap=plt.cm.jet);
plt.colorbar(im1,ax=ax1)
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
_,_,_,im2 = ax2.hist2d(rf[1]*1e3, rf[3]*1e3, bins=(nbins, nbins), cmap=plt.cm.jet);
plt.colorbar(im2,ax=ax2)
ax2.set_xlabel(r"$\theta$ (mrad)")
ax2.set_ylabel(r"$\phi$ (mrad)")

fig.tight_layout()

###########################
#SLAB TEST: Deflect rays in -ve x-direction
import particle_tracker as pt

N_V = 100
M_V = 2*N_V+1
ne_extent = 6.0e-3
ne_x = np.linspace(-ne_extent,ne_extent,M_V)
ne_y = np.linspace(-ne_extent,ne_extent,M_V)
ne_z = np.linspace(-ne_extent,ne_extent,M_V)

slab = pt.ElectronCube(ne_x,ne_y,ne_z,ne_extent)
slab.test_slab(s=10, n_e0=1e25)
slab.calc_dndr()

## Initialise rays and solve
s0 = pt.init_beam(Np = 100000, beam_size=5e-3, divergence = 0, ne_extent = ne_extent)
slab.solve(s0)
rf = slab.rf

## Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
nbins = 201

_,_,_,im1 = ax1.hist2d(rf[0]*1e3, rf[2]*1e3, bins=(nbins, nbins), cmap=plt.cm.jet);
plt.colorbar(im1,ax=ax1)
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
_,_,_,im2 = ax2.hist2d(rf[1]*1e3, rf[3]*1e3, bins=(nbins, nbins), cmap=plt.cm.jet);
plt.colorbar(im2,ax=ax2)
ax2.set_xlabel(r"$\theta$ (mrad)")
ax2.set_ylabel(r"$\phi$ (mrad)")

fig.tight_layout()
"""

import numpy as np
from scipy.integrate import odeint,solve_ivp
from scipy.interpolate import RegularGridInterpolator
from time import time
import scipy.constants as sc
import pickle
from datetime import datetime

c = sc.c # honestly, this could be 3e8 *shrugs*

class ElectronCube:
    """A class to hold and generate electron density cubes
    """
    
    def __init__(self, x, y, z, probing_direction = 'z'):
        """
        Example:
            N_V = 100
            M_V = 2*N_V+1
            ne_extent = 5.0e-3
            ne_x = np.linspace(-ne_extent,ne_extent,M_V)
            ne_y = np.linspace(-ne_extent,ne_extent,M_V)
            ne_z = np.linspace(-ne_extent,ne_extent,M_V)

        Args:
            x (float array): x coordinates, m
            y (float array): y coordinates, m
            z (float array): z coordinates, m
            extent (float): physical size, m
        """
        self.z,self.y,self.x = z, y, x
        self.extent_x = x.max()
        self.extent_y = y.max()
        self.extent_z = z.max()
        self.probing_direction = probing_direction
        
    def test_null(self):
        """
        Null test, an empty cube
        """
        # LSH: np.meshgrid --> "Return coordinate matrices from coordinate vectors"
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        # LSH: electron density for an empty cube (no electrons)
        # LSH: np.zeros_like --> "Return an array of zeros with the same shape and type as a given array."
        self.ne = np.zeros_like(self.XX)
        
    def test_slab(self, s=1, n_e0=2e23):
        """A slab with a linear gradient in x:
        n_e =  n_e0 * (1 + s*x/extent)

        Will cause a ray deflection in x

        Args:
            s (int, optional): scale factor. Defaults to 1.
            n_e0 ([type], optional): mean density. Defaults to 2e23 m^-3.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = n_e0*(1.0+s*self.XX/self.extent_x)

    # LSH Lansing edit, 27 July 2023
    def test_slab_fixed(self, s=1, n_e0=2e23):
        """A slab with a linear gradient in x:
        n_e =  n_e0 * (1 + s*x/extent)

        Will cause a ray deflection in x

        Args:
            s (int, optional): scale factor. Defaults to 1.
            n_e0 ([type], optional): mean density. Defaults to 2e23 m^-3.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = n_e0*(1.0+s*(self.XX + self.extent_x)/self.extent_x)
        
    # LSH: 03 May 2023: added test example case
    def test_inverse_slab(self, s=1, n_e0=2e23):
        # 
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        RR_XY = np.sqrt(self.XX**2 + self.YY**2)
        extent_XY = np.mean([self.extent_x, self.extent_y])
        self.ne = n_e0*(1.0+s*RR_XY/extent_XY)

    # LSH: 10 May 2023: added test example case
    def test_elliptic_Gaussian(self, s=1, n_e0=2e23, delta_x=0.5e-3, L_y=20e-3):
        # 
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = n_e0 * np.exp( -( s * self.XX / (2*delta_x) )**2 ) * np.exp( -( s * self.YY / (2*L_y) )**2 )
    
    # LSH: 10 May 2023: added test example case
    def test_ellipticXZ_Gaussian(self, s=1, n_e0=2e23, delta_x=0.5e-3, L_z=18e-3):
        # 
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        #self.ne = n_e0 * np.exp( -( s * self.XX / delta_x )**2 ) * np.exp( -( s * self.ZZ / L_z )**2 )
        self.ne = n_e0 * np.exp( -( s * self.XX / (2*delta_x) )**2 ) * np.exp( -( s * self.ZZ / (2*L_z) )**2 )

    # LSH: 9 March 2024: added test example case
    def elliptic_Gaussian_XrayZ_constrayY(self, n_0=1e24, L_x=1e-3, L_z=10e-3):  # elliptic_Gaussian_XmarzY_constmarzZ
        # 
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        #self.ne = n_0 * np.exp( -( self.XX / L_x )**2 ) * np.exp( -( self.ZZ / L_z )**2 )  # before 13 March 2024
        self.ne = n_0 * np.exp( -( self.XX / (np.sqrt(2) * L_x) )**2 ) * np.exp( -( self.ZZ / (np.sqrt(2) * L_z) )**2 )  # after 13 March 2024, I think there should be a factor of two in the denomenator here?  https://mathworld.wolfram.com/GaussianFunction.html

    # LSH: 9 March 2024: added test example case
    def dual_wire_array_div_mass(self, n_edge=1e28, array_cent_m=30e-3, L_x=1e-3):  # elliptic_Gaussian_XmarzY_constmarzZ
        '''
        array_diam_m = 40 * m_per_mm  # m
        array_cent_m = 30 * m_per_mm  # m
        right_array_edge_m = 10 * m_per_mm  # m
        left_array_edge_m = -10 * m_per_mm  # m
        zeros_XX, zeros_YY, zeros_ZZ = np.zeros(self.XX.shape), np.zeros(self.YY.shape), np.zeros(self.ZZ.shape)
        '''
        # 
        #RR_XrayZ = np.sqrt( (self.XX)**2 + self.ZZ**2 )  # m, radial distance away from central (0,0) origin
        RR1_XrayZ = np.sqrt( (self.XX - array_cent_m)**2 + self.ZZ**2 )  # m, radial distance away from array #1
        RR2_XrayZ = np.sqrt( (self.XX + array_cent_m)**2 + self.ZZ**2 )  # m, radial distance away from array #2
        # filters/masks
        mask1 = abs(RR1_XrayZ) < array_cent_m - L_x
        mask2 = abs(RR2_XrayZ) < array_cent_m - L_x
        #mask1 = abs(self.XX) > L_x
        #mask2 = mask1
        # 
        pwr = 2.
        c_0 = (abs(array_cent_m - self.extent_x))**pwr * n_edge
        wire1_ne = c_0 * (1/(RR1_XrayZ)**pwr) * mask1  # linear rather than 1/R^2 b/c manual adjustment
        wire2_ne = c_0 * (1/(RR2_XrayZ)**pwr) * mask2
        # 
        self.ne += wire1_ne
        self.ne += wire2_ne

    def test_linear_cos(self,s1=0.1,s2=0.1,n_e0=2e23,Ly=1):
        """Linearly growing sinusoidal perturbation

        Args:
            s1 (float, optional): scale of linear growth. Defaults to 0.1.
            s2 (float, optional): amplitude of sinusoidal perturbation. Defaults to 0.1.
            n_e0 ([type], optional): mean electron density. Defaults to 2e23 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = n_e0*(1.0+s1*self.XX/self.extent_x)*(1+s2*np.cos(2*np.pi*self.YY/Ly))
        
    def test_exponential_cos(self,n_e0=1e24,Ly=1e-3, s=2e-3):
        """Exponentially growing sinusoidal perturbation

        Args:
            n_e0 ([type], optional): mean electron density. Defaults to 2e23 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1e-3 m.
            s ([type], optional): scale of exponential growth. Defaults to 2e-3 m.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = n_e0*10**(self.XX/s)*(1+np.cos(2*np.pi*self.YY/Ly))

    def test_lens(self,n_e0=1e24,LR=1e-3):
        """Exponentially growing sinusoidal perturbation

        Args:
            n_e0 ([type], optional): mean electron density. Defaults to 2e23 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1e-3 m.
            s ([type], optional): scale of exponential growth. Defaults to 2e-3 m.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        RR = np.sqrt(self.XX**2 + self.YY**2)
        self.ne = n_e0*np.exp(-RR**2/LR**2)
        
    def external_ne(self, ne):
        """Load externally generated grid

        Args:
            ne ([type]): MxMxM grid of density in m^-3
        """
        self.ne = ne

    def external_ne_LSH(self, ne):
        """Load externally generated grid

        Args:
            ne ([type]): MxMxM grid of density in m^-3
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = ne

    def calc_dndr(self, lwl):
        """Generate interpolators for derivatives.

        Args:
            lwl (float, optional): laser wavelength. Defaults to 1053e-9 m.
        """
        # lwl=1053e-9
        # LSH:  omega = 2 * pi * c / lambda
        # angular frequency = 2 * pi * f (ordinary frequency) = 2 * pi * (c / lambda)
        self.omega = 2*np.pi*(c/lwl)
        # critical/cutoff density; 
        # "density at which the plasma frequency equals the frequency of an electromagnetic electron wave in plasma"
        # n_cutoff = (m / 4 * pi * e^2) * omega^2 (?)
        # unclear if this is coefficient is from
        nc = 3.14207787e-4*self.omega**2

        # 
        self.ne_nc = self.ne/nc #normalise to critical density
        
        #More compact notation is possible here, but we are explicit
        # LSH: some sort of derivative of the normalized density in x/y/z
        self.dndx = -0.5*c**2*np.gradient(self.ne_nc,self.x,axis=0)
        self.dndy = -0.5*c**2*np.gradient(self.ne_nc,self.y,axis=1)
        self.dndz = -0.5*c**2*np.gradient(self.ne_nc,self.z,axis=2)
        
        # LSH: args: points, values, if outside bounds throw error or not, what value to take outside bounds
        # LSH: interpolates the discrete gradient values
        self.dndx_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndx, bounds_error = False, fill_value = 0.0)
        self.dndy_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndy, bounds_error = False, fill_value = 0.0)
        self.dndz_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndz, bounds_error = False, fill_value = 0.0)

 
    # Plasma refractive index
    def n_refrac(self):
        def omega_pe(ne):
            '''Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28'''
            return 5.64e4*np.sqrt(ne)
        ne_cc = self.ne*1e-6
        o_pe  = omega_pe(ne_cc)
        o_pe[o_pe > self.omega] = self.omega # remove densities greater than critical density - ray tracing breaks down here anyway.
        return np.sqrt(1.0-(o_pe/self.omega)**2)
   
    def dndr(self,x):
        """returns the gradient at the locations x

        Args:
            x (3xN float): N [x,y,z] locations

        Returns:
            3 x N float: N [dx,dy,dz] electron density gradients
        """
        grad = np.zeros_like(x)
        grad[0,:] = self.dndx_interp(x.T)
        grad[1,:] = self.dndy_interp(x.T)
        grad[2,:] = self.dndz_interp(x.T)
        return grad

    # LSH: ex:  "null.init_beam(Np = 100000, beam_size= 2e-3, divergence = 5e-3)"
    def init_beam(self, Np, beam_size, divergence):
        """[summary]

        Args:
            Np (int): Number of photons
            beam_size (float): beam radius, m
            divergence (float): beam divergence, radians
            ne_extent (float): size of electron density cube, m. Used to back propagate the rays to the start
            probing_direction (str): direction of probing. I suggest 'z', the best tested

        Returns:
            s0, 6 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s
        """
        # LSH: rows to store (x, y, z, vx, vy, vz), cols to store each photon
        s0 = np.zeros((6,Np))
        # position, uniformly within a circle
        # LSH: 2 * pi * {random value between 0 and 1}
        t  = 2*np.pi*np.random.rand(Np) #polar angle of position
        # LSH: {random value between 0 and 1} + {random value between 0 and 1} = {value between 0 and 2}
        u  = np.random.rand(Np)+np.random.rand(Np) # radial coordinate of position
        # LSH: changes {1.4} to {0.6}, {1.9} to {0.1}, {1.01} to {0.99}, etc. -- but, why?  idk
        u[u > 1] = 2-u[u > 1]
        # angle
        # LSH: pi * {random value between 0 and 1} = {random value between 0 and pi}, ray is going up or down
        ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
        # LSH: random.randn samples from standard normal (Gaussian) dist., mean = 0, std = 1
        χ = divergence*np.random.randn(Np) #polar angle of velocity

        if(self.probing_direction == 'x'):
            self.extent = self.extent_x
            # Initial velocity
            s0[3,:] = c * np.cos(χ)
            s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = self.extent
            s0[1,:] = beam_size*u*np.cos(t)
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'y'):
            self.extent = self.extent_y
            # Initial velocity
            s0[4,:] = c * np.cos(χ)
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = beam_size*u*np.cos(t)
            s0[1,:] = -self.extent
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'z'):
            # LSH: physical size/length/thickness in z
            self.extent = self.extent_z
            # Initial velocity
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
            s0[5,:] = c * np.cos(χ)
            # Initial position
            s0[0,:] = beam_size*u*np.cos(t)
            s0[1,:] = beam_size*u*np.sin(t)
            s0[2,:] = -self.extent
        # LSH: rows to store (x, y, z, vx, vy, vz), cols to store each photon
        self.s0 = s0

    # LSH: ex:  "null.init_beam(Np = 100000, beam_size= 2e-3, divergence = 5e-3)"
    def init_beam_seed(self, Np, beam_size, divergence, seed=1234567):
        """[summary]

        Args:
            Np (int): Number of photons
            beam_size (float): beam radius, m
            divergence (float): beam divergence, radians
            ne_extent (float): size of electron density cube, m. Used to back propagate the rays to the start
            probing_direction (str): direction of probing. I suggest 'z', the best tested

        Returns:
            s0, 6 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s
        """
        # LSH: addition, add a fixed seed for all random number generations
        np.random.seed(seed)
        # LSH: rows to store (x, y, z, vx, vy, vz), cols to store each photon
        s0 = np.zeros((6,Np))
        # position, uniformly within a circle
        # LSH: 2 * pi * {random value between 0 and 1}
        t  = 2*np.pi*np.random.rand(Np) #polar angle of position
        # LSH: {random value between 0 and 1} + {random value between 0 and 1} = {value between 0 and 2}
        u  = np.random.rand(Np)+np.random.rand(Np) # radial coordinate of position
        # LSH: changes {1.4} to {0.6}, {1.9} to {0.1}, {1.01} to {0.99}, etc. -- but, why?  idk
        u[u > 1] = 2-u[u > 1]
        # angle
        # LSH: pi * {random value between 0 and 1} = {random value between 0 and pi}, ray is going up or down
        ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
        # LSH: random.randn samples from standard normal (Gaussian) dist., mean = 0, std = 1
        χ = divergence*np.random.randn(Np) #polar angle of velocity

        if(self.probing_direction == 'x'):
            self.extent = self.extent_x
            # Initial velocity
            s0[3,:] = c * np.cos(χ)
            s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = self.extent
            s0[1,:] = beam_size*u*np.cos(t)
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'y'):
            self.extent = self.extent_y
            # Initial velocity
            s0[4,:] = c * np.cos(χ)
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = beam_size*u*np.cos(t)
            s0[1,:] = -self.extent
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'z'):
            # LSH: physical size/length/thickness in z
            self.extent = self.extent_z
            # Initial velocity
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
            s0[5,:] = c * np.cos(χ)
            # Initial position
            s0[0,:] = beam_size*u*np.cos(t)
            s0[1,:] = beam_size*u*np.sin(t)
            s0[2,:] = -self.extent
        # LSH: rows to store (x, y, z, vx, vy, vz), cols to store each photon
        self.s0 = s0

        
        
        
        

    # LSH: ex:  "null.init_beam(Np = 100000, beam_size= 2e-3, divergence = 5e-3)"
    def init_beam_rect_seed(self, Np, beam_size, divergence, seed=1234567):
        """[summary]

        Args:
            Np (int): Number of photons
            beam_size (float): rectangular beam size, [x_beam_width_m, z_beam_height_m]
            divergence (float): beam divergence, radians
            ne_extent (float): size of electron density cube, m. Used to back propagate the rays to the start
            probing_direction (str): direction of probing. I suggest 'z', the best tested

        Returns:
            s0, 6 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s
        """
        # LSH: addition, add a fixed seed for all random number generations
        np.random.seed(seed)
        # extract the rectangular beam dimensions from beam_size
        x_beam_width_m, z_beam_height_m = beam_size  # m
        # LSH: rows to store (x, y, z, vx, vy, vz), cols to store each photon
        s0 = np.zeros((6,Np))
        
        # position, uniformly within a rectangle (using XZ MARZ convention, so 'z' is this code's "height y")
        x_pos_samples = np.random.uniform(low = -x_beam_width_m, high = x_beam_width_m, size = Np)
        z_pos_samples = np.random.uniform(low = -z_beam_height_m, high = z_beam_height_m, size = Np)
        
        # angle
        # LSH: pi * {random value between 0 and 1} = {random value between 0 and pi}, ray is going up or down
        ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
        # LSH: random.randn samples from standard normal (Gaussian) dist., mean = 0, std = 1
        χ = divergence*np.random.randn(Np) #polar angle of velocity

        if(self.probing_direction == 'x'):
            self.extent = self.extent_x
            # Initial velocity
            s0[3,:] = c * np.cos(χ)
            s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = self.extent
            s0[1,:] = beam_size*u*np.cos(t)
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'y'):
            self.extent = self.extent_y
            # Initial velocity
            s0[4,:] = c * np.cos(χ)
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = beam_size*u*np.cos(t)
            s0[1,:] = -self.extent
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'z'):
            # LSH: physical size/length/thickness in z
            self.extent = self.extent_z
            # Initial velocity
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
            s0[5,:] = c * np.cos(χ)
            # Initial position
            s0[0,:] = x_pos_samples  # beam_size*u*np.cos(t)
            s0[1,:] = z_pos_samples  # beam_size*u*np.sin(t)
            s0[2,:] = -self.extent
        # LSH: rows to store (x, y, z, vx, vy, vz), cols to store each photon
        self.s0 = s0

        



    # LSH: ex:  "null.init_beam(Np = 100000, beam_size= 2e-3, divergence = 5e-3)"
    def init_beam_rect(self, Np, beam_size, divergence):
        """[summary]

        Args:
            Np (int): Number of photons
            beam_size (float): rectangular beam size, [x_beam_width_m, z_beam_height_m]
            divergence (float): beam divergence, radians
            ne_extent (float): size of electron density cube, m. Used to back propagate the rays to the start
            probing_direction (str): direction of probing. I suggest 'z', the best tested

        Returns:
            s0, 6 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s
        """
        # LSH: addition, add a fixed seed for all random number generations
        #np.random.seed(seed)
        # extract the rectangular beam dimensions from beam_size
        x_beam_width_m, z_beam_height_m = beam_size  # m
        # LSH: rows to store (x, y, z, vx, vy, vz), cols to store each photon
        s0 = np.zeros((6,Np))
        
        # position, uniformly within a rectangle (using XZ MARZ convention, so 'z' is this code's "height y")
        x_pos_samples = np.random.uniform(low = -x_beam_width_m, high = x_beam_width_m, size = Np)
        z_pos_samples = np.random.uniform(low = -z_beam_height_m, high = z_beam_height_m, size = Np)
        
        # angle
        # LSH: pi * {random value between 0 and 1} = {random value between 0 and pi}, ray is going up or down
        ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
        # LSH: random.randn samples from standard normal (Gaussian) dist., mean = 0, std = 1
        χ = divergence*np.random.randn(Np) #polar angle of velocity

        if(self.probing_direction == 'x'):
            self.extent = self.extent_x
            # Initial velocity
            s0[3,:] = c * np.cos(χ)
            s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = self.extent
            s0[1,:] = beam_size*u*np.cos(t)
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'y'):
            self.extent = self.extent_y
            # Initial velocity
            s0[4,:] = c * np.cos(χ)
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = beam_size*u*np.cos(t)
            s0[1,:] = -self.extent
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'z'):
            # LSH: physical size/length/thickness in z
            self.extent = self.extent_z
            # Initial velocity
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
            s0[5,:] = c * np.cos(χ)
            # Initial position
            s0[0,:] = x_pos_samples  # beam_size*u*np.cos(t)
            s0[1,:] = z_pos_samples  # beam_size*u*np.sin(t)
            s0[2,:] = -self.extent
        # LSH: rows to store (x, y, z, vx, vy, vz), cols to store each photon
        self.s0 = s0
        
    
    # 5 March 2024, LSH edit, adding in diverging beam source (per David Yager email about first shadowgraphic images obtained on Z)
    def beam_diverger(self, z_pt_m):
        '''
        self = to get self.s0, the array of size (6 x Np) that stores each ray's (x, y, z, vx, vy, vz); we will shift the vx/vy/vz values based on ray(x,y)/marz(x,z) position
        #ref_angle_rad = the reference angle of divergence, the opening angle towards the ref_rad_m
        #ref_rad_m = the reference radial location where the max opening angle is acheived
        z_pt_m = the "point" where the laser beam diverges from (z-loc of the "point" relative to the target location, z = 0)
        #z_beam_m = z-loc of the initialized beam profile (relative to the target location, z = 0)
        '''
        # (x, ray-y/marz-z) = (s0[0,:], s0[1,:])
        rxy0 = np.sqrt( self.s0[0,:]**2 + self.s0[1,:]**2 )  # m, radius in the xy-plane
        cosϕ = self.s0[0,:] / rxy0  # cos of azimuthal angle, 0 to 2pi
        sinϕ = self.s0[1,:] / rxy0  # sin of azimuthal angle, 0 to 2pi
        χ = np.arctan( rxy0 / abs( z_pt_m ) )  # polar angle, 0 to pi (0 = pointing along ray-z/marz-y probe direction)  # abs( z_pt_m + z_beam_m )
        # altered velocities to accomodate beam divergence
        self.s0[3,:] = c * np.sin(χ) * cosϕ  # vx
        self.s0[4,:] = c * np.sin(χ) * sinϕ  # vy
        self.s0[5,:] = c * np.cos(χ)  # vz
        # update z location to z = 0, the target location
        self.s0[2,:] = 0.
        # store angles for each ray
        #self.cosϕ = cosϕ
        #self.sinϕ = sinϕ
        #self.χ = χ
    
    # 7 March 2024, LSH edit, adding in fxn to move from ray-z = 0 (the target plane) back to ray-z = -extent_z (the entrance plane of the 3D ne mesh)
    def beam_shifter(self):
        # calculate timesteps to move rays from ray-z = 0 back to ray-z = -extent_z
        Δt = abs( ( self.s0[2,:] - (-self.extent_z) ) / self.s0[5,:] )  # s, for each ray
        # altered x/y/z positions to move beam backwards in time from ray-z = 0 to ray-z = -extent_z
        self.s0[0,:] = self.s0[0,:] - self.s0[3,:] * Δt  # x, x_-44 = x_0 - v_x * dt
        self.s0[1,:] = self.s0[1,:] - self.s0[4,:] * Δt  # y, y_-44 = y_0 - v_y * dt
        self.s0[2,:] = -self.extent_z  # z
    
    
    def solve(self, method = 'RK45'):
        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume

        # LSH: linspace of times, from 0 to the time corresponding to all rays leaving volume (over-estimates diagonal max length?)
        t  = np.linspace(0.0,np.sqrt(8.0)*self.extent/c,2)

        # LSH: flattens the (6 x Np) s0 array to a (1 x 6Np) array
        s0 = self.s0.flatten() #odeint insists

        # LSH: records the time that calc starts
        start = time()
        # LSH: defines the ODE ds/dt by calling the dsdt() fxn
        # LSH: dsdt(t, s, ElectronCube)
        # LSH: t (float array): I think this is a dummy variable for ode_int - our problem is time invarient
        # LSH: s (6N float array): flattened 6xN array of rays used by ode_int
        # LSH: ElectronCube (ElectronCube): an ElectronCube object which can calculate gradients
        # LSH: returns -- 6N float array: flattened array for ode_int
        dsdt_ODE = lambda t, y: dsdt(t, y, self)
        sol = solve_ivp(dsdt_ODE, [0,t[-1]], s0, t_eval=t, method = method)
        # records the time that calc stops
        finish = time()
        # LSH
        # print("Ray trace completed in:\t",finish-start,"s")
        # LSH
        self.solve_time = finish-start  # s

        Np = s0.size//6
        # rehapes the flattened 6N float array back into (6 x Np) matrix
        self.sf = sol.y[:,-1].reshape(6,Np)
        # Fix amplitudes
        self.rf = self.ray_at_exit()
        return self.rf

    def ray_at_exit(self):
        """Takes the output from the 6D solver and returns 4D rays for ray-transfer matrix techniques.
        Effectively finds how far the ray is from the end of the volume, returns it to the end of the volume.

        Args:
            ode_sol (6xN float): N rays in (x,y,z,vx,vy,vz) format, m and m/s and amplitude, phase and polarisation
            ne_extent (float): edge length of cube, m
            probing_direction (str): x, y or z.

        Returns:
            [type]: [description]
        """
        ode_sol = self.sf
        Np = ode_sol.shape[1] # number of photons
        ray_p = np.zeros((4,Np))

        x, y, z, vx, vy, vz = ode_sol[0], ode_sol[1], ode_sol[2], ode_sol[3], ode_sol[4], ode_sol[5]

        # Resolve distances and angles
        # YZ plane
        if(self.probing_direction == 'x'):
            t_bp = (x-self.extent)/vx
            # Positions on plane
            ray_p[0] = y-vy*t_bp
            ray_p[2] = z-vz*t_bp
            # Angles to plane
            ray_p[1] = np.arctan(vy/vx)
            ray_p[3] = np.arctan(vz/vx)
        # XZ plane
        elif(self.probing_direction == 'y'):
            t_bp = (y-self.extent)/vy
            # Positions on plane
            ray_p[0] = x-vx*t_bp
            ray_p[2] = z-vz*t_bp
            # Angles to plane
            ray_p[1] = np.arctan(vx/vy)
            ray_p[3] = np.arctan(vz/vy)
        # XY plane
        elif(self.probing_direction == 'z'):
            t_bp = (z-self.extent)/vz
            # Positions on plane
            ray_p[0] = x-vx*t_bp
            ray_p[2] = y-vy*t_bp
            # Angles to plane
            ray_p[1] = np.arctan(vx/vz)
            ray_p[3] = np.arctan(vy/vz)

        return ray_p

    def save_output_rays(self, fn = None):
        """
        Saves the output rays as a binary numpy format for minimal size.
        Auto-names the file using the current date and time.
        """
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        if fn is None:
            fn = '{} rays.npy'.format(dt_string)
        else:
            fn = '{}.npy'.format(fn)
        with open(fn,'wb') as f:
            np.save(f, self.rf)

    def clear_memory(self):
        """
        Clears variables not needed by solve method, saving memory

        Can also use after calling solve to clear ray positions - important when running large number of rays

        """
        self.dndx = None
        self.dndx = None
        self.dndx = None
        self.ne = None
        self.ne_nc = None
        self.sf = None
        self.rf = None
    
def dsdt(t, s, ElectronCube):
    """Returns an array with the gradients and velocity per ray for ode_int. Cannot be a method of ElectronCube due to expected call signature for the ODE solver

    Args:
        t (float array): I think this is a dummy variable for ode_int - our problem is time invarient
        s (6N float array): flattened 6xN array of rays used by ode_int
        ElectronCube (ElectronCube): an ElectronCube object which can calculate gradients

    Returns:
        6N float array: flattened array for ode_int
    """
    # LSH: number of photons (rays) in simulation
    Np     = s.size//6
    # LSH: rehapes the flattened s vector back into a (6 x Np) matrix
    s      = s.reshape(6,Np)
    # LSH: copies structure of s and fills with 0s
    sprime = np.zeros_like(s)
    # Velocity and position
    # LSH: gets (vx, vy, vz) (rows) for all photons (columns)
    v = s[3:,:]
    # LSH: gets (x, y, z) (rows) for all photons (columns)
    x = s[:3,:]

    # LSH: gets e- density gradients for each ray
    # LSH: ElectronCube.dndr(x) = returns the gradient at the locations x
    # LSH: input -- x (3xN float): N [x,y,z] locations /// output -- 3 x N float: N [dx,dy,dz] electron density gradients
    sprime[3:6,:] = ElectronCube.dndr(x)
    # LSH: gets velocities for each ray
    sprime[:3,:]  = v

    return sprime.flatten()
