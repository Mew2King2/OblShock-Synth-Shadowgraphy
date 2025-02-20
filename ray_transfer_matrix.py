import numpy as np
import matplotlib.pyplot as plt

'''
Example:

###INITIALISE RAYS###
#Rays are a 4 vector of x, theta, y, phi,
#here we initialise 10*7 randomly distributed rays
rr0=np.random.rand(6,int(1e7))
rr0[0,:]-=0.5 #rand generates [0,1], so we recentre [-0.5,0.5]
rr0[2,:]-=0.5

rr0[4,:]-=0.5 #rand generates [0,1], so we recentre [-0.5,0.5]
rr0[5,:]-=0.5

#x, θ, y, ϕ
scales=np.diag(np.array([10,0,10,0,1,1j])) #set angles to 0, collimated beam. x, y in [-5,5]. Circularly polarised beam, E_x = iE_y
rr0=np.matmul(scales, rr0)
r0=circular_aperture(5, rr0) #cut out a circle

### Shadowgraphy, no polarisation
## object_length: determines where the focal plane is. If you object is 10 mm long, object length = 5 will
## make the focal plane in the middle of the object. Yes, it's a bad variable name.
s = Shadowgraphy(rr0, L = 400, R = 25, object_length=5)
s.solve()
s.histogram(bin_scale = 25)
fig, axs = plt.subplots(figsize=(6.67, 6))

cm='gray'
clim=[0,100]

s.plot(axs, clim=clim, cmap=cm)

'''
def m_to_mm(r):
    rr = np.ndarray.copy(r)
    rr[0::2,:]*=1e3
    return rr

def lens(r, f1,f2):
    '''4x4 matrix for a thin lens, focal lengths f1 and f2 in orthogonal axes
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    l1= np.array([[1,    0],
                [-1/f1, 1]])
    l2= np.array([[1,    0],
                [-1/f2, 1]])
    L=np.zeros((4,4))
    L[:2,:2]=l1
    L[2:,2:]=l2

    return np.matmul(L, r)

def sym_lens(r, f):
    '''
    helper function to create an axisymmetryic lens
    '''
    return lens(r, f,f)

def distance(r, d):
    '''4x4 matrix  matrix for travelling a distance d
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    d = np.array([[1, d],
                    [0, 1]])
    L=np.zeros((4,4))
    L[:2,:2]=d
    L[2:,2:]=d
    return np.matmul(L, r)

def circular_aperture(r, R):
    '''
    Filters rays to find those inside a radius R
    '''
    filt = r[0,:]**2+r[2,:]**2 > R**2
    r[:,filt]=None
    return r

def circular_stop(r, R):
    '''
    Filters rays to find those inside a radius R
    '''
    filt = r[0,:]**2+r[2,:]**2 < R**2
    r[:,filt]=None
    return r

def rect_aperture(r, Lx, Ly):
    '''
    Filters rays inside a rectangular aperture, total size 2*Lx x 2*Ly
    '''
    filt1 = (r[0,:]**2 > Lx**2)
    filt2 = (r[2,:]**2 > Ly**2)
    filt=filt1*filt2
    r[:,filt]=None
    return r

def knife_edge(r, axis, edge = 0):
    '''
    Filters rays using a knife edge.
    Default is a knife edge in y, can also do a knife edge in x.
    '''
    #if axis is 'y':
    # LSH: changed below line from 'is' to '=='
    if axis == 'y':
        a=2
    else:
        a=0
    filt = r[a,:] < edge
    r[:,filt]=None
    return r


def myfunc(a,b, *args, **kwargs):
      c = kwargs.get('c', None)
      d = kwargs.get('d', None)
      #etc


class Rays:
    """
    Inheritable class for ray diagnostics.
    """
    def __init__(self, r0, focal_plane = 0, L=400, R=25, Lx=18, Ly=13.5, **kwargs):
        """Initialise ray diagnostic.

        Args:
            r0 (4xN float array): N rays, [x, theta, y, phi]

            L (int, optional): Length scale L. First lens is at L. Defaults to 400.
            R (int, optional): Radius of lenses. Defaults to 25.
            Lx (int, optional): Detector size in x. Defaults to 18.
            Ly (float, optional): Detector size in y. Defaults to 13.5.
        """        
        self.focal_plane, self.L, self.R, self.Lx, self.Ly = focal_plane, L, R, Lx, Ly
        self.r0 = m_to_mm(r0)
        # 25 July 2023, Lansing edit
        self.f = kwargs.get('f', None)  # mm, focal length of lens
        self.M = kwargs.get('M', None)  # magnification factor
    def histogram(self, bin_scale=10, pix_x=3448, pix_y=2574, clear_mem=False):
        """Bin data into a histogram. Defaults are for a KAF-8300.
        Outputs are H, the histogram, and xedges and yedges, the bin edges.

        Args:
            bin_scale (int, optional): bin size, same in x and y. Defaults to 10.
            pix_x (int, optional): number of x pixels in detector plane. Defaults to 3448.
            pix_y (int, optional): number of y pixels in detector plane. Defaults to 2574.
        """        
        x=self.rf[0,:]
        y=self.rf[2,:]

        x=x[~np.isnan(x)]
        y=y[~np.isnan(y)]

        self.H, self.xedges, self.yedges = np.histogram2d(x, y, 
                                           bins=[pix_x//bin_scale, pix_y//bin_scale], 
                                           range=[[-self.Lx, self.Lx],[-self.Ly,self.Ly]])  # range=[[-self.Lx/2, self.Lx/2],[-self.Ly/2,self.Ly/2]]
        self.H = self.H.T

        # Optional - clear ray attributes to save memory
        if(clear_mem):
            self.clear_rays()

    def plot(self, ax, clim=None, cmap=None):
        ax.imshow(self.H, interpolation='nearest', origin='lower', clim=clim, cmap=cmap,
                extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
        #ax.set_xlabel('x (mm)')
        #ax.set_ylabel('y (mm)')

    def clear_rays(self):
        '''
        Clears the r0 and rf variables to save memory
        '''
        self.r0 = None
        self.rf = None
        
class Shadowgraphy(Rays):
    def solve(self):
        rl1=np.matmul(distance(self.L - self.focal_plane), self.r0) #displace rays to lens. Accounts for object with depth
        rc1=circular_aperture(self.R, rl1) # cut off
        r2=np.matmul(sym_lens(self.L/2), rc1) #lens 1

        rl2=np.matmul(distance(3*self.L/2), r2) #displace rays to lens 2.
        rc2=circular_aperture(self.R, rl2) # cut off
        r3=np.matmul(sym_lens(self.L/3), rc2) #lens 2

        rd3=np.matmul(distance(self.L), r3) #displace rays to detector

        self.rf = rd3


#
#
# 2024.06.30, Lansing edit
#
#

#
class RaysOneLens:
    """
    Inheritable class for ray diagnostics.
    """
    # def __init__(self, r0, focal_plane = 0, L=400, R=25, Lx=18, Ly=13.5, **kwargs):
    # r0, r0_at_y_mm, focus_at_y_mm, ueff, f, R, v, Lx, Ly
    def __init__(self, r0, ueff, f, R, v, **kwargs):
        """Initialise ray diagnostic.

        Args:
            r0 (4xN float array): [x, theta_x, z, phi_z] x [N rays]
            ueff (float): the *effective* distance that the rays travel (from r0_at_y_mm to y-plane of lens), mm
            f, R (floats): focal length + radius of lens, mm
            v (float): distance from lens to camera CCD sensor, mm
        """
        self.r0 = m_to_mm(r0)
        self.ueff, self.f, self.R, self.v = ueff, f, R, v

#
class RaysTwoLens:
    """
    Inheritable class for ray diagnostics.
    """
    # def __init__(self, r0, focal_plane = 0, L=400, R=25, Lx=18, Ly=13.5, **kwargs):
    def __init__(self, r0, u1eff, f1, R1, d, f2, R2, v2, **kwargs):
        """Initialise ray diagnostic.

        Args:
            r0 (4xN float array): [x, theta_x, z, phi_z] x [N rays]
            u1eff (float): the *effective* distance that the rays travel (from r0_at_y_mm to y-plane of first lens), mm
            f1, R1 (floats): focal length + radius of first lens, mm
            d (float): distance between first and second lens, mm
            f2, R2 (floats): focal length + radius of second lens, mm
            v2 (float): distance from second lens to camera CCD sensor, mm
        """
        self.r0 = m_to_mm(r0)
        self.u1eff, self.f1, self.R1 = u1eff, f1, R1
        self.d, self.f2, self.R2, self.v2 = d, f2, R2, v2

