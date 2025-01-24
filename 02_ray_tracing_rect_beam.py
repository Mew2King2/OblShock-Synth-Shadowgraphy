"""
Example code for running on hpc using mpiexec

On top of usual requirements to run the particle tracking code it requires:

pickle
mpi4py

MPI allows multiple nodes to be used while multiprocessing does not
I also have more experience with MPI so know how to keep the memory usage low

Example PBS queue submission script to use 48 cores on 1 node with 2e7 rays per processor,
The resulting synthetic diagnostics are saved to the directory output

#!/bin/sh
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=48:mpiprocs=48:mem=128gb
#PBS -j oe
cd [HOME]/turbulence_tracing/particle_tracking

module load anaconda3/personal

mpiexec python example_MPI.py 2e7 ./output/

Using this set up took 2 hours of computing time for a total of 9.8e8 rays

There are a couple of variables which can be changed depending on your computing set up:

Np_ray_split - the number of rays per "bundle" considered, if you request number of rays per processor Np > Np_ray_split,
then the task is split such that bundles of Np_ray_split are considered until Np rays have been calculated,
this variable can be changed to fit into your memory budget. For 48 processors, 5e5 rays uses around 60 GB

bin_scale - this is the ratio of the computational to experimental pixel count, this can be reduced when more rays are considered

Outputs are pickled to keep their object structure, however information on the rays is not saved

"""

################################################################################


# 
import numpy as np
from mpi4py import MPI
import sys
import gc
#
import particle_tracker as pt
import ray_transfer_matrix as rtm

# 
import datetime
import resource
import os


################################################################################


m_per_nm = 1e-9
cm3_per_m3 = 1e6
m3_per_cm3 = 1/cm3_per_m3
m_per_mm = 1e-3
mm_per_m = 1/m_per_mm
m_per_cm = 1e-2
num_degs = 360
num_rads = 2 * np.pi
deg_per_rad = num_degs / num_rads
m_per_um = 1e-6
um_per_m = 1 / m_per_um
rad_per_mrad = 1e-3
mrad_per_rad = 1 / rad_per_mrad


################################################################################


# from 02_ray_tracing_rect_beam.slurm
# python -u ./${FILENAME}.py ${BASE_DIRECTORY} ${TID} ${NP} ${NP_RAY_SPLIT} ${X_BEAM_WIDTH} ${Z_BEAM_HEIGHT} ${BEAM_X_OFFSET} ${BEAM_Z_OFFSET} ${LAMBDA} ${DIVERGENCE} ${SCRATCH_DIRECTORY} ${SLURM_NTASKS}

## input directory
directory_to_load_npy_files = sys.argv[1]
## ns, time ID
tid = int(float(sys.argv[2]))
## number of rays per processors
Np = int(float(sys.argv[3]))
## number of rays at which the computation is split to save memory, default int(5e5)
Np_ray_split = int(float(sys.argv[4]))
## m, half-width and half-height of rectangular laser beam profile
x_beam_width_m, z_beam_height_m = float(sys.argv[5])/2 * m_per_mm, float(sys.argv[6])/2 * m_per_mm
beam_size = np.array( [x_beam_width_m, z_beam_height_m] )
## m, how much the circular beam is shifted in the +x/+z directions (to fall on top of the wedge target)
beam_x_offset, beam_z_offset = float(sys.argv[7]) * m_per_mm, float(sys.argv[8]) * m_per_mm
## m, laser photon wavelength
λ = float(sys.argv[9]) * m_per_nm
## 0 if no divergence, non-zero if overall laser beam is composed of several smaller diverging laser beams (bundle of smaller diverging lasers = one big laser)
divergence = float(sys.argv[10])
## output directory
output_dir = sys.argv[11]
## number of processors
num_cpus = int(float(sys.argv[12]))

# params to load .npy 3D e- density mesh
save_name = f"{tid}ns_3D_ne_per_m3.npy"

## m, diameter of a circular laser beam profile
#beam_diameter = int(float(sys.argv[5])) * m_per_mm

################################################################################


# load data from files
ne_m3_tid = np.load(f'{directory_to_load_npy_files}{save_name}')  # m^-3
num_x_pts, num_y_pts, num_z_pts = ne_m3_tid.shape  # our convention (y is probe direction, z is axial height)
#print(num_x_pts, num_y_pts, num_z_pts)  # (800, 800, 420)

# transition to ray-tracer x/y/z coordinate convention
ne_m3_tid = np.swapaxes(ne_m3_tid, 1, 2)  # swaps y and z axes
num_x_pts, num_y_pts, num_z_pts = ne_m3_tid.shape  # ray-tracer convention (z is probe direction, y is axial height)
#print(num_x_pts, num_y_pts, num_z_pts)  # (800, 420, 800)

# 
# volume cubes with dx = dy = dz = 5e-5 m
dx_m = 50 * m_per_um  # m
# pts * dx ---> (x = 5 mm, y = 88 mm, z = 36 mm)
extent_z_m = num_z_pts * dx_m / 2  # m, half-length in z
extent_y_m = num_y_pts * dx_m / 2  # m, half-length in y
extent_x_m = num_x_pts * dx_m / 2  # m, half-length in x
#print(extent_x_m * mm_per_m, extent_y_m * mm_per_m, extent_z_m * mm_per_m)  # (20, 10.5, 20) mm

# construct the spatial grid in x (array to array), y (axial direction / height), and z (along the reconnection layer, the "probing direction" in ray-tracer code land)
zs_edges_m = np.linspace(-extent_z_m, extent_z_m, num_z_pts + 1)  # m, array of z edge spatial locations
ys_edges_m = np.linspace(-extent_y_m, extent_y_m, num_y_pts + 1)  # m, array of y edge spatial locations
xs_edges_m = np.linspace(-extent_x_m, extent_x_m, num_x_pts + 1)  # m, array of x edge spatial locations
#print(xs_edges_m[0:5] * mm_per_m, xs_edges_m[395:405] * mm_per_m)  # ([-20.   -19.95 -19.9  -19.85 -19.8 ] [-0.25 -0.2  -0.15 -0.1  -0.05  0.    0.05  0.1   0.15  0.2 ])

# get center spatial locations from the grid edges
zs_m = (zs_edges_m[1:] + zs_edges_m[:-1]) / 2  # m, array of z center spatial locations
ys_m = (ys_edges_m[1:] + ys_edges_m[:-1]) / 2  # m, array of y center spatial locations
xs_m = (xs_edges_m[1:] + xs_edges_m[:-1]) / 2  # m, array of x center spatial locations

# in case needed
#M_V_x = len(xs_m)
#M_V_y = len(ys_m)
#M_V_z = len(zs_m)


################################################################################


# makes the e- cube
gorgon_struct = pt.ElectronCube(xs_m, ys_m, zs_m)
gorgon_struct.external_ne_LSH(ne = ne_m3_tid)
gorgon_struct.calc_dndr(lwl = λ)
gorgon_struct.clear_memory()  # Clears variables not needed by solve method, saving memory


################################################################################


def full_system_solve(Np, beam_size, divergence, ne_cube, beam_x_offset, beam_z_offset):
    '''
    Main function called by all processors, considers Np rays traversing the electron density volume ne_cube

    beam_size and divergence set the initial properties of the laser beam

    '''

    ## Initialise rays and solve
    ## Initialise rays ("beam_size" = beam radius = beam diameter / 2)
    #ne_cube.init_beam(Np = Np, beam_size = beam_diameter / 2, divergence = divergence)
    ne_cube.init_beam_rect(Np = Np, beam_size = beam_size, divergence = divergence)
    
    # special mod to move rays to the right (+x)
    ne_cube.s0[0,:] += beam_x_offset

    # special mod to move rays up (+z)
    ne_cube.s0[3,:] += beam_z_offset
    
    # Solve the system
    ne_cube.solve()
    sxyz0 = ne_cube.s0[0:3, :]
    rf = ne_cube.rf  # output of solve saved into .rf is the rays in (x, theta, y, phi) format
    dt = ne_cube.solve_time  # s, time it took for this processor to complete ray-tracing for this batch of rays

    # Save memory by deleting initial ray positions
    #del ss
    ne_cube.s0 = None
    ne_cube.clear_memory()  # Can also use after calling solve to clear ray positions - important when running large number of rays
    
    # return origin uniformly distributed rays (3, Np), rays at the exit plane (4, Np), and the time taken to solve this bundle of rays
    return sxyz0, rf, dt


################################################################################


## Initialise the MPI
comm = MPI.COMM_WORLD
## Each processor is allocated a number (rank)
rank = comm.Get_rank()
## Number of processors being used
num_processors = comm.Get_size()  # comm.size
## names of the processor
name_processors = MPI.Get_processor_name()
## if root processor then print number of processors and rays per processor
if(rank == 0):
    print("Number of processors: %s"%num_processors)
    print("Rays per processors: %s"%Np)


################################################################################


def header_msg(rank, num_processors, name_processors):
    # msg = "Hello World! I am process {0} of {1} on {2}.\n"
    # sys.stdout.write(msg.format(rank, size, name))
    msg = "cpu-task # {0} of {1} on {2}"
    print(msg.format(rank + 1, num_processors, name_processors))

def split_remainder_msg(number_of_splits, remaining_rays):
    # split and remainder:
    print("Splitting to %d ray bundles"%number_of_splits)
    print("Solve for remainder: %d"%remaining_rays)

def bundle_msg(i, number_of_splits):
    # ray bundles
    print("bundle # %d of %d"%(i+1,number_of_splits))

def first_ray_msg(rf):
    try:
        print(f"first ray: {rf[:, 0]}")
    except:
        print(f"first ray: N/A (0 rays)")

def rf_output_msg(rf):
    print(f"rf.shape = {rf.shape}")

def rf_merged_output_msg(i, rf):
    if i > 0:
        print(f"concat'ed rf.shape = {rf.shape}")

def ray_trace_dt_msg(dt):
    print("Ray trace completed in:\t",dt,"s")

def newline_msg():
    print('')


################################################################################


# May trip memory limit, so split up calculation
if(Np > Np_ray_split):
    number_of_splits = Np//Np_ray_split
    remaining_rays   = Np%Np_ray_split
    # Remaining_rays could be zero, this doesn't matter
    #sxyz0, rf, dt = full_system_solve(Np = remaining_rays, beam_size = beam_size, divergence = divergence, ne_cube = gorgon_struct, z_pt2target_m = z_pt2target_m, zs_clipped_m = zs_clipped_m)
    sxyz0, rf, dt = full_system_solve(Np = remaining_rays, beam_size = beam_size, divergence = divergence, ne_cube = gorgon_struct, beam_x_offset = beam_x_offset, beam_z_offset = beam_z_offset)

    # LSH
    header_msg(rank, num_processors, name_processors)
    split_remainder_msg(number_of_splits, remaining_rays)
    ray_trace_dt_msg(dt)
    rf_output_msg(rf)
    first_ray_msg(rf)
    newline_msg()
    #sys.stdout.flush()
    
    # Iterate over remaining ray bundles
    for i in range(number_of_splits):
        # Solve subsystem
        #sxyz0_split, rf_split, dt_split = full_system_solve(Np = Np_ray_split, beam_size = beam_size, divergence = divergence, ne_cube = gorgon_struct, z_pt2target_m = z_pt2target_m, zs_clipped_m = zs_clipped_m)
        sxyz0_split, rf_split, dt_split = full_system_solve(Np = Np_ray_split, beam_size = beam_size, divergence = divergence, ne_cube = gorgon_struct, beam_x_offset = beam_x_offset, beam_z_offset = beam_z_offset)
        # Force garbage collection - this may be unnecessary but better safe than sorry
        gc.collect()
        # Add in results from splitting
        sxyz0 = np.concatenate((sxyz0, sxyz0_split), axis=1)
        rf = np.concatenate((rf, rf_split), axis=1)
        
        # LSH
        header_msg(rank, num_processors, name_processors)
        bundle_msg(i, number_of_splits)
        ray_trace_dt_msg(dt_split)
        rf_output_msg(rf_split)
        first_ray_msg(rf_split)
        rf_merged_output_msg(i, rf)
        newline_msg()
        #sys.stdout.flush()
        
else:
    print("Solving whole system...")
    #sxyz0, rf, dt = full_system_solve(Np = Np, beam_size = beam_size, divergence = divergence, ne_cube = gorgon_struct, z_pt2target_m = z_pt2target_m, zs_clipped_m = zs_clipped_m)
    sxyz0, rf, dt = full_system_solve(Np = Np, beam_size = beam_size, divergence = divergence, ne_cube = gorgon_struct, beam_x_offset = beam_x_offset, beam_z_offset = beam_z_offset)

    # LSH
    header_msg(rank, num_processors, name_processors)
    ray_trace_dt_msg(dt)
    rf_output_msg(rf)
    first_ray_msg(rf)
    newline_msg()
    #sys.stdout.flush()

## Now each processor has calculated results
## Must merge arrays and give to root processor

# LSH
def mymerge(rf_cpuA, rf_cpuB):
    #return [a+b for a, b in zip(x, y)]
    return np.concatenate((rf_cpuA, rf_cpuB), axis=1)
# Collect and merge all results and store on only root processor
#rf = comm.reduce(rf,root=0,op=MPI.SUM)
sxyz0 = comm.reduce(sxyz0,root=0,op=mymerge)
rf = comm.reduce(rf,root=0,op=mymerge)
#rf = comm.gather(sendobj=rf, root=0)  # to merge/append all arrays
if(rank == 0):
    print(f"final merged rf.shape = {rf.shape}")

'''
# LSH
rf_finalmerge = None
if(rank == 0):
    rf_finalmerge = np.empty((4, num_processors * Np))
comm.Gather(sendbuf = rf, recvbuf = rf_finalmerge, root=0)
if(rank == 0):
    print(f"final merged rf.shape = {rf_finalmerge.shape}")
'''

# Perform file saves on root processor only
if(rank == 0):

    ## save rays
    #gorgon_struct200ns.save_output_rays(f'npys/rays/tid{tid}ns_Np{Np}')
    np.save(f'{output_dir}3D_sxyz0_tid{tid}ns_Np{num_cpus * Np}.npy', sxyz0)  # 3 x Np
    np.save(f'{output_dir}3D_rf_tid{tid}ns_Np{num_cpus * Np}.npy', rf)  # 4 x Np
    ## load rays
    #s0200ns = np.load(f'npys/rays/3D_s0_tid{tid}ns_edge{z_chopoff_m * mm_per_m:.0f}mm_Np{Np}.npy')
    #rf200ns = np.load(f'npys/rays/3D_rf_tid{tid}ns_edge{z_chopoff_m * mm_per_m:.0f}mm_Np{Np}.npy')


################################################################################