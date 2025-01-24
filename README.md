# OblShock-Synth-Shadowgraphy
 Synthetic Shadowgraphy for COBRA Oblique Shock Experiments

# 3 Easy Steps (if you don't count the 2 prerequisite steps)

## -1/3: download all files in this GitHub repo, and transfer/upload them to your home directory on Engaging.  

## 0/3: building the necessary Anaconda environment (with mpi4py) for your user account in the MIT PSFC Engaging cluster

SSH into an Engaging terminal.  (I use MobaXterm on Windows.)  Then, line-by-line, type and enter (execute) the following commands in sequence:
  
> conda deactivate
  
> module purge
  
> module load anaconda3/2023.07
  
> conda clean --all
  
> conda env create -f conda_v1.yml

Note: the "--offline" flag might be needed to the last command above; let me know if there are issues and this is one thing we can try.

Next, to test whether mpi4py is probably going to work or not:
  
> conda deactivate
  
> module purge
  
> module load anaconda3/2023.07
  
> source activate conda_v1
  
> module load gcc/6.2.0
  
> module load openmpi/3.0.4
  
> python -c "import mpi4py"

If there are no errors, then I think it is good to go.*  If there are errors, we might need to more directly do what I did, referencing this guide: https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py

## 1/3: 01_load_and_save_e-_density.ipynb

Open a Jupyter Notebook session on Engaging-on-Demand (website).  Open this Jupyter Notebook file.  

In theory, the only user inputs that you should change are the three in the very top code cell.  And of those three, really only "save_dir" should probably be changed to your desired working directory.  

Go to Kernel / Restart Kernal and Run All Cells, and give it a few minutes.  You should see some plots of the 3D e- density mesh from GORGON, and a .npy file (of the 3D e- density mesh) should be generated.  

## 2/3: 02_ray_tracing_rect_beam.slurm (and 02_ray_tracing_rect_beam.py)

SSH into an Engaging terminal and navigate to your working directory (where all the files from this GitHub repo are stored, and where the e- density mesh .npy file is stored from Step #1/3.  Open 02_ray_tracing_rect_beam.slurm in a text editor.  In the first 27 lines of the file, you might wish to make adjustments to the following variables (under "#!/bin/bash" and "# input parameters"):

> "#!/bin/bash"
> #SBATCH --nodes=2                  # node count -N
> #SBATCH --ntasks=16                # total number of tasks (cores) -n
> #SBATCH --ntasks-per-node=8        # this converts the PBS -l mpiprocs=num (from example_MPI.py comment header)
> #SBATCH --time=01:30:00            # total run time limit (HH:MM:SS)
> #SBATCH --mail-user=your_at_mit_dot_edu_email_here

Notes / tips / rules-of-thumb: 
1. If you ask for NP=1e6 rays/cpu, this takes roughly ~1 hour of wall-time to complete.  Ask for --time=01:30:00 or --time=02:00:00 to be safe.  Or, scale up or down accordingly to your desired number of rays, it should be roughly linear.  
2. I could only safely run with up to 8 cpus/node (--ntasks-per-node=8).  Beyond that, you risk out-of-memory errors/failures, as the node's memory is divided among too many cpus and each cpu does not have enough RAM to work with.
3. --ntasks should be set to the product of --nodes times --ntasks-per-node (which should remain set to 8).
4. Change --mail-user=your_at_mit_dot_edu_email_here to your MIT email.  

> # input parameters
> BASE_DIRECTORY=/net/eofe-data005/psfclab001/lansing/docs/ray-tracing/COBRA/OblShock-3/  # base directory where all files are stored/run
> NP=1e6  # number of photons to run \*per processor\* (per "task", see header above)
> NP_RAY_SPLIT=5e5  # number of rays at which the computation is split to save memory, default 5e5
> X_BEAM_WIDTH=10  # mm, width of rectangular laser beam profile
> Z_BEAM_HEIGHT=18  # mm, height of rectangular laser beam profile
> BEAM_X_OFFSET=15  # mm, how much the circular beam is shifted in the +x direction (to fall on top of the wedge target)
> BEAM_Z_OFFSET=0  # mm, how much the circular beam is shifted in the +z direction (to fall on top of the wedge target)
> LAMBDA=532  # nm, laser photon wavelength
> DIVERGENCE=0  # 0 if no divergence (collimated/parallel rays), non-zero if beam divergence (radians)

Notes / tips / rules-of-thumb: 
1. BASE_DIRECTORY should be your personal working directory where all the files are stored, the same as "save_dir" in 01_load_and_save_e-_density.ipynb's first code cell.
2. This version of the code is for a rectangular laser beam because I thought it was more efficient to probe the crocodile region of the OblShock sims.  I also have a circular laser beam version, let me know if you'd prefer that.
3. X_BEAM_WIDTH, Z_BEAM_HEIGHT, BEAM_X_OFFSET, and BEAM_Z_OFFSET are currently set to something that I think is reasonable to target the crocodile region of the OblShock sims.

This should be more-or-less all that you need to edit in 02_ray_tracing_rect_beam.slurm.  Save any changes you made in your text editor, and close this file.  In theory, you should not need to edit anything in 02_ray_tracing_rect_beam.py.  

Finally, we can submit a batch job (using Engaging's slurm job scheduler) to run this ray-tracing code in parallel.  In your SSH terminal, so long as your working directory is where all your files are located, run this command:
> sbatch 02_ray_tracing_rect_beam.slurm

You will see a message such as "Submitted batch job 62552249" (where the last number is your job-id).  To check the status of your submitted job (i.e., to see if it is in the queue or running):
> squeue -u your_engaging_username_here

In my case, I run "squeue -u lansing", and I see a list of the jobs I have submitted to Engaging.  Let this run, and in the end you should generate a "runs/" directory, and within this another directory of your current run, and the two ray-trace-related .npy files (3D_sxyz0 and 3D_rf).  

## 3/3: 03_ray_transfer_matrix_analysis_rect_beam.ipynb

Open a Jupyter Notebook session on Engaging-on-Demand (website).  Open this Jupyter Notebook file.  

In theory, the only user inputs that you should change are the three in the very top code cell.
