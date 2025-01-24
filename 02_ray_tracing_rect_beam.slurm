#!/bin/bash
#SBATCH --job-name=OblShock        # create a name for your job
#SBATCH --nodes=2                  # node count -N
#SBATCH --ntasks=16                # total number of tasks (cores) -n
#SBATCH --ntasks-per-node=8        # this converts the PBS -l mpiprocs=num (from example_MPI.py comment header)
#SBATCH --cpus-per-task=1          # cpu-cores per task
#SBATCH --mem=124G                 # memory per node  # SBATCH --mem-per-cpu=10000M       # memory per cpu-core
#SBATCH --time=01:30:00            # total run time limit (HH:MM:SS)
#SBATCH --partition=sched_mit_psfc # name of partition -p
#SBATCH --output=jobid_%j.out      # redirect std output to jobid_JOBID.out
#SBATCH --error=jobid_%j.err       # redirect std errors to jobid_JOBID.err
#SBATCH --mail-type=begin          # send email when job begins
#SBATCH --mail-type=end            # send email when job ends
#SBATCH --mail-user=lansing@mit.edu

# input parameters
FILENAME=02_ray_tracing_rect_beam  # 02_ray_tracing_rect_beam.py
BASE_DIRECTORY=/net/eofe-data005/psfclab001/lansing/docs/ray-tracing/COBRA/OblShock-3/  # base directory where all files are stored/run
TID=120  # ns, time ID
NP=1e6  # number of photons to run *per processor* (per "task", see header above)
NP_RAY_SPLIT=5e5  # number of rays at which the computation is split to save memory, default 5e5
X_BEAM_WIDTH=10  # mm, width of rectangular laser beam profile
Z_BEAM_HEIGHT=18  # mm, height of rectangular laser beam profile
BEAM_X_OFFSET=15  # mm, how much the circular beam is shifted in the +x direction (to fall on top of the wedge target)
BEAM_Z_OFFSET=0  # mm, how much the circular beam is shifted in the +z direction (to fall on top of the wedge target)
LAMBDA=532  # nm, laser photon wavelength
DIVERGENCE=0  # 0 if no divergence (collimated/parallel rays), non-zero if beam divergence (radians)
#NUMCPU=16  # number of CPUs to run on
#DESC=120ns_NP6e6_v0  # description of the run

#load default system modules
# . /etc/profile.d/modules.sh

#Uncomment below to echo the running environment
env

# define, create, and change to a unique scratch directory
#SCRATCH_DIRECTORY=/home/lansing/docs/ray-tracing/parallel/${FILENAME}/${DESC}-${SLURM_JOB_ID} # /nobackup1/${USER}/parallel_ray_tracing/${FILENAME}_${SLURM_JOB_ID}
SCRATCH_DIRECTORY=${BASE_DIRECTORY}runs/${SLURM_JOB_NAME}-TID${TID}ns-NP${SLURM_NTASKS}x${NP}-${SLURM_JOB_ID}/  # ${FILENAME}-TID${TID}ns-NP${SLURM_NTASKS}x${NP}-${SLURM_JOB_ID}/ # /nobackup1/${USER}/parallel_ray_tracing/${FILENAME}_${SLURM_JOB_ID}
mkdir -p ${SCRATCH_DIRECTORY}  # only make directory if it doesn't exist
# cd ${SCRATCH_DIRECTORY}
# copy input files
cp ./${FILENAME}.py ${SCRATCH_DIRECTORY}${FILENAME}.py
cp ./${FILENAME}.slurm ${SCRATCH_DIRECTORY}${FILENAME}.slurm

# load necessary bash shell, modules, and conda envs
# source /home/${USER}/.bashrc
module purge
module load anaconda3/2023.07
module load gcc/6.2.0
module load openmpi/3.0.4
# conda init bash
source activate conda_v1

# execute the Python script
#mpirun --verbose --display-map --display-allocation --output-proctable --output-filename ${SCRATCH_DIRECTORY}/mpirun_${SLURM_JOB_ID}.out --merge-stderr-to-stdout \
#python -u ./${FILENAME}.py ${BASE_DIRECTORY} ${TID} ${NP} ${NP_RAY_SPLIT} ${BEAM_DIAMETER} ${BEAM_X_OFFSET} ${LAMBDA} ${DIVERGENCE} ${SCRATCH_DIRECTORY}
mpirun --verbose --display-map --output-filename ${SCRATCH_DIRECTORY}mpirun_${SLURM_JOB_ID}.out --merge-stderr-to-stdout \
python -u ./${FILENAME}.py ${BASE_DIRECTORY} ${TID} ${NP} ${NP_RAY_SPLIT} ${X_BEAM_WIDTH} ${Z_BEAM_HEIGHT} ${BEAM_X_OFFSET} ${BEAM_Z_OFFSET} ${LAMBDA} ${DIVERGENCE} ${SCRATCH_DIRECTORY} ${SLURM_NTASKS}
# srun, --tag-output --timestamp-output

# move output/error files
#mv ./task* ${SCRATCH_DIRECTORY}/
mv ./jobid_${SLURM_JOB_ID}.out ${SCRATCH_DIRECTORY}jobid_${SLURM_JOB_ID}.out
mv ./jobid_${SLURM_JOB_ID}.err ${SCRATCH_DIRECTORY}jobid_${SLURM_JOB_ID}.err