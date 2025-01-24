# OblShock-Synth-Shadowgraphy
 Synthetic Shadowgraphy for COBRA Oblique Shock Experiments

# Steps

## -1/3: download all files in this GitHub repo, and transfer/upload them to your home directory on Engaging.  

## 0/3: building the necessary Anaconda environment (with mpi4py) for your user account in the MIT PSFC Engaging cluster

SSH into an Engaging terminal.  (I use MobaXterm on Windows.)  Then, line-by-line, type and enter (execute) the following commands in sequence:
  
  conda deactivate
  
  module purge
  
  module load anaconda3/2023.07
  
  conda clean --all
  
  conda env create -f conda_v1.yml

Note: the "--offline" flag might be needed to the last command above; let me know if there are issues and this is one thing we can try.

Next, to test whether mpi4py is probably going to work or not:
  
  conda deactivate
  
  module purge
  
  module load anaconda3/2023.07
  
  source activate conda_v1
  
  module load gcc/6.2.0
  
  module load openmpi/3.0.4
  
  python -c "import mpi4py"

If there are no errors, then I think it is good to go.*  If there are errors, we might need to more directly do what I did, referencing this guide: https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py

## 1/3: 01_load_and_save_e-_density.ipynb

Open a Jupyter Notebook session on Engaging-on-Demand (website).  Open this Jupyter Notebook file.  

In theory, the only user inputs that you should change are the three in the very top code cell.  And of those three, really only "save_dir" should probably be changed to your desired working directory.  

Go to Kernel / Restart Kernal and Run All Cells, and give it a few minutes.  You should see some plots of the 3D e- density mesh from GORGON, and a .npy file (of the 3D e- density mesh) should be generated.  

## 2/3: 
