import os

root_working_dir = os.path.expanduser("~/odyssey/tmp")
slurm_options = \
"""#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 16:00:00
#SBATCH -p hips
#SBATCH --mem-per-cpu=4000
"""

# root_working_dir = "/tmp"
# slurm_options= ""
