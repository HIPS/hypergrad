import time
from glob import glob
import subprocess
import os
from odyssey import run_signal_stem, slurm_fname, temp_dir, jobdir

if __name__ == "__main__":
    print "Monitoring slurm jobs in {0}".format(os.getcwd())
    while True:
        for fname in glob(run_signal_stem + "*"):
            jobname = fname[len(run_signal_stem):]
            print "Launching job {0}".format(jobname)
            with temp_dir(jobdir(jobname)):
                subprocess.call(["sbatch", slurm_fname])
            os.remove(fname)

        time.sleep(2)
