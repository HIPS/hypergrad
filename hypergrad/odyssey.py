""" Provides `omap`, which works like Python's `map`, but farms out to
Odyssey. Here's how to use it:

* Make a scratch space directory on your Odyssey account (e.g. ~/tmp/)
* Mount this (or the whole Odyssey home dir) on your local machine with sshfs
* Make an `odyssey_config.py` that points to it (look at `odyssey_config_example.py`)
* On Odyssey, run "python -m slurm_job_watcher" from within the tmp directory
* Run a script (e.g. test_odyssey.py) from your local machine...it should just work
"""

import inspect
import subprocess
import time
import sys
import os
import numpy.random as npr
import shutil
import pickle
from contextlib import contextmanager
from os.path import exists, getsize
from os import path
from odyssey_config import root_working_dir, slurm_options

arg_fname = "args.pkl"
slurm_fname = "batch_script.slurm"
results_fname = lambda i : "results_{0}.pkl".format(i+1)
stderr_fname = lambda i : "stderr_{0}.txt".format(i+1)
stdout_fname = lambda i : "stdout_{0}.txt".format(i+1)
run_signal_stem = "please_run_"
complete_signal = lambda i : "task_complete_{0}".format(i+1)
run_signal = lambda jobname : run_signal_stem + jobname
jobdir = lambda jobname : "jobdir_" + jobname
banner = lambda s : "{0} {1} {0}".format("="*10, s)

slurm_template = \
"""#!/bin/bash
#SBATCH -o stdout_%a.txt
#SBATCH -e stderr_%a.txt
#SBATCH -J {jobname}
#SBATCH -a 1-{N_tasks} # Array range
{other_options}
python -um hypergrad.slurm_run {module_name} {fun_name} $SLURM_ARRAY_TASK_ID
"""

def omap(fun, arglist):
    print banner("Starting omap")
    N_tasks = len(arglist)
    jobname = str(npr.RandomState().randint(10**12))
    working_dir = path.join(root_working_dir, jobdir(jobname))
    module_path = path.join(os.getcwd(), inspect.getsourcefile(fun))
    module_name = inspect.getmodulename(module_path)
    run_signal_path = path.join('..', run_signal(jobname))
    fun_name = fun.__name__
    slurm_str = slurm_template.format(jobname=jobname,
                                      N_tasks=N_tasks,
                                      other_options=slurm_options,
                                      module_name=module_name,
                                      fun_name=fun_name)
    print "Submitting {0} tasks (output in {1})".format(N_tasks, working_dir)
    with temp_dir(working_dir):
        shutil.copy(module_path, ".")
        with open(arg_fname, 'w') as f: pickle.dump(arglist, f)
        with open(slurm_fname, 'w') as f: f.write(slurm_str)
        with open(run_signal_path, 'w'): pass
        monitor_tasks(N_tasks, run_signal_path)
        return map(get_results, range(N_tasks))

def monitor_tasks(N_tasks, run_signal_path):
    while path.exists(run_signal_path):
        time.sleep(1)
    print "Tasks submitted"
    while True:
        status_str = "".join(map(status, range(N_tasks)))
        sys.stdout.write("\rStatus: [{0}]".format(status_str))
        sys.stdout.flush()
        if all([s in ".E" for s in status_str]): break
        time.sleep(1)
    print ""
    print banner("All tasks complete")

def status(i):
    if path.exists(complete_signal(i)):
        return "." # Complete
    elif path.exists(stderr_fname(i)):
        if path.getsize(stderr_fname(i)) == 0:
            return "r" # Running
        else:
            return "E" # Error
    else:
        return "*" # Pending

def get_results(i):
    with open(results_fname(i)) as f:
        return pickle.load(f)

@contextmanager
def temp_dir(new_dir):
    old_dir = os.getcwd()
    if not path.exists(new_dir): os.mkdir(new_dir)
    os.chdir(new_dir)
    yield
    os.chdir(old_dir)
