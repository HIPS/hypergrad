import pickle
import sys
from odyssey import arg_fname, results_fname, complete_signal

if __name__ == "__main__":
    module_name, fun_name, idx_str = sys.argv[1:]
    idx = int(idx_str) - 1 # Go from 1-based slurm indexing to 0-based python
    __import__(module_name)
    function_to_run = getattr(sys.modules[module_name], fun_name)

    # Get function argument
    with open(arg_fname) as f:
        arg = pickle.load(f)[idx]

    # This is where the action is
    ans = function_to_run(arg)

    # Save results
    with open(results_fname(idx), 'w') as f:
        pickle.dump(ans, f)

    # Signal that the job is done
    with open(complete_signal(idx), 'w'): pass
