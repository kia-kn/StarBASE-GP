import argparse
import ray
# NEW: for HPC node as a potential solution:
# ray.init(include_dashboard=False)
import numpy as np
from Source.evolver import EA
import time
import os


def main(args):
    # set experiment configurations using arguments from SLURM array job
    ea_config = {
        'seed': np.uint16(args.seed),
        'pop_size': np.uint16(args.pop_size),
        'uni_cnt_max': np.uint16(args.uni_cnt_max),
        'uni_cnt_min': np.uint16(args.uni_cnt_min),
        'cores': args.cores,
        'mut_selector_p': np.float64(1.0),  # This remains constant
        'mut_regressor_p': np.float64(.5),  # This remains constant
        # 'mut_ran_p': np.float64(args.mut_ran_p),
        # 'mut_smt_p': np.float64(args.mut_smt_p),
        # 'smt_in_in_p': np.float64(args.smt_in_in_p),
        # 'smt_in_out_p': np.float64(args.smt_in_out_p),
        # 'smt_out_out_p': np.float64(args.smt_out_out_p),
        'mut_prob': np.float64(args.mut_prob),
        'cross_prob': np.float64(args.cross_prob),
        'save_directory': args.save_directory,
        'rand_init': False if args.rand_init == 0 else True
    }

    ea = EA(**ea_config)

    total_time = time.time()

    # Update to use the data_dir passed as argument
    # NEW: adding new y target label (not default =="y")
    start_time = time.time()
    ea.data_loader(args.data_dir, (args.seed) % 40) # added the seed parameter
    # ea.data_loader(args.data_dir, (args.seed) % 40, target_label="OFT_LatencyCensored") # added the seed parameter
    print(f"Data loaded in {(time.time() - start_time) / 60} mins")

    start_time = time.time()
    ea.initialize_hubs(500)
    print(f"Hubs done in {(time.time() - start_time) / 60} mins")

    start_time = time.time()
    ea.evolve(args.gens)
    print(f"Evolution done in {(time.time() - start_time) / 60 / 60} hours")

    start_time = time.time()
    ea.post_analysis_with_good_snps()
    print(f"Post analysis done in {(time.time() - start_time) / 60 / 60} hours")

    print(f"Time taken: {(time.time() - total_time) / 60 / 60} hours")

    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EA with SLURM parameters")

    # Match arguments from SLURM job
    parser.add_argument('--seed', type=int, required=True, help="Random seed")
    parser.add_argument('--pop_size', type=int, default=100, help="Population size")
    parser.add_argument('--uni_cnt_max', type=int, default=200, help="Max episode count")
    parser.add_argument('--uni_cnt_min', type=int, default=10, help="Min episode count")
    parser.add_argument('--cores', type=int, default=10, help="Number of cores")
    parser.add_argument('--mut_ran_p', type=float, default=0.45, help="Mutation random probability")
    parser.add_argument('--mut_smt_p', type=float, default=0.45, help="Mutation smooth probability")
    parser.add_argument('--smt_in_in_p', type=float, default=0.10, help="Smooth in-in probability")
    parser.add_argument('--smt_in_out_p', type=float, default=0.450, help="Smooth in-out probability")
    parser.add_argument('--smt_out_out_p', type=float, default=0.450, help="Smooth out-out probability")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the data file")
    parser.add_argument('--save_directory', type=str, required=True, help="Path to the results director")
    parser.add_argument('--bin_size', type=int, required=True, help="Bin size")
    parser.add_argument('--gens', type=int, required=True, help="Number of generations")
    parser.add_argument('--mut_prob', type=float, required=True, help="Mutation Probablity")
    parser.add_argument('--cross_prob', type=float, required=True, help="Crossover Probablity")
    parser.add_argument('--rand_init', type=int, required=True, help="Random initialization flag (0 or 1)")

    args = parser.parse_args()
    main(args)