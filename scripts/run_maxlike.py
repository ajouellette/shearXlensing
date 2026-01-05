import argparse
import os
from os import path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chain")
    parser.add_argument("--objective", default="post", choices=["post", "like"], help="maximize posterior or likelihood (default: %(default)s)")
    args = parser.parse_args()

    chain_input = args.chain
    
    print(f"Will maximize {'posterior' if args.objective == 'post' else 'likelihood'} starting from the best point in {chain_input}")
    print()

    # get cosmosis files from chain
    prefix = path.join("/scratch/aaronjo2/cosmosis-scratch", path.basename(chain_input).split('.')[0])
    command = f"cosmosis-extract {chain_input} {prefix}"
    subprocess.run(command, shell=True)
    params_file = prefix + "_params.ini"
    values_file = prefix + "_values.ini"
    priors_file = prefix + "_priors.ini"
    
    cosmosis = f"cosmosis {params_file} --params pipeline.values={values_file} pipeline.priors={priors_file}"

    command_maxlike = f"{cosmosis} runtime.sampler=maxlike maxlike.max_posterior={'T' if args.objective == 'post' else 'F'}"
    command_maxlike += f" maxlike.start_method=chain maxlike.start_input={chain_input}"

    print(command_maxlike)
    print()

    subprocess.run(command_maxlike, shell=True)


if __name__ == "__main__":
    main()
