from judge_utils import *
import argparse


def get_args():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True, choices=["BASE", "CLAR_Q", "COMM_SH_REF", "IMPL_REF"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    # Get arguments from the parser
    args = get_args()

    # Load results into a DataFrame
    results_data = load_all_results(results_path)

    # Drop unnecessary columns since it's all the same
    to_drop = ["Model",	"Quantization",	"Device",	"Number of models",	"Max new tokens",	"Repetition Penalty",	"Max rounds"]
    results_data = results_data.drop(columns=to_drop).sort_values('run_time').reset_index(drop=True)

    run_judge(args.command, results_data)
