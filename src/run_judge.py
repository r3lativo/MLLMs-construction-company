from judge_utils import *


if __name__ == "__main__":
    
    # Load results into a DataFrame
    results_data = load_all_results(results_path)

    # Drop unnecessary columns since it's all the same
    to_drop = ["Model",	"Quantization",	"Device",	"Number of models",	"Max new tokens",	"Repetition Penalty",	"Max rounds"]
    results_data = results_data.drop(columns=to_drop).sort_values('run_time').reset_index(drop=True)

    # Run Judge for each analysis type
    commands = ["BASE", "CLAR_Q", "COMM_SH_REF", "IMPL_REF"]
    for cmd in commands:
        run_judge(cmd, results_data)
    
    # Save DataFrame to pickle file
    pickle_path = os.path.join(main_path, "analysis", 'results_data.pkl')
    results_data.to_pickle(pickle_path)
