import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings

def print_summary(group_names, stats_vals, p_vals, q_vals, reject_status_bools, alpha, output_file_path=None):
    """
    Prints the summary table to the console and optionally writes it to a file.

    Args:
        group_names (list): List of group identifier strings.
        stats_vals (list): List of test statistics.
        p_vals (list): List of original p-values.
        q_vals (list): List of FDR corrected q-values.
        reject_status_bools (np.array): Boolean array indicating H0 rejection.
        alpha (float): Significance level used for FDR.
        output_file_path (str, optional): Path to the output text file. Defaults to None.
    """
    lines_to_output = []

    title_line = f"\n--- Summary of Test Results (FDR alpha = {alpha}) ---"
    lines_to_output.append(title_line)
    
    max_group_name_len = 10 # Default
    if group_names: 
         max_group_name_len = max(len(name) for name in group_names) if group_names else 10
         max_group_name_len = max(max_group_name_len, len("Group")) # Ensure header "Group" fits

    header_group_col = f"{'Group':<{max_group_name_len}}"
    header = f"{header_group_col} | {'Wilcoxon W':<11} | {'Original p':<12} | {'Corrected q':<13} | {'Reject H0':<12}"
    lines_to_output.append(header)
    separator = "-" * len(header)
    lines_to_output.append(separator)

    for i in range(len(p_vals)): # p_vals determines the number of rows
        group_name_str = group_names[i] if i < len(group_names) else "N/A"
        stat_str = f"{stats_vals[i]:.2f}" if i < len(stats_vals) and not np.isnan(stats_vals[i]) else "N/A"
        p_val_str = f"{p_vals[i]:.4f}" if not np.isnan(p_vals[i]) else "N/A"
        q_val_str = f"{q_vals[i]:.4f}" if i < len(q_vals) and not np.isnan(q_vals[i]) else "N/A"
        
        reject_str = "N/A"
        # reject_status_bools corresponds to valid_p_values, so q_val_str check is key
        if q_val_str != "N/A": # If q-value is available
             # reject_status_bools might be shorter if there were NaNs in p_values
             # This relies on final_reject_status passed to print_summary being correctly sized and populated
             reject_str = str(reject_status_bools[i]) # Assumes reject_status_bools is full length with False for NaN p/q
        
        line = f"{group_name_str:<{max_group_name_len}} | {stat_str:<11} | {p_val_str:<12} | {q_val_str:<13} | {reject_str:<12}"
        lines_to_output.append(line)
    lines_to_output.append(separator)

    # 1. Print to console
    for line in lines_to_output:
        print(line)

    # 2. Write to file if path is provided
    if output_file_path:
        try:
            with open(output_file_path, 'w') as f:
                for line in lines_to_output:
                    f.write(line + '\n')
            print(f"\nSummary also written to: {output_file_path}")
        except IOError as e:
            print(f"\nError writing summary to file {output_file_path}: {e}")


def run_multiple_wilcoxon_tests_with_fdr(datasets_dict, 
                                         hypothesized_median=0, 
                                         fdr_alpha=0.05, 
                                         zero_method='wilcox', 
                                         correction=True,
                                         output_file_path=None): # Added output_file_path
    """
    Performs Wilcoxon Signed-Rank tests on multiple datasets (provided as a dictionary)
    to test if the median is different from a hypothesized value (default 0),
    applies FDR correction, and prints/saves the summary.
    """

    print("--- Performing Wilcoxon Signed-Rank Tests (Non-Parametric) ---")
    print(f"(H0: population median = {hypothesized_median})")
    print(f"Using zero_method='{zero_method}', correction={correction}\n")

    if not datasets_dict:
        print("No datasets provided in the dictionary.")
        if output_file_path: # Try to write an empty report if path provided
            print_summary([], [], [], [], np.array([]), fdr_alpha, output_file_path)
        return

    group_names_ordered = list(datasets_dict.keys())
    p_values = []
    test_statistics = []

    for group_name in group_names_ordered:
        data = datasets_dict[group_name]
        
        if not isinstance(data, np.ndarray):
            current_data_arr = np.array(data, dtype=float)
        else:
            current_data_arr = data.astype(float)
        
        current_data_no_nan = current_data_arr[~np.isnan(current_data_arr)]

        if len(current_data_no_nan) == 0:
            print(f"Group: {group_name}: Skipped (empty after NaN removal or was initially empty).")
            p_values.append(np.nan)
            test_statistics.append(np.nan)
            continue

        data_to_test = current_data_no_nan - hypothesized_median
        
        if np.all(data_to_test == 0):
            print(f"Group: {group_name}: All values are equal to the hypothesized median ({hypothesized_median}). P-value = 1.0.")
            p_values.append(1.0)
            test_statistics.append(0.0)
            continue
        
        non_zero_diffs = data_to_test[data_to_test != 0]
        if len(non_zero_diffs) > 0 and len(np.unique(non_zero_diffs)) == 1 and zero_method == 'wilcox':
            print(f"Group: {group_name}: All non-zero differences from hypothesized median are identical. Wilcoxon P-value = 1.0.")
            p_values.append(1.0)
            test_statistics.append(0.0)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                stat, p_val = stats.wilcoxon(data_to_test, 
                                             zero_method=zero_method, 
                                             correction=correction, 
                                             alternative='two-sided')
            
            if np.isnan(p_val):
                print(f"Group: {group_name}: Wilcoxon test resulted in NaN p-value. P-value set to 1.0.")
                p_values.append(1.0)
                test_statistics.append(0.0 if np.isnan(stat) else stat)
            else:
                p_values.append(p_val)
                test_statistics.append(stat)
            
            print(f"Group: {group_name}: Wilcoxon W = {test_statistics[-1]:.4f}, Original p-value = {p_values[-1]:.4f}")

        except Exception as e:
            print(f"Group: {group_name}: Error during Wilcoxon test: {e}. P-value set to NaN.")
            p_values.append(np.nan)
            test_statistics.append(np.nan)

    print("\n--- Applying FDR Correction (Benjamini-Hochberg) ---")
    
    valid_p_indices = [i for i, p in enumerate(p_values) if not np.isnan(p)]
    valid_p_values_list = [p_values[i] for i in valid_p_indices]

    # Initialize final arrays for q-values and rejection status (full length)
    final_q_values = np.full(len(p_values), np.nan)
    final_reject_status_bools = np.full(len(p_values), False, dtype=bool)


    if not valid_p_values_list:
        print("No valid p-values to correct.")
        # Call print_summary even if no valid p-values, to output NaNs or empty table to file if specified
        print_summary(group_names_ordered, test_statistics, p_values, 
                      final_q_values, # all NaNs 
                      final_reject_status_bools, # all False
                      fdr_alpha,
                      output_file_path=output_file_path)
        return

    reject_bools_for_valid, q_values_corrected, _, _ = multipletests(valid_p_values_list, alpha=fdr_alpha, method='fdr_bh')

    for i, original_idx_in_p_values in enumerate(valid_p_indices):
        final_q_values[original_idx_in_p_values] = q_values_corrected[i]
        final_reject_status_bools[original_idx_in_p_values] = reject_bools_for_valid[i]
    
    # Call print_summary with the full-length arrays including NaNs/False for non-processed groups
    print_summary(group_names_ordered, test_statistics, p_values, 
                  final_q_values, final_reject_status_bools, 
                  fdr_alpha, 
                  output_file_path=output_file_path)

# --- Main script execution ---
if __name__ == "__main__":
    # Define the output file for the summary results
    results_output_filename = "stats/salt_wilcoxon_fdr_summary_results.txt" # You can change this

    # --- 1. Data Preparation ---
    try:
        # IMPORTANT: Replace with the correct path to your CSV file
        # df = pd.read_csv('data/Fructose_agarose_stats.csv') 
        df = pd.read_csv('data/Salt_agarose_stats.csv') 
        print("Successfully loaded data from CSV.")
    except FileNotFoundError:
        print(f"\nError: CSV file not found. Please update the path in the script.")
        print("Using a randomly generated dummy DataFrame for demonstration purposes.\n")
        num_rows = 200
        ids_unique_dummy = [f"{s}_{a:.1f}" for s in ['female', 'male'] for a in np.random.choice([5.0, 10.0, 15.0, 20.0], size=2, replace=False)]
        dummy_ids = np.random.choice(ids_unique_dummy, num_rows)
        dummy_sexes = [id_str.split('_')[0] for id_str in dummy_ids]
        dummy_amplitudes = [float(id_str.split('_')[1]) for id_str in dummy_ids]
        
        dummy_data_for_df = {
            'sex': dummy_sexes,
            'stimulus_01_name': np.random.choice(['Fructose', 'Sucrose', 'Agarose'], num_rows),
            'stimulus_01_amplitude': dummy_amplitudes,
            'preference_index': np.concatenate([
                stats.skewnorm.rvs(a=5, loc=-0.2, scale=0.5, size=num_rows//2),
                stats.skewnorm.rvs(a=-5, loc=0.2, scale=0.5, size=num_rows - num_rows//2)
            ]),
            'id': dummy_ids
        }
        df = pd.DataFrame(dummy_data_for_df)

    df['preference_index'] = pd.to_numeric(df['preference_index'], errors='coerce')
    df['pool_id'] = df['stimulus_01_name'].astype(str) + "_" + df['stimulus_01_amplitude'].astype(str)

    datasets_to_test_dict = {}
    print("\n--- Preparing 'id' groups ---")
    for group_identifier in df['id'].unique():
        if pd.isna(group_identifier): continue
        group_data = df[df['id'] == group_identifier]['preference_index'].dropna().values
        if group_data.size > 0:
            datasets_to_test_dict[f"id_{group_identifier}"] = group_data
        # else: print(f"No data for 'id_{group_identifier}' after filtering.")

    print("\n--- Preparing 'pool_id' groups (pooled sexes) ---")
    for pooled_identifier in df['pool_id'].unique():
        if pd.isna(pooled_identifier): continue
        pooled_data = df[df['pool_id'] == pooled_identifier]['preference_index'].dropna().values
        if pooled_data.size > 0:
            datasets_to_test_dict[f"pool_{pooled_identifier}"] = pooled_data
        # else: print(f"No data for 'pool_{pooled_identifier}' after filtering.")
    
    if not datasets_to_test_dict:
        print("\nNo datasets prepared for testing. Exiting.")
    else:
        print(f"\n--- Total groups prepared for testing: {len(datasets_to_test_dict)} ---")
        run_multiple_wilcoxon_tests_with_fdr(datasets_to_test_dict, 
                                             hypothesized_median=0, 
                                             fdr_alpha=0.05,
                                             output_file_path=results_output_filename) # Pass the filename