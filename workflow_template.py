import argparse
from process_mrxs_christine_data import ProcessMRXSData

def main():
    parser = argparse.ArgumentParser(description="Run MRXS data processing workflow")
    parser.add_argument("directory_path", help="Path to the directory containing .mrxs files")
    parser.add_argument("inventory_file", help="Path to the inventory file")
    parser.add_argument("output_path", help="Path to the directory containing output files")
    parser.add_argument("output_extension", help="Extension of the output CSV file antibody-specific")

    args = parser.parse_args()

    directory_path = args.directory_path
    inventory_file = args.inventory_file
    #output_filename = args.output_filename
    output_ex = args.output_extension
    output_path = args.output_path
    final_data_filename = "final_data" + f".{output_ex}"
    print ("NCHDDH", final_data_filename)

    final_data = ProcessMRXSData.process_directory(directory_path, inventory_file, output_path)
    final_files = ProcessMRXSData.process_rate(output_path, final_data_filename)
    
    #print (final_rate)
    #print ("Final:", final_files)
    for file in final_files:
        ProcessMRXSData.process_heatmaps(file)
        ProcessMRXSData.process_scatterplots(file)

    # Print the final DataFrame
    #print(f"Final DataFrame saved to {output_filename}")
    print(f"Final Immunopositivity DataFrame saved to {final_data_filename}")
    print(f"Processing complete.")

if __name__ == "__main__":
    main()
