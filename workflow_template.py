import argparse
import os

from process_mrxs_data import ProcessMRXSData
from process_svs_data import ProcessSVSData
from process_ndpi_data import ProcessNDPIData
from process_bif_data import ProcessBIFData
from process_czi_data import ProcessCZIData


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
    #print ("NCHDDH", final_data_filename)
    
   # print (final_data_filename)
    unwanted_suffixes = {'.txt'}
    extension_processors = {
        'mrxs': ProcessMRXSData,
        'svs': ProcessSVSData,
        'ndpi': ProcessNDPIData,
        'bif': ProcessBIFData,
        'czi': ProcessCZIData
    }

    extension_files = {}

    for filename in os.listdir(directory_path):
        base_name, extension = os.path.splitext(filename)
        #_, extension = os.path.splitext(filename)
        print (extension)
    
        components = base_name.rsplit('.', 1)
        
        if len(components) > 1:
            primary_extension = components[-1]
        else:
            primary_extension = None

        if primary_extension:
            print(f"Filename: {filename}, Primary Extension: {primary_extension}")
            extension_files[directory_path] = primary_extension
        

    for item in extension_files: 
        print ('item:', item)
        print ('ext', extension_files[item])
        if extension_files[item]  == 'mrxs':
            final_data = ProcessMRXSData.process_directory(directory_path, inventory_file, output_path, output_ex)
            final_files = ProcessMRXSData.process_rate(output_path, final_data_filename)
            for file in final_files:
                ProcessMRXSData.process_heatmaps(file)
                ProcessMRXSData.process_scatterplots(file)
        elif extension_files[item] == 'svs':
            final_data = ProcessSVSData.process_directory(directory_path, inventory_file, output_path, output_ex)
            final_files = ProcessSVSData.process_rate(output_path, final_data_filename)
            for file in final_files:
                ProcessSVSData.process_heatmaps(file)
                ProcessSVSData.process_scatterplots(file)
        elif extension_files[item] == 'ndpi':                        
            final_data = ProcessNDPIData.process_directory(directory_path, inventory_file, output_path, output_ex)
            final_files = ProcessNDPIData.process_rate(output_path, final_data_filename)
            for file in final_files:            
                ProcessNDPIData.process_heatmaps(file)
                ProcessNDPIData.process_scatterplots(file)
        elif extension_files[item] == 'bif':
            final_data = ProcessBIFData.process_directory(directory_path, inventory_file, output_path, output_ex)
            final_files = ProcessBIFData.process_rate(output_path, final_data_filename)
            for file in final_files:
                ProcessBIFData.process_heatmaps(file)
                ProcessBIFData.process_scatterplots(file)
        elif extension_files[item] == 'czi':
            final_data = ProcessCZIData.process_directory(directory_path, inventory_file, output_path, output_ex)
            final_files = ProcessCZIData.process_rate(output_path, final_data_filename)
            for file in final_files:
                ProcessCZIData.process_heatmaps(file)
                ProcessCZIData.process_scatterplots(file)



#    final_data = ProcessMRXSData.process_directory(directory_path, inventory_file, output_path, output_ex)
#    final_files = ProcessMRXSData.process_rate(output_path, final_data_filename)
    
    #print (final_rate)
    #print ("Final:", final_files)
#    for file in final_files:
#        ProcessMRXSData.process_heatmaps(file)
#        ProcessMRXSData.process_scatterplots(file)

    # Print the final DataFrame
    #print(f"Final DataFrame saved to {output_filename}")
    #print(f"Final Immunopositivity DataFrame saved to {final_data_filename}")
    #print(f"Processing complete.")

if __name__ == "__main__":
    main()
