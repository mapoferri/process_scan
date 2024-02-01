import pandas as pd
import os
import sys
import numpy as np
from pandas import read_excel
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class ProcessMRXSData:
    def __init__(self, mrxs_file, inventory_file):

        """
        Initialize the ProcessMRXSData object with MRXS file and inventory file.

        :param mrxs_file: Path to the MRXS data file.
        :param inventory_file: Path to the inventory file (CSV or Excel).
        """

        self.mrxs_file = mrxs_file
        self.inventory_file = inventory_file


    def process_data(self):

        """
        Process MRXS data, merge with inventory, and calculate immunopositivity statistics.

        :return: Final DataFrame with immunopositivity statistics.
        """

        print(f"Processing data for {self.mrxs_file}...")

        # Open the text file and transform it into a DataFrame
        df_mrxs = pd.read_csv(self.mrxs_file, sep='\t', engine='python')
        df_mrxs['Image'] = df_mrxs['Image'].str.rstrip('.mrxs.txt')
        #print("DataFrame from text file:")
        #print(df_mrxs)
        
        if 'Class' not in df_mrxs.columns:
            print(f"Skipping MRXS file: {self.mrxs_file} - 'Class' column not found.")
            return pd.DataFrame()

        final_df = pd.DataFrame() #Initial empty dataframe

        # Task 1A: Set the different collection for different regions
        unique_parents = df_mrxs['Parent'].unique()
        dfs_list = []  # To separate the dataframes

        for parent_value in unique_parents:
            # Filter dataframe based on 'Parent' value
            sub_df_mrxs = df_mrxs[df_mrxs['Parent'] == parent_value]
        
            # Start a counter for the Class column
            positive_counter = (sub_df_mrxs['Class'] == 'PositiveCell').sum()
            negative_counter = (sub_df_mrxs['Class'] == 'NegativeCell').sum()

            # Open the inventory file and select corresponding rows
            if self.inventory_file.endswith('.csv'):
                df_inventory = pd.read_csv(self.inventory_file)
            else:
                df_inventory = pd.read_excel(self.inventory_file, sheet_name=None)
                merged_df = pd.DataFrame() # Initial empty dataframe
                for sheet_name, sheet_df in df_inventory.items():
                    if not sheet_df.empty:
                        #print (sheet_df.columns)
                        if 'ID_Slidescanner' not in sheet_df.columns:
                            print("Error: 'ID_Slidescanner' column not found in the sheet.")
                            #sys.exit(1)
                            continue
                        sheet_df['ID_Slidescanner'] = sheet_df['ID_Slidescanner'].astype(str)
                        result = sub_df_mrxs.merge(sheet_df, left_on='Image', right_on='ID_Slidescanner', how='inner')
                        if not result.empty:
                            #antibody = sheet_name
                            merged_df = pd.concat([merged_df, result], ignore_index=True)

        # Create the merged dataframe based on the correspondence between the Image value and the ID_Slidescanner value in the Inventory
                if not merged_df.empty:
                    final_df = merged_df[['ID_Sample', 'Antibody', 'ID_Slidescanner', 'Image', 'Parent']]
                    final_df = final_df.head(1)
                    final_df['ID_Slidescanner'] = final_df['ID_Slidescanner'].values[0]
                    final_df['Image'] = final_df['Image'].values[0]
                    final_df['Parent'] = final_df['Parent'].values[0]
                    #print(f"FINAL for Parent {parent_value}")
                    final_df['Positive Class'] = positive_counter
                    final_df['Negative Class'] = negative_counter
                    
                        
                    if positive_counter == 0 or negative_counter == 0:
                        print(f"Error: Counter for 'PositiveCell' or 'NegativeCell' is zero for {self.mrxs_file}. Skipping Positivity Rate calculation.")
                    else:
                        final_df['Positivity Rate'] = (positive_counter * 100) / (positive_counter + negative_counter)

                    dfs_list.append(final_df)


                    #print(final_df.columns)
                else:
                    print("Merged DataFrame is empty.")
                    final_df = pd.DataFrame({'ID_Sample': [None], 'Antibody': [None], 'ID_Slidescanner': [None], 'Image': [None]})
                    return final_df


        if not dfs_list:
            print("No data available for any Parent value.")
        else:
            print(f"Processed data for {self.mrxs_file}.")

        return dfs_list

        
     

    @staticmethod
    def process_positivity(xls_file, final_df):

        """
        Process immunopositivity rate from Excel files and merge with the main DataFrame.

        :param xls_file: Path to the Excel file containing positivity rate data.
        :param final_df: Main DataFrame to merge the positivity rate data into.

        :return: Merged DataFrame.
        """

        print(f"Processing immunopositivity rate for {xls_file}...")
        df_xls = pd.read_excel(xls_file)
        hd_and_rate = df_xls[['ID_Sample', 'Positivity Rate']]
        
        # print(f"Checking columsn {hd_and_rate}")
        
        # Merge the extracted data based on the 'HD' column
        final_df = final_df.merge(hd_and_rate, on='ID_Sample', how='left')

        return final_df


    @staticmethod
    def process_directory(directory_path, inventory_file, output_path, output_ex):
        """
        Process MRXS data from a directory, save antibody-specific data, and return the final DataFrame.

        :param directory_path: Path to the directory containing .mrxs files.
        :param inventory_file: Path to the inventory file (CSV or Excel).
        :param output_path: Path to save antibody-specific data.
        
        :return: Final DataFrame.
        """

        #Array of the dataframes produced for each file
        result_dfs= []
        #final_result = pd.DataFrame(columns=['HD'])
        no_mrxs_files = True
        final_result = None
        #print(output_ex)

        # For each slide, call the process_data function
        for filename in os.listdir(directory_path):
            if filename.endswith('.mrxs'):
                mrxs_file = os.path.join(directory_path, filename)
                print(f"Processing file: {mrxs_file}")
                no_mrxs_files = False
                
                processor = ProcessMRXSData(mrxs_file, inventory_file)
                result_df = processor.process_data()
                #print (result_df)
                #result_dfs.append(result_df)
                #if isinstance(result_df, pd.DataFrame):
                result_dfs.extend(result_df)
                #else:
                #    print(f"Skipping file {mrxs_file} - Processed data is not a DataFrame.")

                #result_dfs.extend(processor.process_data())
            # Appending to an array, so should be fine

        #print("Contents of result_dfs:")
        #print(result_dfs)
        #for df in result_dfs:
        #    print(type(df))
        #    print(df)
        #    print(df[1]['Parent'].iloc[0], df[0])
        valid_dataframes = [df for df in result_dfs if isinstance(df, pd.DataFrame)]

        if not valid_dataframes:
            print("No valid pandas DataFrames found in result_dfs. Skipping concatenation.")
        else:    
            if no_mrxs_files:
                print("No files with .mrxs extension found in the specified directory.")
                sys.exit(1)

        # Concatenate all result DataFrames into a single DataFrame
            final_result = pd.concat(valid_dataframes)
            #print (final_result)
        # Use defaultdict to group data based on 'Parent'
            grouped = defaultdict(lambda: defaultdict(list))
            for result_df in valid_dataframes:
            #grouped[result_df['Parent'].iloc[0]].append(result_df)
                parent = result_df['Parent'].iloc[0]
                antibody = result_df['Antibody'].iloc[0]
                grouped[parent][antibody].append(result_df)


        #grouped = final_result.groupby(["Antibody", "Parent"])
            for group_parent, parent_data in grouped.items():
                for group_antibody, group_data_list in parent_data.items():
                    group_data = pd.concat(group_data_list, axis=0, ignore_index=True)
                    output_filename = f"{group_parent}_{group_antibody}_data"
                    if 'HD' in group_data.columns: 
                        group_data.rename(columns={'HD': 'ID_Sample'}, inplace=True)  # Rename 'HD' column to 'sample_ID'
                    output_filepath = os.path.join(output_path, output_filename + "."+ output_ex) 
                    print(f"This is the {output_ex} and this is the filepath {output_filepath}, but the filename is {output_filename}.")
                    if output_ex == 'csv':
                        group_data.to_csv(output_filepath, index=False)
                    elif output_ex == 'xlsx':
                        group_data.to_excel(output_filepath, index=False)
                    else:
                        print(f"Unsupported format for {output_ex}.")
                    print(f"Saved data for and Parent {group_parent} to {output_filename}")

        # Save the final DataFrame to a CSV file with the specified name
        #with open(output_filename, 'w', encoding='utf-8') as file:
            #final_result.to_csv(file, index=False)
            #final_result.to_excel(output_filename, index=False)


        return final_result

    @staticmethod
    def process_rate(output_path, final_data_filename):

        final_df = pd.DataFrame(columns=['ID_Sample', 'Parent'])
        no_xls_files = True
        
        # Check if the output path exists, and create it if not
        #if not os.path.exists(output_path):
        #    os.makedirs(output_path)


        for filename in os.listdir(output_path):
            print(f"Found file: {filename}")
            if filename.endswith(('.xlsx', 'csv')):
                xlsx_file = os.path.join(output_path, filename)
                print(f"Processing file: {xlsx_file}")
                no_xls_files = False

                #processor = ProcessMRXSData.process_positivity(xlsx_file, final_df)
                
                if filename.endswith('.csv'):
                    xlsx_data = pd.read_csv(xlsx_file)
                elif filename.endswith('.xlsx'):
                    xlsx_data = pd.read_excel(xlsx_file)
                    
        #        print (f"Columns:  {xlsx_data.columns}")
                if all(col in xlsx_data.columns for col in ['ID_Sample', 'Antibody', 'Positivity Rate', 'Parent']):
                    data = xlsx_data[['ID_Sample','Antibody','Parent', 'Positivity Rate']]
                    data = data.copy()
                    #data.rename(columns={'Positivity Rate': f"Positivity Rate ({xlsx_data['Antibody'].iloc[0]})"}, inplace=True)
     #               print(f"Data columns: {data.columns}")
                    

                    final_df = pd.concat([final_df, data])
                    #final_df = pd.merge(final_df, data, on=['sample_ID', 'Parent'], how='outer')

        #print (final_df) 
        
        # Merge duplicate samples based on 'sample_ID'
        final_files = ProcessMRXSData.merge_samples(final_df, final_data_filename)
        
        #print (final_files)
        #print (final_df_merged)

        if no_xls_files:
            print(f"No files has been found to study the correlation of the data, check them.")
            sys.exit(1)
        if final_data_filename.endswith('.csv'):
            final_df.to_csv(final_data_filename, index=False)
        elif final_data_filename.endswith('.xlsx'):
            final_df.to_excel(final_data_filename, index=False)
        #final_data_filename += '.csv'
        #final_df.to_csv(final_data_filename, index=False)

        return final_files
        #return a list of the files produced, so for each region, the graphs are going to be produced 
        

    def merge_duplicate_samples(data):
        merged_data = data.pivot_table(index=['ID_Sample', 'Parent', 'Antibody'], columns='Antibody', values='Positivity Rate', aggfunc='first').reset_index()
        #merged_data.columns = [f"Positivity Rate ({antibody})" if antibody != 'sample_ID' else 'sample_ID' for antibody in merged_data.columns]

        #merged_data.columns = [f"{col[0]} ({col[1]})" if col[0] not in ('sample_ID', 'Parent') else col[0] for col in merged_data.columns]
        new_columns = []
        for col in merged_data.columns:
            if col not in ('ID_Sample', 'Parent', 'Antibody'):
                antibody_name = data.loc[data['Antibody'] == col, 'Antibody'].iloc[0] if any(data['Antibody'] == col) else col
                new_columns.append(f"Positivity Rate ({antibody_name})")
            else:
                new_columns.append(col)

        #new_columns.append("Antibody")
        merged_data.columns = new_columns
        return merged_data


    def merge_samples(data, file):

        """
        Merge the files per region with different Positivity rate, so to have dtataframes to consult for the creation of the scatterplots
        :params final_dataframe: Dataframe obtained by process_rate function
        """

        parents = data['Parent'].unique()
        created_files = []

        for parent in parents: 
            parent_df = data[data['Parent'] == parent]
            sample_ids = data['ID_Sample'].unique()

            #Create a df for the current Parent value

            parent_result_df = pd.DataFrame(columns=['ID_Sample'])

            for sample_id in sample_ids:
                sample_id_df = parent_df[parent_df['ID_Sample'] == sample_id]
                if len(sample_id_df) > 1:
                    #Populate the dataframe
                    row = {'ID_Sample' : sample_id}
                    for _, entry in sample_id_df.iterrows():
                        antibody_col = f"Positivity Rate ({entry['Antibody']})"
                        row[antibody_col] = entry['Positivity Rate']

                    parent_result_df = pd.concat([parent_result_df, pd.DataFrame([row])], ignore_index=True)

            if not parent_result_df.empty:
                extension = file.split('.')[1]
                file_name= f"{parent}_data" + f".{extension}"
                #print ("Porva orva", file_name)
                if extension.lower() == 'csv':
                     parent_result_df.to_csv(file_name, index=False)
                elif extension.lower() == 'xlsx':
                    parent_result_df.to_excel(file_name, index=False)
                else:
                    print(f"Unsupported file extension: {extension}. Skipping file {file_name}")

                #parent_result_df.to_csv(file_name, index=False)
                created_files.append(file_name)

        return created_files

    

    @staticmethod
    def process_heatmaps(filename):

        """
        Generate and save correlation heatmaps based on immunopositivity rate data.

        :param filename: Path to the data file for generating heatmaps.
        """

        if filename.endswith(('csv', 'xlsx')):
            if filename.endswith('csv'):
                    graph_data = pd.read_csv(filename)
            elif filename.endswith('xlsx'):
                graph_data = pd.read_excel(filename,  engine='openpyxl')
            
            graph_data = graph_data.dropna()
            
            pos_rate_columns = [col for col in graph_data.columns if 'Positivity Rate' in col]
            
            #print("Data columns:", graph_data.columns)

            if len(pos_rate_columns) < 2:
                print("Not enough 'Positivity Rate' columns found for heatmaps.")
                return 
            #data = graph_data.select_dtypes(include=[np.number])
            #selected_columns = graph_data.filter(like="Positivity Rate")
            selected_columns = graph_data[pos_rate_columns]
            pearson_corr = selected_columns.corr(method='pearson')
            spearman_corr = selected_columns.corr(method='spearman')
            kendall_corr = selected_columns.corr(method='kendall')

            #plt.figure(figsize=(10, 8))
            #sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

            #plt.savefig("pearson_correlation_heatmap.png")
            #plt.show()
            base_filename = os.path.basename(filename)
            prefix = base_filename.split('_')[0]
 
            # Create a single figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))

            # Plot Pearson Correlation in the first subplot
            sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
            axes[0].set_title(f'{prefix} - Pearson Correlation')

            # Plot Spearman Correlation in the second subplot
            sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
            axes[1].set_title(f'{prefix} - Spearman Correlation')

            # Plot Kendall Correlation in the third subplot
            sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
            axes[2].set_title(f'{prefix} - Kendall Correlation')

            # Adjust spacing between subplots
            plt.tight_layout()

            # Save the figure to a file (e.g., "correlation_heatmaps.png")
            plt.savefig(f"{prefix}_correlation_heatmaps.png")


        else:
            print(f"No files has been found to study the correlation of the data, no graphs has been produced.")
            sys.exit(1)

    @staticmethod
    def newprocess_heatmap(filename):
        """
    Generate and save correlation heatmaps based on immunopositivity rate data.

    :param filename: Path to the data file for generating heatmaps.
    """
        if filename.endswith(('.csv', '.xlsx')):
            if filename.endswith('.csv'):
                graph_data = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                graph_data = pd.read_excel(filename, engine='openpyxl')

            graph_data = graph_data.dropna()

            pos_rate_columns = [col for col in graph_data.columns if 'Positivity Rate' in col]

            print("Data columns:", graph_data.columns)

            if len(pos_rate_columns) < 2:
                print("Not enough 'Positivity Rate' columns found for heatmaps.")
                return

            selected_columns = graph_data[pos_rate_columns]
            pearson_corr = selected_columns.corr(method='pearson')
            spearman_corr = selected_columns.corr(method='spearman')
            kendall_corr = selected_columns.corr(method='kendall')

            # Check for statistical significance (e.g., p-value threshold of 0.05)
            significance_threshold = 0.05
            is_pearson_significant = (pearson_corr.apply(lambda x: x.apply(lambda y: y < significance_threshold))).any().any()
            is_spearman_significant = (spearman_corr.apply(lambda x: x.apply(lambda y: y < significance_threshold))).any().any()
            is_kendall_significant = (kendall_corr.apply(lambda x: x.apply(lambda y: y < significance_threshold))).any().any()

            if is_pearson_significant or is_spearman_significant or is_kendall_significant:
                # Create a single figure with three subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 6))

                # Plot Pearson Correlation in the first subplot if significant
                if is_pearson_significant:
                    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
                    axes[0].set_title(f'{prefix} - Pearson Correlation')

                # Plot Spearman Correlation in the second subplot if significant
                if is_spearman_significant:
                    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
                    axes[1].set_title(f'{prefix} - Spearman Correlation')

                # Plot Kendall Correlation in the third subplot if significant
                if is_kendall_significant:
                    sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
                    axes[2].set_title(f'{prefix} - Kendall Correlation')

                # Adjust spacing between subplots
                plt.tight_layout()

                # Save the figure to a file (e.g., "correlation_heatmaps.png")
                base_filename = os.path.basename(filename)
                prefix = base_filename.split('_')[0]
                plt.savefig(f"{prefix}_correlation_heatmaps.png")

            else:
                print("No statistically significant correlations found. No heatmaps will be produced.")

        else:
            print("No files have been found to study the correlation of the data, no heatmaps have been produced.")
            sys.exit(1)


    @staticmethod
    def process_scatterplots(filename):
        """
        Generate and save scatterplots based on immunopositivity rate data.

        :param filename: Path to the data file for generating scatterplots.
        """
        print (filename)
        if filename.endswith(('.csv', '.xlsx')):
            if filename.endswith('.csv'):
                    graph_data = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                graph_data = pd.read_excel(filename, engine='openpyxl')
            

            graph_data = graph_data.dropna()
            # Extract columns with 'Positivity Rate' in their names
            pos_rate_columns = [col for col in graph_data.columns if 'Positivity Rate' in col]
            num_columns = len(pos_rate_columns)
            
            print (graph_data.columns)
            if num_columns < 2:
                print("Not enough 'Positivity Rate' columns found for scatterplots.")
                return

            fig, ax = plt.subplots(figsize=(10, 8))

            for i in range(num_columns):
                for j in range(i+1, num_columns):
                    if i != j:
                        sns.scatterplot(x=graph_data[pos_rate_columns[i]], y=graph_data[pos_rate_columns[j]],label=f'{pos_rate_columns[i]} vs. {pos_rate_columns[j]}')
                        #axes[i, j].set_xlabel(pos_rate_columns[i])
                        #axes[i, j].set_ylabel(pos_rate_columns[j])
                        #axes[i, j].set_title(f'{pos_rate_columns[i]} vs. {pos_rate_columns[j]}')
            
            ax.set_xlabel(pos_rate_columns[0])
            ax.set_ylabel(pos_rate_columns[1])
            ax.set_title(f'{pos_rate_columns[0]} vs. {pos_rate_columns[1]}')

            plt.legend() 
            plt.tight_layout()

            
            # Save the figure with all scatterplots as a single PNG file
            base_filename = os.path.basename(filename)
            prefix = base_filename.split('_')[0]
            plt.savefig(f"{prefix}_scatterplots.png")

            # Close the figure to release resources
            plt.close()


        else:
            print(f"No files has been found to study the correlation of the data, no graphs has been produced.")
            sys.exit(1)




# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MRXS data with the ProcessMRXSData class")
    parser.add_argument("directory_path", help="Path to the directory containing .mrxs files")
    parser.add_argument("inventory_file", help="Path to the inventory file")
    parser.add_argument("output_path", help="Path to the directory containing output files")
    parser.add_argument("output_filename", help="Name of the output CSV file antibody-specific")
    parser.add_argument("output_extension", help="Extension for the output files, can choose between .xlsl and ,csv")

    args = parser.parse_args()


    directory_path = args.directory_path
    inventory_file = args.inventory_file
    output_filename = args.output_filename
    output_path = args.output_path
    output_ex = args.output_extension
    final_data_filename = "final_data" + f".{output_ex}"
    #print ("Final data filename", final_data_filename)

#    final_data = ProcessMRXSData.process_directory(directory_path, inventory_file,  output_path)
#    final_rate = ProcessMRXSData.process_rate(output_path, final_data_filename)

#    ProcessMRXSData.process_heatmaps(final_rate)
#    ProcessMRXSData.process_scatterplots(final_rate)
    

    # Print the final DataFrame
    print(f"Final DataFrame saved to {output_filename}")
    print(f"Final Immunoposivity DataFrame saved to {final_data_filename}")
    print(f"Processing complete.")
    
