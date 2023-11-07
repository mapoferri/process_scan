import pandas as pd
import os
import sys
import numpy as np
from pandas import read_excel
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns


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
        df_mrxs = pd.read_csv(self.mrxs_file, sep='\t')
        df_mrxs['Image'] = df_mrxs['Image'].str.rstrip('.mrxs')
        #print("DataFrame from text file:")
        #print(df_mrxs)
        
        if 'Class' not in df_mrxs.columns:
            print(f"Skipping MRXS file: {self.mrxs_file} - 'Class' column not found.")
            return None

        
        final_df = pd.DataFrame() #Initial empty dataframe
        
        # Start a counter for the Class column
        positive_counter = (df_mrxs['Class'] == 'PositiveCell').sum()
        negative_counter = (df_mrxs['Class'] == 'NegativeCell').sum()

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
                    result = df_mrxs.merge(sheet_df, left_on='Image', right_on='ID_Slidescanner', how='inner')
                    if not result.empty:
                        #antibody = sheet_name
                        merged_df = pd.concat([merged_df, result], ignore_index=True)

        # Create the merged dataframe based on the correspondence between the Image value and the ID_Slidescanner value in the Inventory
                        if not merged_df.empty:
                            final_df = merged_df[['HD', 'Antibody', 'ID_Slidescanner', 'Image']]
                            final_df = final_df.head(1)
                            final_df['ID_Slidescanner'] = final_df['ID_Slidescanner'].values[0]
                            final_df['Image'] = final_df['Image'].values[0]
                            #print(f"FINAL")
                            #print(final_df.columns)
                        else:
                            print("Merged DataFrame is empty.")
                            final_df = pd.DataFrame({'HD': [None], 'Antibody': [None], 'ID_Slidescanner': [None], 'Image': [None]})
                            return final_df
        

        # Create the final DataFrame

        final_df['Positive Class'] = positive_counter
        final_df['Negative Class'] = negative_counter
        #print("Final DataFrame with counters:")
        #print(final_df)

        # Calculate Positivity Rate
        if positive_counter == 0 or negative_counter == 0:
            print(f"Error: Counter for 'PositiveCell' or 'NegativeCell' is zero for {self.mrxs_file}. Skipping Positivity Rate calculation.")
        else:
            final_df['Positivity Rate'] = (positive_counter * 100) / (positive_counter + negative_counter)

        print(f"Processed data for {self.mrxs_file}.")
        
        return final_df

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
        hd_and_rate = df_xls[['HD', 'Positivity Rate']]
        
        # print(f"Checking columsn {hd_and_rate}")
        
        # Merge the extracted data based on the 'HD' column
        final_df = final_df.merge(hd_and_rate, on='HD', how='left')

        return final_df


    @staticmethod
    def process_directory(directory_path, inventory_file, output_path):
        """
        Process MRXS data from a directory, save antibody-specific data, and return the final DataFrame.

        :param directory_path: Path to the directory containing .mrxs files.
        :param inventory_file: Path to the inventory file (CSV or Excel).
        :param output_path: Path to save antibody-specific data.
        
        :return: Final DataFrame.
        """

        #Array of the dataframes produced for each file
        result_dfs= []
        final_df = pd.DataFrame(columns=['HD'])
        no_mrxs_files = True

        # For each slide, call the process_data function
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                mrxs_file = os.path.join(directory_path, filename)
                print(f"Processing file: {mrxs_file}")
                no_mrxs_files = False
                
                processor = ProcessMRXSData(mrxs_file, inventory_file)
                result_df = processor.process_data()
                print (result_df)
                result_dfs.append(result_df)
                # Appending to an array, so should be fine

            
        if no_mrxs_files:
            print("No files with .mrxs extension found in the specified directory.")
            sys.exit(1)

         
        # Concatenate all result DataFrames into a single DataFrame
        final_result = pd.concat(result_dfs)
        
        grouped = final_result.groupby("Antibody")
        for group_name, group_data in grouped:
            output_filename = f"{group_name}_data.csv"
            output_filepath = os.path.join(output_path, output_filename) 
            group_data.to_csv(output_filepath, index=False)
            print(f"Saved data for {group_name} to {output_filename}")

        # Save the final DataFrame to a CSV file with the specified name
        #with open(output_filename, 'w', encoding='utf-8') as file:
            #final_result.to_csv(file, index=False)
            #final_result.to_excel(output_filename, index=False)


        return final_df

    @staticmethod
    def process_rate(output_path, final_data_filename):

        final_df = pd.DataFrame(columns=['HD'])
        no_xls_files = True
        for filename in os.listdir(output_path):
            if filename.endswith(('.xlsx', 'csv')):
                xlsx_file = os.path.join(output_path, filename)
                print(f"Processing file: {xlsx_file}")
                no_xls_files = False

                #processor = ProcessMRXSData.process_positivity(xlsx_file, final_df)
                
                if filename.endswith('.csv'):
                    xlsx_data = pd.read_csv(xlsx_file)
                elif filename.endswith('.xlsx'):
                    xlsx_data = pd.read_excel(xlsx_file)

                if all(col in xlsx_data.columns for col in ['HD', 'Antibody', 'Positivity Rate']):
                    data = xlsx_data[['HD', 'Positivity Rate']]
                    data = data.copy()
                    data.rename(columns={'Positivity Rate': f"Positivity Rate ({xlsx_data['Antibody'].iloc[0]})"}, inplace=True)

                    final_df = pd.merge(final_df, data, on='HD', how='outer')
        
        if no_xls_files:
            print(f"No files has been found to study the correlation of the data, check them.")
            sys.exit[1]
        
        if final_data_filename.endswith('.csv'):
            final_df.to_csv(final_data_filename, index=False)
        elif final_data_filename.endswith('.xlsx'):
            final_df.to_excel(final_data_filename, index=False)
        return final_data_filename


    @staticmethod
    def process_heatmaps(filename):

        """
        Generate and save correlation heatmaps based on immunopositivity rate data.

        :param filename: Path to the data file for generating heatmaps.
        """

        if filename.endswith(('.csv', '.xlsx')):
            if filename.endswith('.csv'):
                    graph_data = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                graph_data = pd.read_excel(filename, engine='xlrd')
            data = graph_data.select_dtypes(include=[np.number])
            selected_columns = graph_data.filter(like="Positivity Rate")
            pearson_corr = selected_columns.corr(method='pearson')
            spearman_corr = selected_columns.corr(method='spearman')
            kendall_corr = selected_columns.corr(method='kendall')

            #plt.figure(figsize=(10, 8))
            #sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

            #plt.savefig("pearson_correlation_heatmap.png")
            #plt.show()
            
            # Create a single figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))

            # Plot Pearson Correlation in the first subplot
            sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
            axes[0].set_title('Pearson Correlation')

            # Plot Spearman Correlation in the second subplot
            sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
            axes[1].set_title('Spearman Correlation')

            # Plot Kendall Correlation in the third subplot
            sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
            axes[2].set_title('Kendall Correlation')

            # Adjust spacing between subplots
            plt.tight_layout()

            # Save the figure to a file (e.g., "correlation_heatmaps.png")
            plt.savefig("correlation_heatmaps.png")


        else:
            print(f"No files has been found to study the correlation of the data, no graphs has been produced.")
            sys.exit(1)


    @staticmethod
    def process_scatterplots(filename):
        """
        Generate and save scatterplots based on immunopositivity rate data.

        :param filename: Path to the data file for generating scatterplots.
        """

        if filename.endswith(('.csv', '.xlsx')):
            if filename.endswith('.csv'):
                    graph_data = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                graph_data = pd.read_excel(filename, engine='xlrd')

            # Extract columns with 'Positivity Rate' in their names
            pos_rate_columns = [col for col in graph_data.columns if 'Positivity Rate' in col]
            num_columns = len(pos_rate_columns)

            if num_columns < 2:
                print("Not enough 'Positivity Rate' columns found for scatterplots.")
                return

            fig, axes = plt.subplots(num_columns, num_columns, figsize=(12, 8))

            for i in range(num_columns):
                for j in range(num_columns):
                    if i != j:
                        sns.scatterplot(x=graph_data[pos_rate_columns[i]], y=graph_data[pos_rate_columns[j]], ax=axes[i, j])
                        axes[i, j].set_xlabel(pos_rate_columns[i])
                        axes[i, j].set_ylabel(pos_rate_columns[j])
                        axes[i, j].set_title(f'{pos_rate_columns[i]} vs. {pos_rate_columns[j]}')

            plt.tight_layout()


            # Save the figure with all scatterplots as a single PNG file
            plt.savefig("scatterplots.png")

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


    args = parser.parse_args()


    directory_path = args.directory_path
    inventory_file = args.inventory_file
    output_filename = args.output_filename
    output_path = args.output_path
    final_data_filename = "final_data.csv"

    final_data = ProcessMRXSData.process_directory(directory_path, inventory_file,  output_path)
    final_rate = ProcessMRXSData.process_rate(output_path,final_data_filename)

    ProcessMRXSData.process_heatmaps(final_rate)
    ProcessMRXSData.process_scatterplots(final_rate)
    

    # Print the final DataFrame
    print(f"Final DataFrame saved to {output_filename}")
    print(f"Final Immunoposivity DataFrame saved to {final_data_filename}")
    print(f"Processing complete.")
