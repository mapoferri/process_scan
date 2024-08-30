import pandas as pd
import os
import argparse

class PositivityRateMerger:
    def __init__(self, patient_file_path, positivity_rate_dir, output_file_path):
        if not os.path.isfile(patient_file_path):
            raise FileNotFoundError(f"The patient file {patient_file_path} does not exist.")
        self.patient_file_path = patient_file_path
        self.positivity_rate_dir = positivity_rate_dir
        self.output_file_path = output_file_path
        self.patient_df = pd.read_excel(patient_file_path)

    def merge_files(self):
        # List to hold DataFrames for merging
        data_frames = [self.patient_df]

        # Iterate over all files in the directory
        for file_name in os.listdir(self.positivity_rate_dir):
            if file_name.endswith('.xlsx') or file_name.endswith('.csv'):
                # Determine the file path
                file_path = os.path.join(self.positivity_rate_dir, file_name)
                
                # Load the positivity rate file into a DataFrame
                if file_name.endswith('.xlsx'):
                    positivity_df = pd.read_excel(file_path)
                else:
                    positivity_df = pd.read_csv(file_path)
                
                # Extract the antibody name from the file name
                antibody_name = os.path.splitext(file_name)[0]  # e.g., "Antibody1" from "Antibody1.xlsx"
                
                # Prepare the positivity DataFrame with the antibody-specific column
                positivity_df = positivity_df[['ID_Sample', 'Positivity Rate']]
                positivity_df.rename(columns={'Positivity Rate': f'{antibody_name} Positivity Rate'}, inplace=True)
                positivity_df = positivity_df.drop_duplicates(subset=['ID_Sample'])

                # Merge with the patient DataFrame
                data_frames.append(positivity_df)

        for idx, df in enumerate(data_frames):
            print(f"DataFrame {idx} columns: {df.columns.tolist()}")
            print(df.head())

        # Concatenate all DataFrames along columns

        final_df = data_frames[0]
        for df in data_frames[1:]:
            final_df = pd.merge(final_df, df, on='ID_Sample', how='left')

        positivity_rate_columns = [col for col in final_df.columns if 'Positivity Rate' in col]
        final_df = final_df.dropna(subset=positivity_rate_columns, how='all')

        # Save the final DataFrame to the output file

        file_ext = os.path.splitext(self.output_file_path)[1]
        if file_ext == '.csv':
            final_df.to_csv(self.output_file_path, index=False)
        elif file_ext in ['.xlsx', '.xls']:
            final_df.to_excel(self.output_file_path, index=False)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        print(f"Data has been successfully merged and saved to {self.output_file_path}.")



def main():
    parser = argparse.ArgumentParser(description='Merge positivity rate files with a patient file.')
    parser.add_argument('patient_file', type=str, help='Path to the patient file (xlsx or csv)')
    parser.add_argument('positivity_rate_dir', type=str, help='Directory containing positivity rate files (xlsx or csv)')
    parser.add_argument('output_file', type=str, help='Path to the output file (xlsx)')

    args = parser.parse_args()

    # Create an instance of PositivityRateMerger
    merger = PositivityRateMerger(args.patient_file, args.positivity_rate_dir, args.output_file)
    merger.merge_files()

if __name__ == '__main__':
    main()
