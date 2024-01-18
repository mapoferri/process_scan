# ProcessMRXSData

ProcessMRXSData is a Python class designed to process data from MRXS files, merge it with inventory data, calculate immunopositivity statistics, and generate useful visualizations like heatmaps and scatterplots.

## Prerequisites

- Python 3.7 or higher
- Required Python packages: `pandas`, `numpy`, `seaborn`, `matplotlib`

## Non-expert Usage

1. Clone or download this repository to your local machine.

2. Extract the directory (whenever you like) and open the directory.

```
unzip process_mrxs-main -d process_mrxs
cd process_mrxs
```

3. Call directly the class (internal script providing already with the resulting rates and images):

There are different ways to do it via BASH terminal:

Using the already workflow\_template in this directory: 

```
python workflow_template.py path/to/your/mrxs_files path/to/your/inventory.csv path/to/your/directory/to/outputs

```


Calling the script directly with the path stated for complete:

```
python process_mrxs_data.py path/to/your/mrxs_files path/to/your/inventory.csv path/to/your/directory/to/outputs
```

Or using an ENV (environmental) variables.
```
EXPORT mrxs_directory="path/to/your/mrxs_files"
EXPORT inventory_file="path/to/your/inventory.csv"
EXPORT output_path="path/to/your/directory/to/outputs" #make sure to create it previously
```

```python

python process_mrxs_data.py $mrxs_directory $inventory_file $output_path 
```  

### Limitations

Please, consider that the current development is only considering to be launched with the Prerequisites complied on a SH Operating System. 

The Inventory file can be .csv or .xslx, and multi-sheet format is expected.

When defining the output\_filename, both the option .csv and .xslx are implemented, so could choose based on your preference.


## Python Usage

1. Clone or download this repository to your local machine.

2. Import the `ProcessMRXSData` class into your Python script:

```python
from process_mrxs_data import ProcessMRXSData

```
3. Create an instance of the ProcessMRXSData class by providing paths to your MRXS files in a specific directory and inventory file, as well as the other files necessary to launch the call:

```python
mrxs_directory = "path/to/your/mrxs_files"
inventory_file = "path/to/your/inventory.csv"
output_path = "path/to/your/directory/for/collecting/all/dataframes"
output_filename = "path/to/common/output/dataframe"

processor = ProcessMRXSData.process_directory(mrxs_directory, inventory_file, output_path, output_filename)
```

4. Call the other functions for the rate calculation and relative images:


```python

rate = ProcessMRXSData.process_rate(output_path, output_filename)

ProcessMRXSData.process_heatmaps(rate)
ProcessMRXSData.process_scatterplots(rate)

```
5. Check the output images in the output\_path 


## Functions specifics

Call the process\_data method to process MRXS data, merge it with inventory, and calculate immunopositivity statistics for every slide (this function is called internally for each file in the process\_directory method):

```python
result_df = processor.process_data()
```

To process immunopositivity rate from Excel/CSV files and merge into a unique main DataFrame, call the process\_positivity method:

```python 
xlsx_file = "path/to/your/immunopositivity_data.xlsx"
final_df = ProcessMRXSData.process_positivity(xlsx_file, result_df)
```

Process MRXS data from a directory, save antibody-specific data, and return the final DataFrame using the process\_directory method (internally calling the process\_data):


```python 
directory_path = "path/to/your/mrxs/files/directory"
output_path = "path/to/output/data/directory"
output_filename = "output_data.csv"  # Name of the output CSV file
final_data = ProcessMRXSData.process_directory(directory_path, inventory_file, output_path, output_filename)
```

Process immunopositivity rate from saved files and merge it into a final DataFrame using the process\_rate method:

```python
final_rate = ProcessMRXSData.process_rate(output_path, final_data)
```

To generate and save correlation heatmaps based on immunopositivity rate data, call the process\_heatmaps method:

```python
data_file = "path/to/your/immunopositivity_data.csv"
ProcessMRXSData.process_heatmaps(data_file)
```

To generate and save scatterplots based on immunopositivity rate data, use the process\_scatterplots method:

```python
data_file = "path/to/your/immunopositivity_data.csv"
ProcessMRXSData.process_scatterplots(data_file)
```

For more details, examples, and command-line usage, please refer to the code and documentation in this repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Note: This is a simplified example README file. You should replace the placeholder paths and filenames with the actual paths to your data files and directories.
