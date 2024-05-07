# Process Scanning Data

ProcessSCanningData is a Python class designed to process data from MRXS, NDPI and SVS files, merge it with inventory data, calculate immunopositivity statistics, and generate useful visualizations like heatmaps and scatterplots.


This is the second half of the HDAB - Immunopositivity Tutorial available here ().


## Prerequisites

- Python 3.8
- Required Python packages: `pandas==1.5.3`, `numpy==1.21.4`, `seaborn==0.11.2`, `matplotlib`, `scipy==1.7.3` , `openpyxl==3.0.10`
- Pip library opepyl > 3.2.0
- QuPath 5.01 required!

## Non-expert Usage

1. Clone or download this repository to your local machine.

2. Extract the directory (whenever you like) and open the directory.

```
unzip process_scan-main
cd process_scan-main
```

3. Call directly the class (internal script providing already with the resulting rates and images):

There are different ways to do it via BASH terminal:

Using the already workflow\_template in this directory: 

### In this example the reference files are gonna be the mrxs, but it will work with the aforementioned scanning files.

```
python workflow_template.py path/to/your/results path/to/your/Inventory.xlsx path/to/your/directory/to/outputs xlsx/csv (you will need to choose one of the two)

```


Calling the script directly with the path stated for complete:
### In this example the reference files are gonna be the mrxs, but it will work with the aforementioned scanning files.
#### Please, change the class according to your scanning images files extension: process_mrxs_data.py, process_svs_data.py and/or process_ndpi_data.py

```
python process_mrxs_data.py path/to/your/results path/to/your/Inventory.xlsx path/to/your/directory/to/outputs xlsx/csv
```

Or using an ENV (environmental) variables.
```
EXPORT mrxs_directory="path/to/your/mrxs_files"
EXPORT inventory_file="path/to/your/Inventory.xlsx"
EXPORT output_path="path/to/your/directory/to/outputs" #make sure to create it previously
EXPORT output_extension="xlsx/csv"
```

```python

python process_mrxs_data.py $mrxs_directory $inventory_file $output_path $output_extension
```  

### Limitations

Please, consider that the current development is only considering to be launched with the Prerequisites complied on a SH Operating System. 

The Inventory file can be .csv or .xslx, and multi-sheet format is expected.

When defining the output\_filename, both the option .csv and .xslx are implemented, so could choose based on your preference.


## Python Usage

1. Clone or download this repository to your local machine.

2. Import the `ProcessScanData` class into your Python script:

###  Please, change the class according to your scanning images files extension: ProcessMRXSData, ProcessSVSData and/or ProcessNDPIData

```python
from process_mrxs_data import ProcessMRXSData

```
3. Create an instance of the ProcessMRXSData class by providing paths to your MRXS files in a specific directory and inventory file, as well as the other files necessary to launch the call:

```python
mrxs_directory = "path/to/your/mrxs_files"
inventory_file = "path/to/your/Inventory.xslx"
output_path = "path/to/your/directory/to/outputs"
output_extension = "xlsx/csv"

processor = ProcessMRXSData.process_directory(mrxs_directory, inventory_file, output_path, output_extension)
```

4. Call the other functions for the rate calculation and relative images:


```python

rate = ProcessMRXSData.process_rate(output_path, output_filename)
for file in rate:
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

final_data = ProcessMRXSData.process_directory(directory_path, inventory_file, output_path, output_extension)
```

Process immunopositivity rate from saved files and merge it into a final DataFrame using the process\_rate method:

```python
final_rate = ProcessMRXSData.process_rate(output_path, final_filename)
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
