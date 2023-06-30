# Quick GMX EDR Check: A Tool for Fast Exploration of GROMACS Energy Files

The **"Quick GMX EDR Check"** is a Python toolkit designed to assist in the quick and efficient analysis of GROMACS energy files (.edr). This toolkit is particularly useful for traversing complex directory structures and searching for specific simulation conditions using fine-tunable fuzzy search. This toolkit is also capable of working with a HPC infrastructure, allowing it to load necessary modules for GROMACS.

Disclaimer: Please note that this toolkit is primarily intended for fast exploration and convergence checking. The plotting functionality provided is rudimentary and might not meet all your needs for sophisticated data visualization.

## Features
 - Traverse complex directory structures to find and analyze GROMACS .edr files.
 - Flexible file selection using fine-tunable fuzzy search.
 - Quick and simple convergence checking.
 - Basic plotting for fast data exploration.

## Installation

You can install Quick GMX EDR Check directly from this GitHub repository using pip:

```
pip install git+https://github.com/TobiasMaterzok/quick_gmx_edr_check.git
```

## Usage

After installing the toolkit, you can use it in your Python scripts or Jupyter notebooks like this:

```
from quick_gmx_edr_check import GMXEnergy

# Create a GMXEnergy instance and specify the base directory and module load commands with which you load Gromacs
gmx = GMXEnergy(
    base_dir="base_directory_path", 
    load_modules="module load X &&"
)

# e.g., ~/Simulations/role_of_seq/ from AIDPET on my HPC environment
gmx = GMXEnergy(
    base_dir="~/Simulations/role_of_seq/", 
    load_modules="module purge && module load intel/env/2018 fftw/intel/single/sse/3.3.8 gromacs/nompi/cpu/intel/single/2018.4 &&"
)

# Scan for .edr files
gmx.scan_edr_files()

# Print out all found .edr files in the whole project directory
print(gmx.edr_files)

# Use fuzzy search to select files
matches = gmx.search_files_fuzzy('search_pattern', fuzzy_threshold=70)

# You can display fuzzy score and the files with
matches = gmx.search_files_fuzzy('search_pattern', show_scores=True)

# Set the active files
gmx.set_active_files(matches)

# Run GROMACS energy analysis
gmx.run_gmx_energy(term="Potential")

# Plot the data
gmx.plot_energy()
```

Please replace **"base_directory_path"** and **"search_pattern"** with the actual base directory and search pattern you wish to use.
Make sure you set **load_modules** correctly for your HPC environment.

## Contributing

Contributions to Quick GMX EDR Check are welcome! Please feel free to submit a pull request or open an issue on GitHub.
