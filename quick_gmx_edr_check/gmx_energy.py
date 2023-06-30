import os
import subprocess
import glob
import fnmatch
from fuzzywuzzy import fuzz # conda install -c conda-forge fuzzywuzzy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GMXEnergy:
    """
    A class to handle operations on GROMACS energy files (.edr).

    ...

    Attributes
    ----------
    base_dir : str
        the base directory where the GROMACS simulations are stored.
    edr_files : list
        a list to store the paths of .edr files.
    cwd : str
        current working directory.
    term : str
        term to select for energy analysis.
    load_modules : str
        command to load the necessary modules on a HPC infrastructure.
    discard_perc : float
        What ratio in the beginning should be discarded. Default 0.05 (i.e. first 5%)
    
    Usage Example
    --------------
    1. Instantiate the class:
    gmx = GMXEnergy(
        "directory_with_edr_files",
        load_modules=load_modules
        discard_perc=0.05)

    2. Scan for all .edr files:
    gmx.scan_edr_files()

    3. (Optional) You can search for specific .edr files:
    matches = gmx.search_files_fuzzy('npt_relax_strong_w', fuzzy_threshold=70)
    
    4. Set the .edr files on which you want to run the energy analysis:
    gmx.set_active_files(matches)

    5. Run GROMACS energy analysis for a specific term:
    gmx.run_gmx_energy(term="Potential")

    6. Plot the resulting energy data:
    gmx.plot_energy()

    Note: Ensure GROMACS and the required modules are installed and properly configured in your environment.

    Methods
    -------
    search_files(pattern):
        Searches for .edr files matching a specific pattern.
    search_files_fuzzy(pattern, show_scores=False, fuzzy_threshold=None):
        Searches for .edr files matching a specific pattern using fuzzy matching.
    set_active_files(files):
        Sets the .edr files on which run_gmx_energy() should be run.
    scan_edr_files():
        Searches for .edr files and stores their paths.
    run_gmx_energy(term):
        Runs the GROMACS energy analysis command on each .edr file.
    plot_energy():
        Plots the energy data extracted from each .edr file.
    """
    
    def __init__(self,
                 base_dir,
                 load_modules="module purge && module load intel/env/2018 fftw/intel/single/sse/3.3.8 gromacs/nompi/cpu/intel/single/2018.4 &&",
                 discard_perc=0.05):
        
        self.base_dir = base_dir
        self.edr_files = []
        self.cwd = os.getcwd()
        self.term = None
        self.load_modules = load_modules
        self.discard_perc = discard_perc
        self.__version__ = "1.1.9"
    
    def search_files(self, pattern):
        """
        Searches for .edr files matching a specific pattern.

        Parameters
        ----------
        pattern : str
            A string pattern to match file names.

        Returns
        -------
        list
            A list of matched files.
        """
        matches = [file for file in self.edr_files if fnmatch.fnmatch(file, pattern)]
        return matches

    def search_files_fuzzy(self, pattern, show_scores=False, fuzzy_threshold=None):
        """
        Searches for .edr files matching a specific pattern using fuzzy matching.

        Parameters
        ----------
        pattern : str
            A string pattern to match file names.
        show_scores : bool, optional
            Whether to print the match scores (default is False).
        fuzzy_threshold : int, optional
            A threshold for fuzzy matching (default is 70).

        Returns
        -------
        list
            A list of matched files.
        """
        matches = []
        if fuzzy_threshold == None:
            fuzzy_threshold = 70
        for file in self.edr_files:
            file_name = file.split('/')[-1]  # Extract the file name from the full path
            match_score = fuzz.ratio(file_name, pattern)
            if show_scores == True:
                print(match_score, file_name)
            if match_score >= fuzzy_threshold:  # Define a threshold for fuzzy matching
                matches.append(file)
        return pd.Series(matches).unique().tolist()

    def set_active_files(self, files):
        """
        Sets the .edr files on which run_gmx_energy() should be run.

        Parameters
        ----------
        files : list
            A list of .edr files on which the GROMACS energy analysis should be run.
        """
        self.edr_files = files

    def scan_edr_files(self):
        """
        Searches for .edr files and stores their paths in the `edr_files` attribute.
        """
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".edr"):
                    self.edr_files.append(os.path.join(root, file))
                    
    def run_gmx_energy(self, term):
        """
        Runs the GROMACS energy analysis command on each .edr file in `self.edr_files`. 

        Parameters
        ----------
        term : str
            The term to select for energy analysis. This is the quantity that will be extracted from the .edr files. 
        """
        self.term = term
        for edr_file in self.edr_files:
            dir_path, filename = os.path.split(edr_file)
            # Switch to directory containing .edr file
            os.chdir(dir_path)
            
            # Remove old energy.xvg files
            for old_file in glob.glob("#energy*#"):
                os.remove(old_file)
                
            gmx_energy = f'{self.load_modules} echo \"{term}\" | gmx energy -f {filename}'
            subprocess.run(gmx_energy, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Switch back to base directory
            os.chdir(self.cwd)

    def plot_energy(self, temp_discard_perc=None):
        """
        Plots the energy data extracted from each .edr file.

        This method will read the energy.xvg file generated by `run_gmx_energy()` and plot the data it contains. 
        The first 5% of the data is discarded to ensure good plot scaling.
        
        Parameters
        ----------
        temp_discard_perc : float
            Plot with different discard percentage. Do not change value with which object was initialized.
        """
        
        if not temp_discard_perc == None:
            discard_perc = temp_discard_perc
        else:
            discard_perc = self.discard_perc
        
        for edr_file in self.edr_files:
            dir_path, filename = os.path.split(edr_file)
            energy_file = os.path.join(dir_path, "energy.xvg")

            if os.path.exists(energy_file):
                data = pd.read_csv(energy_file, comment='@', skiprows=13, delim_whitespace=True, header=None)

                plt.figure(figsize=(10, 6))
                for idx, column in enumerate(data.columns[1:]):
                    label = self.term.split(" ")[idx]
                    length = len(data[0])
                    idx_discard = int(round(length * discard_perc))
                    print("Number of trajectory energy steps: " + str(length) + " steps discarded: " + str(idx_discard))
                    plt.plot(data[0][idx_discard:], data[column][idx_discard:], label=f"Column {label}")

                plt.xlabel("Time")
                plt.legend()
                plt.title(f"Energy for {filename}\nExcluding the first {discard_perc*100}% trajectory")
                plt.show()
