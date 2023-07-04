import os
import subprocess
import glob
import fnmatch
from fuzzywuzzy import fuzz # conda install -c conda-forge fuzzywuzzy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import re
from scipy.optimize import curve_fit

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
    terms : list
        terms to select for energy analysis.
    load_modules : str
        command to load the necessary modules on a HPC infrastructure.
    discard_perc : float
        What ratio in the beginning should be discarded. Default 0.05 (i.e. first 5%)
    energy_data : dict
        Dictionary to store the energy data extracted from .edr files.
    term_to_col_name : dict
        Mapping of term names to gmx energy column names.
    __version__ : str
        Version number of the class.
    
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
    gmx.run_gmx_energy(terms=["Potential","Pressure"])

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
    run_gmx_energy(terms):
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
        self.terms = None
        self.load_modules = load_modules
        self.discard_perc = discard_perc
        self.energy_data = {}
        self.term_to_col_name = {}
        self._counter = 0
        self.__version__ = "1.2.3"
        
    
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

    def search_files_fuzzy(self, pattern, files=None, show_scores=False, fuzzy_threshold=None):
        """
        Searches for .edr files matching a specific pattern using fuzzy matching.

        Parameters
        ----------
        pattern : str
            A string pattern to match file names.
        files : list, optional
            A list of files to search among. If not provided, all files are searched.
        show_scores : bool, optional
            Whether to print the match scores (default is False).
        fuzzy_threshold : int, optional
            A threshold for fuzzy matching (default is 70).

        Returns
        -------
        list
            A list of matched files.
        """
        if files is None:
            files = self.edr_files
        matches = []
        if fuzzy_threshold is None:
            fuzzy_threshold = 23
            
        for file in files:
            match_score = fuzz.ratio(file, pattern)
            if show_scores == True:
                print(match_score, file)
            if match_score >= fuzzy_threshold:
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
                    
    def sort_files_by_mtime(self, files=None):
        """
        Sorts the .edr files in the class attribute `edr_files` based on their modification time (mtime).

        This function modifies the order of the .edr files in-place, arranging them in ascending order of their modification time.

        Parameters
        ----------
        files : list, optional
            A list of files to sort by date. If not provided, edr_files are sorted..

        Returns
        -------
        None

        Example
        -------
        gmx = GMXEnergy("directory_with_edr_files")
        gmx.scan_edr_files()
        gmx.sort_files_by_mtime()
        print(gmx.edr_files)
        # Output: The .edr files sorted by modification time.

        """
        if files is None:
            self.edr_files.sort(key=os.path.getmtime)
        else:
            files.sort(key=os.path.getmtime)

    def run_gmx_energy(self, files=None, terms=None):
        """
        Runs the GROMACS energy analysis command on each .edr file in `self.edr_files`. 

        Parameters
        ----------
        term : str
            The term to select for energy analysis. This is the quantity that will be extracted from the .edr files. 
        """
        if terms is None and self.terms is None:
            raise ValueError("Error: Terms not specified")
        elif terms is None:
            terms = self.terms
        else:
            self.terms = terms
        if files is None:
            files = self.edr_files
        elif isinstance(files, str):
            files = [files]

        for term in self.terms:
            prefix = ''.join(term.split(" "))
            for edr_file in files:
                dir_path, filename = os.path.split(edr_file)
                # Switch to directory containing .edr file
                os.chdir(dir_path)
                
                # Remove old energy.xvg files
                for old_file in glob.glob(f"#energy_QGEC_{prefix}_{filename}*#"):
                    os.remove(old_file)
                
                self._counter = self._counter + 1
                gmx_energy = f'{self.load_modules} echo {term} | gmx energy -f {filename} -o energy_QGEC_{prefix}_{filename}.xvg'
                #subprocess.run(gmx_energy, shell=True, check=True)
                subprocess.run(gmx_energy, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Switch back to base directory
                os.chdir(self.cwd)
            
    def _get_column_names(self, energy_file):
        """
        Extract column names from a GROMACS energy (.xvg) file.
        
        The method reads the .xvg file and parses the headers to extract the column names. It also appends the units to the column names, if available.
        
        Parameters
        ----------
        energy_file : str
            The file path to the .xvg file.

        Returns
        -------
        column_names : list of str
            The list of column names (with appended units, if available) extracted from the .xvg file.

        Notes
        -----
        The method assumes that the .xvg file has a specific header structure. It expects that:
        - The x-axis label line (`@xaxislabel`) appears before the y-axis label line (`@yaxislabel`).
        - The y-axis label line appears before the lines starting with '@s'.
        - The number of units in the y-axis label line is equal to or greater than the number of lines starting with '@s'.
        """
        with open(energy_file, 'r') as f:
            lines = f.readlines()
            
        column_names = []
        units = []
        num_columns = 0  # Keep track of the number of columns found so far

        for line in lines:
            line_cleaned = ''.join(line.split(" "))
            if line_cleaned.startswith('@xaxislabel'):
                # Extract first column name.
                column_name = line[line.index('\"')+1:line.rindex('\"')]
                column_names.append(column_name)
            if line_cleaned.startswith('@yaxislabel'):
                # Extract y-axis labels (units).
                unit_labels = line[line.index('\"')+1:line.rindex('\"')]
                units = unit_labels.split(", ")
            if line_cleaned.startswith('@s'):
                # Extract column name. It's the part of the string between double quotes
                column_name = line[line.index('\"')+1:line.rindex('\"')]
                # Add the unit to the column name if it exists.
                if num_columns < len(units):
                    column_name += ' ' + units[num_columns]
                column_names.append(column_name)
                num_columns += 1
                
        return column_names

    def get_data_col_name(self, term):
        """
        This function returns a full data column name for the specified term.
        """
        # this assumes that self.term_to_col_name is a dictionary mapping terms to data column names
        return self.term_to_col_name[term]


    def extract_data(self, files=None, terms=None):
        if terms is None and self.terms is None:
            raise ValueError("Error: Terms not specified")
        elif terms is None:
            terms = self.terms
        else:
            self.terms = terms
        if files is None:
            files = self.edr_files
        elif isinstance(files, str):
            files = [files]

        for term in terms:
            prefix = ''.join(term.split(" "))
            for edr_file in files:
                dir_path, filename = os.path.split(edr_file)
                energy_file = os.path.join(dir_path, f"energy_QGEC_{prefix}_{filename}.xvg")
                
                # Check if the energy.xvg file exists, if not run gmx energy
                if not os.path.isfile(energy_file):
                    original_terms = self.terms = self.terms
                    self.run_gmx_energy(files=[edr_file], terms=[term])
                    self.terms = original_terms
                    
                column_names = self._get_column_names(energy_file)
                data = pd.read_csv(energy_file, comment='@', skiprows=13, delim_whitespace=True, names=column_names)
                self.energy_data[(edr_file, term)] = data

                # Create mapping of term to column name with units.
                # The term should be associated with its corresponding column name in column_names (not "Time").
                # As the structure of column_names is ["Time (ps)", "term (unit)"], the term's column name is at index 1.
                self.term_to_col_name[term] = column_names[1]
    
    def get_data(self, files=None, terms=None, patterns=None):
        """
        This function returns a DataFrame containing the data from the specified files and terms.
        """
        if terms is None and self.terms is None:
            raise ValueError("Error: Terms not specified")
        elif terms is None:
            terms = self.terms
        else:
            self.terms = terms
        if files is None:
            files = self.edr_files
        elif isinstance(files, str): 
            files = [files]
        
        data = []
        counter = 0
        for file in files:
            for term in terms:
                if (file, term) not in self.energy_data:
                    original_terms = self.terms
                    print("get_data(): Didnt find data in self.energy_data, extracting data...")
                    self.extract_data(files=[file], terms=[term])
                    self.terms = original_terms
                df = self.energy_data[(file, term)]
                df['file'] = file
                df['term'] = self.term_to_col_name[term]
                df['order'] = range(counter, counter + len(df))

                data.append(df)
            counter += len(df)
                
        data = pd.concat(data, ignore_index=True)
        data = data.pivot_table(index=['Time (ps)', 'file', 'order'], columns='term')
        data.columns = data.columns.get_level_values(1)
        data.reset_index(inplace=True)
        data = data.sort_values('order')
        data.set_index('order', inplace=True)

        if patterns is not None:
            data = self.extract_info_from_data(data, patterns)
            
        return data

    @staticmethod
    def extract_info_from_data(data, patterns):
        if patterns is not None and data is not None:
            # Pre-compile regular expression object for faster matching
            compiled_pattern = GMXEnergy.convert_to_regex(patterns)
            
            encoded_info = data['file'].apply(lambda path: pd.Series(GMXEnergy.extract_info_from_path(path, compiled_pattern, patterns), dtype='object'))

            data = pd.concat([data, encoded_info], axis=1)
        else:
            raise ValueError("Error: Data and patterns not specified")
        
        return data

    @staticmethod
    def convert_to_regex(patterns):
        """
        Converts pattern specification to regex patterns.

        Args:
            patterns: Dictionary where keys are the name of the information and values are the patterns.

        Returns:
            Dictionary where keys are the provided patterns and values are regex patterns.
        """
        regex_patterns = {}
        for key, value in patterns.items():
            value = value.replace("{float_no_decimal}", "(\d+)")
            value = value.replace("{decimal}", "(\d+\.\d+)")
            value = value.replace("{alphanumeric}", "([a-zA-Z0-9]+)")  # to match a sequence of alphanumeric characters
            value = value.replace("{string}", "([a-zA-Z]+)")  # to match a sequence of  character
            regex_patterns[key] = re.compile(value) # regex_patterns[key] = value

        # Sort keys in reverse order to avoid partial matching (e.g. 'sample' before 'stress_sample')
        sorted_patterns = {k: regex_patterns[k] for k in sorted(regex_patterns, key=len, reverse=True)}

        return sorted_patterns

    @staticmethod
    def extract_info_from_path(path, compiled_pattern, original_patterns):
        """
        Extracts information based on provided patterns from the file path.

        Args:
            path: String representing the file path.
            compiled_pattern: Pre-compiled regular expressions
            original_patterns: Dictionary where keys are the name of the information and values are the patterns.

        Returns:
            Dictionary where keys are the provided patterns and values are the extracted information.
        """

        # Extract parts of the path
        path_parts = path.split('/')

        extracted_info = {}
        for part in path_parts:
            for key, pattern in compiled_pattern.items():
                match = pattern.search(part)
                if match:
                    #print("Matched!: ", match, key, original_patterns[key], match.group(1))
                    if "{decimal}" in original_patterns[key]:
                        extracted_info[key] = float(match.group(1))
                    elif "{float_no_decimal}" in original_patterns[key]:
                        extracted_info[key] = float(match.group(1))
                    elif "{alphanumeric}" in original_patterns[key]: 
                        extracted_info[key] = match.group(1)
                    elif "{string}" in original_patterns[key]: 
                        extracted_info[key] = match.group(1)

        return extracted_info

    def cumulative_time_shift(self, data, time_column='Time (ps)', group_column='file'):
        """
        Calculate and apply cumulative time shifts based on maximum time in each group.

        Parameters:
        data (pd.DataFrame): The data to be shifted.
        time_column (str): The name of the time column in the data.
        group_column (str): The name of the column to group data by.

        Returns:
        pd.DataFrame: The data with an additional column for shifted time.
        """
        # Calculate the maximum time for each file to determine the time shift required for each group.
        max_time_per_group = data.groupby(group_column, sort=False)[time_column].max()

        # Calculate the cumulative time shift for each group.
        cumulative_time_shift = max_time_per_group.cumsum().shift(fill_value=0)

        # Apply the time shift to each group, creating a new "Cum Time (ps)" column in the DataFrame.
        data['Cum. Time (ps)'] = data.groupby(group_column, sort=False)[time_column].transform(lambda x: x + cumulative_time_shift[x.name])
        
        return data
    
    def process_independent_samples(self, data, groupby=["sample", "water_strain"]):
        """
        This function processes each sample independently over time, computing a separate 
        property (e.g., stress, force, pressure) relationship for each one. This approach is 
        advantageous when the samples may have significant variability due to distinct 
        morphologies, topologies, or conditions. Note that this method might overestimate 
        the uncertainty if the variability between samples is not significant. The function 
        returns parameterized data for each sample individually.
        """
        # Create a dictionary with an entry for each unique sample
        processed_data = {name: group for name, group in data.groupby(groupby)}
        return processed_data
    
    def process_avg_over_samples(self, data, groupby=["sample", "water_strain"]):
        """
        This function calculates the average property relationship over time across all 
        samples, treating them as replicates of the same underlying system. This 
        approach assumes that differences between samples are primarily due to noise and not 
        due to systematic differences in the samples themselves. This could result in an 
        underestimation of the variability if there are systematic differences between samples. 
        The function returns the averaged parameterized data over time.
        """
        # Compute the mean of each group and reset the index
        processed_data = data.groupby(groupby).mean().reset_index()
        return processed_data
    
    def process_pooled_measurements(self, data):
        """
        This function pools all measurements of a property over time across all samples at a given 
        parameter level, treating each measurement as a separate observation. It assumes 
        that the measurements across different samples are equivalent for a 
        given parameter. However, this might not hold true if the different samples 
        show unique characteristics at specific parameter levels. The function returns 
        parameterized data with pooled measurements for each parameter level.
        """
        # Just return the data as is, no further processing required
        return data
    
    def fit_linear(self, processed_data, x_name, y_name):
        # Define the linear function
        def linear_func(x, a, b):
            return a * x + b

        if isinstance(processed_data, dict):
            # If data is a dictionary of dataframes, apply fit to each one
            parameters = {}
            for name, data in processed_data.items():
                popt, _ = curve_fit(linear_func, data[x_name], data[y_name])
                parameters[name] = popt
        else:
            # Otherwise apply fit to whole dataframe
            popt, _ = curve_fit(linear_func, processed_data[x_name], processed_data[y_name])
            parameters = popt

        return parameters
    
    def fit_poly(self, processed_data, x_name, y_name, order=2):
        # Define the polynomial function
        def poly_func(x, *coeffs):
            return np.polyval(coeffs, x)

        if isinstance(processed_data, dict):
            # If data is a dictionary of dataframes, apply fit to each one
            parameters = {}
            for name, data in processed_data.items():
                popt, _ = curve_fit(poly_func, data[x_name], data[y_name], p0=[0]*order)
                parameters[name] = popt
        else:
            # Otherwise apply fit to whole dataframe
            popt, _ = curve_fit(poly_func, processed_data[x_name], processed_data[y_name], p0=[0]*order)
            parameters = popt

        return parameters
    
    def fit_max(self, processed_data, y_name):
        # Define the max function
        def max_func(x):
            return np.max(x)

        if isinstance(processed_data, dict):
            # If data is a dictionary of dataframes, apply max to each one
            maximums = {}
            for name, data in processed_data.items():
                max_val = max_func(data[y_name])
                maximums[name] = max_val
        else:
            # Otherwise apply max to whole dataframe
            max_val = max_func(processed_data[y_name])
            maximums = max_val

        return maximums
    
    def compute_youngs_modulus(self, data, processing_type, fitting_type, x_name, y_name, groupby=["sample", "water_strain"]):
        if processing_type == 'individual':
            processed_data = self.process_independent_samples(data, groupby=groupby)
        elif processing_type == 'average':
            processed_data = self.process_avg_over_samples(data, groupby=groupby)
        elif processing_type == 'pooled':
            processed_data = self.process_pooled_measurements(data)
        if fitting_type == 'linear':
            params = self.fit_linear(processed_data, x_name, y_name,)
        elif fitting_type == 'poly':
            params = self.fit_poly(processed_data, x_name, y_name,)
        elif fitting_type == 'max':
            params = self.fit_max(processed_data, y_name)
        return params


    def plot_energy(self, temp_discard_perc=None):
        """
        Plots the energy data extracted from each .edr file.

        This method will read the energy.xvg file generated by `run_gmx_energy()` and plot the data it contains. 
        The first X% of the data is discarded to ensure good plot scaling.
        
        Parameters
        ----------
        temp_discard_perc : float
            Plot with different discard percentage. Do not change value with which object was initialized.
        """
        
        if not temp_discard_perc == None:
            discard_perc = temp_discard_perc
        else:
            discard_perc = self.discard_perc
        
        if len(self.energy_data) == 0:
                print("plot_energy(): No energy_data dataframe found, extracting data...")
                self.extract_data()
        
        for edr_file in self.edr_files:
            for term in self.terms:
                if (edr_file, term) not in self.energy_data:
                    original_terms = self.terms
                    print("plot_energy(): No energy_data dataframe found, trying to extract data...")
                    self.extract_data([term])
                    self.terms = original_terms
                dir_path, filename = os.path.split(edr_file)
                data = self.energy_data[(edr_file, term)]
                data_col_name = self.get_data_col_name(term)
                
                plt.figure(figsize=(10, 6))
                label = term
                length = len(data)
                idx_discard = int(round(length * discard_perc))
                print("Number of trajectory energy steps: " + str(length) + " steps discarded: " + str(idx_discard))

                plt.plot(data["Time (ps)"][idx_discard:], data[data_col_name][idx_discard:], label=f"{label}")
                
                plt.xlabel("Time (ps)")
                plt.ylabel(self.term_to_col_name[term])
                plt.legend()
                plt.title(f"Energy for {filename}\nExcluding the first {discard_perc*100}% trajectory")
                plt.show()
