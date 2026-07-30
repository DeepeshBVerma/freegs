"""
ITERDB filer reader for FreeGS bootstrap current calculations.

Parses ITERDB format files and provides interpolated profile functions.
"""
import numpy as np
import re
from scipy.interpolate import interp1d

class ITERDBReader:
    """
    Reader temperature and desnity profiules from ITERDB format files.

    ITERDB format:
    -Header line with identifier, description, units, number of points
    -Separate sectionms for each quantity (TE,TI,NE,NM1,NM2,VROT)
    """
    def __init__(self,filename):
        """
        Load and parse ITERDB file.

        Parameters
        __________
        filename: str
            Path to ITERDB file

        Raises
        __________
        FileNotFoundError: Flag if file does not exist.
        """
        self.filename = filename
        self.profiles = {}

        try:
            with open(filename,'r') as f:
                data_in = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"ITERDB file not fond: {filename}")

        #Split into lines for parsing
        data_linesplit = data_in.split('\n')
        
        #Find number of points
        num_points = self._find_num_points(data_linesplit)
        print(f"Number of points in ITERDB file: {num_points}")

        #Parse all available profiles
        self._parse_profiles(data_linesplit,num_points)

        print(f"Successfully loaded profiles: {list(self.profiles.keys())}")

    def _find_num_points(self, data_linesplit):
        """
        Find the number of data points from the fike header.
        
        Parameters
        __________
        data_linesplit: list of file lines

        Returns
        __________
        int Number of data points
        """
        for i,line in enumerate(data_linesplit):
            if re.search(';-# OF X PTS',line):
                num = int(line.split()[0])
                return num

        raise ValueError("Could not find number of points in ITERDB file")

    def _parse_profiles(self,data_linesplit,num_points):
        """
        Parse all profiles from ITERDB file.
           
        Parameters:
        __________
        data_linesplit: list of file lines

        num_points: Number of data points per profile
        """
         # Calculate number of lines needed for this many points (6 values per line)
        sec_num_lines = num_points // 6
        if num_points % 6 != 0:
            sec_num_lines += 1
        
        lnum = 0
        while len(data_linesplit) - lnum > 10:
            # Find next profile section
            keep_going = True
            quantity = None
            units = None
            
            while keep_going and lnum < len(data_linesplit):
                # Look for profile identifier
                test = re.search('-DEPENDENT VARIABLE LABEL', data_linesplit[lnum])
                if test:
                    parts = data_linesplit[lnum].split()
                    quantity = parts[0]
                    units = parts[1] if len(parts) > 1 else "unknown"
                
                # Look for data start marker
                test2 = re.search('DATA FOLLOW', data_linesplit[lnum])
                if test2:
                    keep_going = False
                
                lnum += 1
            
            if quantity is None:
                break
            
            # Parse the identified profile
            rhot, profile_data = self._read_profile_data(
                data_linesplit, lnum, num_points, sec_num_lines
            )
            
            # Store profile
            self.profiles[quantity] = {
                'rhot': rhot,
                'data': profile_data,
                'units': units,
                'quantity': quantity
            }
            
            print(f"Read {quantity} ({units}): {len(profile_data)} points")
            
            lnum += 2 * sec_num_lines + 1

    def _read_profile_data(self, data_linesplit, lnum, num_points, sec_num_lines):
        """
        Read a single profile's independent and dependent variables.
        
        Parameters
        ----------
        data_linesplit : list
            List of file lines
        lnum : int
            Current line number to start reading from
        num_points : int
            Expected number of data points
        sec_num_lines : int
            Number of lines the data occupies
            
        Returns
        -------
        tuple
            (independent_var_array, dependent_var_array)
        """
        # Read independent variable (e.g., rho_t)
        indep_var = np.empty(0)
        lnum0 = lnum
        for j in range(int(lnum0), min(int(lnum0 + sec_num_lines), len(data_linesplit))):
            for k in range(6):
                # Extract fixed-width column: offset 1, width 13 per column
                str_temp = data_linesplit[j][1 + k*13 : 1 + (k+1)*13]
                if re.search('e', str_temp):
                    try:
                        temp = float(str_temp)
                        indep_var = np.append(indep_var, temp)
                    except ValueError:
                        pass
        
        lnum += sec_num_lines + 1
        
        # Read dependent variable (actual profile data)
        dep_var = np.empty(0)
        lnum0 = lnum
        for j in range(int(lnum0), min(int(lnum0 + sec_num_lines), len(data_linesplit))):
            for k in range(6):
                str_temp = data_linesplit[j][1 + k*13 : 1 + (k+1)*13]
                if re.search('e', str_temp):
                    try:
                        temp = float(str_temp)
                        dep_var = np.append(dep_var, temp)
                    except ValueError:
                        pass
        
        return indep_var, dep_var
    def get_Te(self,psi_norm = None):
        """
        Get electron temperature profile as interpolated function.
        
        Parameters
        ----------
        psi_norm : array, optional
            Normalized poloidal flux values for evaluation. 
            If None, interpolation function is returned.
            
        Returns
        -------
        callable or array
            If psi_norm is None: returns callable f(psi_norm) in eV
            If psi_norm is array: returns interpolated values in eV
            
        Raises
        ------
        ValueError
            If TE profile not found in ITERDB file
        """
        if 'TE' not in self.profiles:
            raise ValueError("TE profile not found in ITERDB file")
        
        profile = self.profiles['TE']
        rhot = profile['rhot']
        te_data = profile['data']
        
        # ITERDB typically has TE in eV already
        # If in keV, multiply by 1000
        if profile['units'] == 'keV':
            te_data = te_data * 1000
        
        # Create interpolation function
        f_te = interp1d(rhot, te_data, kind='cubic', 
                        bounds_error=False, fill_value='extrapolate')
        
        if psi_norm is None:
            return f_te
        else:
            return f_te(psi_norm)
        
    def get_Ti(self, psi_norm=None):
        """
        Get ion temperature profile as interpolated function.
        
        Parameters
        ----------
        psi_norm : array, optional
            Normalized poloidal flux values for evaluation.
            
        Returns
        -------
        callable or array
            If psi_norm is None: returns callable f(psi_norm) in eV
            If psi_norm is array: returns interpolated values in eV
            Returns None if TI profile not available.
        """
        if 'TI' not in self.profiles:
            print("TI profile not found, will use TE for ions")
            return None
        
        profile = self.profiles['TI']
        rhot = profile['rhot']
        ti_data = profile['data']
        
        if profile['units'] == 'keV':
            ti_data = ti_data * 1000
        
        f_ti = interp1d(rhot, ti_data, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
        
        if psi_norm is None:
            return f_ti
        else:
            return f_ti(psi_norm)

    def get_ne(self, psi_norm=None):
        """
        Get electron density profile as interpolated function.
        
        Parameters
        ----------
        psi_norm : array, optional
            Normalized poloidal flux values for evaluation.
            
        Returns
        -------
        callable or array
            If psi_norm is None: returns callable f(psi_norm) in m^-3
            If psi_norm is array: returns interpolated values in m^-3
            
        Raises
        ------
        ValueError
            If NE profile not found in ITERDB file
        """
        if 'NE' not in self.profiles:
            raise ValueError("NE profile not found in ITERDB file")
        
        profile = self.profiles['NE']
        rhot = profile['rhot']
        ne_data = profile['data']
        
        # Convert if in units of 10^19 m^-3
        if '1E19' in profile['units'] or '10^19' in profile['units']:
            ne_data = ne_data * 1e19
        
        f_ne = interp1d(rhot, ne_data, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
        
        if psi_norm is None:
            return f_ne
        else:
            return f_ne(psi_norm)
    
    def get_ni(self, psi_norm=None):
        """
        Get ion density profile (NM1) as interpolated function.
        
        Parameters
        ----------
        psi_norm : array, optional
            Normalized poloidal flux values for evaluation.
            
        Returns
        -------
        callable or array
            If psi_norm is None: returns callable f(psi_norm) in m^-3
            If psi_norm is array: returns interpolated values in m^-3
            Returns None if NM1 profile not available (will use ne for quasi-neutrality).
        """
        if 'NM1' not in self.profiles:
            print("NM1 profile not found, will use NE for ions (quasi-neutrality)")
            return None
        
        profile = self.profiles['NM1']
        rhot = profile['rhot']
        ni_data = profile['data']
        
        if '1E19' in profile['units'] or '10^19' in profile['units']:
            ni_data = ni_data * 1e19
        
        f_ni = interp1d(rhot, ni_data, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
        
        if psi_norm is None:
            return f_ni
        else:
            return f_ni(psi_norm)
    
    def get_profile(self, quantity, psi_norm=None):
        """
        Get any profile by quantity identifier.
        
        Parameters
        ----------
        quantity : str
            Profile identifier (e.g., 'TE', 'TI', 'NE', 'NM1', 'NM2', 'VROT')
        psi_norm : array, optional
            Normalized poloidal flux values for evaluation.
            
        Returns
        -------
        callable or array
            If psi_norm is None: returns callable f(psi_norm)
            If psi_norm is array: returns interpolated values
            
        Raises
        ------
        ValueError
            If quantity not found in ITERDB file
        """
        if quantity not in self.profiles:
            raise ValueError(
                f"Profile '{quantity}' not found. "
                f"Available: {self.list_profiles()}"
            )
        
        profile = self.profiles[quantity]
        rhot = profile['rhot']
        data = profile['data']
        
        f_profile = interp1d(rhot, data, kind='cubic',
                            bounds_error=False, fill_value='extrapolate')
        
        if psi_norm is None:
            return f_profile
        else:
            return f_profile(psi_norm)
    
    def list_profiles(self):
        """
        List all available profiles in the ITERDB file.
        
        Returns
        -------
        list
            List of profile identifiers
        """
        return list(self.profiles.keys())
    
    def get_profile_info(self, quantity):
        """
        Get metadata for a profile.
        
        Parameters
        ----------
        quantity : str
            Profile identifier
            
        Returns
        -------
        dict
            Dictionary with keys: 'quantity', 'units', 'num_points', 'rhot_range'
        """
        if quantity not in self.profiles:
            raise ValueError(f"Profile '{quantity}' not found")
        
        profile = self.profiles[quantity]
        rhot = profile['rhot']
        
        return {
            'quantity': quantity,
            'units': profile['units'],
            'num_points': len(rhot),
            'rhot_range': (rhot.min(), rhot.max()),
            'rhot': rhot,
            'data_range': (profile['data'].min(), profile['data'].max())
        }

