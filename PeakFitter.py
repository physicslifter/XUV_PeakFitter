'''
Classes and functions for fitting XUV peaks
'''
import os
import pandas as pd
import tifffile as tiff
import numpy as np
from scipy.signal import find_peaks

class LinearCalibration:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def convert_numpixels_to_wavelength(self, num_pixels):
        return num_pixels*self.m

    def convert_wavelength_to_numpixels(self, wavelength):
        return wavelength/self.m

class XUVImage:
    def __init__(self, fname):
        self.has_wavelengths = False
        self.has_lineout = False
        self.fname = fname
        self.get_data()
        self.flip_image()
        self.zero_data()

    def get_data(self):
        try:
            assert(os.path.exists(self.fname) == True) #check whether the path exists
            ext = self.fname.split(".")[-1]
            if ext.lower() in ["tif", "tiff"]: #handling for tiff files
                self.img = tiff.imread(self.fname)
            else: #if not a Tif file
                assert(False == True) #throw an assertion error
        except:
            raise Exception(f"{self.fname} is an invalid file")
        
    def flip_image(self):
        '''
        flips the image on the x-axis
        '''
        self.img = self.img[:, ::-1]

    def zero_data(self):
        '''
        Sets the minimum of the data to 0
        '''
        self.img = self.img - self.img.min()

    def take_lineout(self):
        '''
        Gets the lineout from the img
        '''
        self.lineout = np.mean(self.img, axis = 0)
        self.pixels = np.arange(len(self.lineout))
        self.has_lineout = True

    def apply_linear_calibration(self, calibration:LinearCalibration):
        '''
        Given a calibration: lambda = m*pixel + b,
        this gives the wavelengths as a function of pixel
        '''
        if self.has_lineout == True:
            self.wavelengths = self.pixels*calibration.m + calibration.b
        self.has_wavelengths = True

    def plot_lineout(self, ax):
        x_data = self.pixels if self.has_wavelengths == False else self.wavelengths
        ax.plot(x_data, self.lineout, label = "Lineout", c = "dodgerblue", linewidth = 2)

    def plot_img(self, ax):
        ax.imshow(self.img)
        if self.has_wavelengths == True:
            interval = int(len(self.pixels)/5)
            ax.set_xticks(np.arange(len(self.wavelengths))[::interval])
            ax.set_xticklabels(np.round(self.wavelengths[::interval], 1))

    def find_peaks(self, threshold=0.0, width=1, index_bounds = None):
        if type(index_bounds) == type(None):
            lineout_data = self.lineout
            x_data = self.wavelengths if self.has_wavelengths == True else self.pixels
        else:
            lineout_data = self.lineout[index_bounds[0]:index_bounds[1]]
            x_data = self.wavelengths[index_bounds[0]:index_bounds[1]] if self.has_wavelengths == True else self.pixels[index_bounds[0]:index_bounds[1]]
        # Make sure we actually have a lineout
        if not getattr(self, "has_lineout", False):
            raise RuntimeError("Lineout has not been computed. Call take_lineout() first.")

        # Use scipy's robust peak finder
        peaks_idx, properties = find_peaks(
            lineout_data,
            height=threshold,   # amplitude threshold
            width=width         # minimum peak width in pixels
        )
        peak_x = x_data[peaks_idx]
        intensity_x = lineout_data[peaks_idx]
        print(peak_x, intensity_x)
        return peaks_idx, peak_x, intensity_x, properties

            
