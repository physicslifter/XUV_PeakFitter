from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, Button, CheckButtons, TextBox
import matplotlib.patches as patches
from PeakFitter import *
from scipy.optimize import curve_fit
from scipy.special import wofz
from pdb import set_trace as st
from adjustText import adjust_text


def add_row(table, row_values):
    '''
    Helper function to add a row to a table
    '''
    celld = table.get_celld()
    rows = [r for (r, c) in celld.keys() if r >= 0 and c >= 0]
    cols = [c for (r, c) in celld.keys() if r >= 0 and c >= 0]
    new_row_idx = max(rows) + 1
    ncols = max(cols) + 1
    ref_cell = celld[(rows[0], cols[0])]
    w, h = ref_cell.get_width(), ref_cell.get_height()
    for col_idx in range(ncols):
        text = str(row_values[col_idx]) if col_idx < len(row_values) else ""
        table.add_cell(new_row_idx, col_idx, width=w, height=h, text=text)
    table.axes.figure.canvas.draw_idle()

def update_row(table, row_idx, row_values):
    '''
    Helper function to update a row in a table
    '''
    for col_idx, value in enumerate(row_values):
        cell = table[row_idx, col_idx]
        cell.get_text().set_text(str(value))

def voigt(x, amp, x0, sigma, gamma, offset):
    '''
    Voigt profile
    '''
    z = ((x - x0) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi)) + offset


class XUVPeak:
    def __init__(self, xdata, ydata, background):
        self.xdata = xdata
        self.ydata = ydata
        self.background = background
        self.has_width = False

    def fit_peak(self, width = None):
        '''
        Fits peak to a Voigt profile
        '''
        if type(width) != type(None):
            self.has_width = True
            self.width = width
        amp_guess = np.max(self.ydata) - np.min(self.ydata)
        x0_guess = self.xdata[np.argmax(self.ydata)]
        sigma_guess = (self.xdata[-1] - self.xdata[0])/50
        gamma_guess = sigma_guess
        offset_guess = np.min(self.ydata)
        p0 = [amp_guess, x0_guess, sigma_guess, gamma_guess]
        def my_voigt(x_, amp_, x0_, sigma_, gamma_):
            return voigt(x_, amp_, x0_, sigma_, gamma_, self.background)
        popt, pcov = curve_fit(my_voigt, self.xdata, self.ydata, p0=p0)
        amp, x0, sigma, gamma = popt
        self.amp, self.x0, self.sigma, self.gamma = amp, x0, sigma, gamma
        self.fit_line = voigt(self.xdata, amp, x0, sigma, gamma, self.background)

    def show_fit(self, ax = None):
        has_ax = True
        if ax == None:
            fig = plt.figure(figsize = (8, 8))
            ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(bottom = 0.2)
            ax.set_xlabel(r"$\lambda$ (nm)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"Peak Fit @ {np.round(self.x0, 3)}")
            has_ax = False
        ax.plot(self.xdata, self.ydata, label = "Raw Data", c = "blue")
        ax.plot(self.xdata, self.fit_line, label = "Voigt Fit", c = "magenta")
        #get area under curve
        if self.has_width == False:
            area = np.trapz(self.fit_line, self.xdata)
            x_data = self.xdata
            fit_data = self.fit_line
            min_loc = 0
            max_loc = -1
        else:
            #get area w/in width & show
            peak_loc = np.argmin(np.abs(self.xdata - self.x0))
            min_loc = peak_loc - int(self.width/2)
            max_loc = peak_loc + int(self.width/2)
            x_data = self.xdata[min_loc:max_loc]
            fit_data = self.fit_line[min_loc:max_loc]
            area = np.trapz(fit_data, x_data)
            #show lines
            #for val in [self.xdata[min_loc], self.xdata[max_loc], self.x0]:
            #    ax.axvline(x = val, c = "magenta")
        #min_wav = min_loc*pixel_per_wavelength + min(self.xdata)
        #max_wav = max_loc*pixel_per_wavelength + min(self.xdata)
        #min_wav = self.xdata[min_loc]
        #max_wav = self.xdata[max_loc]
        background_data = np.ones_like(fit_data)*self.background
        fill_between_xdata = self.xdata[min_loc:max_loc] if self.has_width == True else self.xdata 
        ax.fill_between(x = fill_between_xdata, y1 = background_data, y2 = fit_data, color  = "red", alpha = 0.4, label = "Area")
        #show values
        ax.legend()
        if has_ax == False:
            fig.text(0.5, 0.01, f"Integral: {area}\nAmp: {self.amp}\nCenter: {self.x0}\n$\sigma$: {self.sigma}\n$\gamma$: {self.gamma}\noffset: {self.background}")
        plt.show()

    def get_peak(self, new_xdata):
        #given a different x, this function returns the peak profile for this x
        return voigt(new_xdata, self.amp, self.x0, self.sigma, self.gamma, self.background)
    
class AlamgirPeak:
    def __init__(self, xdata, ydata, background, peak_x, peak_y, min_wavelength):
        self.xdata = xdata
        self.ydata = ydata
        self.background = background
        self.peak_x = peak_x
        self.peak_y = peak_y
        self.has_area = False
        self.min_wavelength = min_wavelength

    def get_area(self, width):
        areas = []
        self.width = width
        #print(self.width/2)
        for x in [-1, 0, 1]:
            bounds = [int(self.peak_x - width/2 + x), int(self.peak_x + width/2 + x)]
            for c, bound in enumerate(bounds):
                val = np.argmin(np.abs(bound - self.xdata))
                bounds[c] = val
                print(bounds)
            #print(bounds)
            intensities = self.ydata[bounds[0]:bounds[1]]
            areas.append(np.sum(intensities))
        self.has_area = True
        self.area = np.mean(areas)
        return self.area
    
    def show_fit(self, ax=None, wavelengths=None, color="red"):
        if self.has_area == False:
            raise Exception("Peak does not have area")
        if type(ax) == type(None):
            fig = plt.figure(figsize = (8, 8))
            ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(bottom = 0.2)
            input_ax = False
        else:
            input_ax = True
        for x in [-1, 0, 1]:
            if x == -1 and input_ax == False:
                data_bounds = [int(self.peak_x - self.width/2 + x - self.width*2), int(self.peak_x + self.width/2 + x + self.width*2)]
                for c, bound in enumerate(data_bounds):
                    val = np.argmin(np.abs(bound - self.xdata))
                    data_bounds[c] = val
                xdata = self.xdata[data_bounds[0]:data_bounds[1]]
                intensities = self.ydata[data_bounds[0]:data_bounds[1]]
                if type(wavelengths) == type(None):
                    ax.plot(xdata, intensities, label = "Raw Data", c = "blue")
            bounds = [int(self.peak_x - self.width/2 + x), int(self.peak_x + self.width/2 + x)]
            for c, bound in enumerate(bounds):
                val = np.argmin(np.abs(bound - self.xdata))
                bounds[c] = val
                #print(bounds)
            print(self.xdata[bounds[0]], bounds)
            region_bound_intensity = self.ydata[bounds[0]:bounds[1]]
            region_bound_background = np.ones_like(region_bound_intensity)*self.background
            region_x = self.xdata[bounds[0]:bounds[1]]
            label = "Integral Regions" if x == -1 else None
            label = label if input_ax == False else None
            if type(wavelengths) != type(None):
                min_wav_loc = np.argmin(np.abs(self.min_wavelength - wavelengths))
                for c, bound in enumerate(bounds):
                    bounds[c] = bound + min_wav_loc
                #print(bounds, len(wavelengths))
                region_x = wavelengths[bounds[0]:bounds[1]]
            #print(len(region_x), len(region_bound_background), len(region_bound_intensity))
            ax.fill_between(x = region_x, y1 = region_bound_background, y2 = region_bound_intensity, color = color, alpha = 0.2, label = label)
        if input_ax == False:
            fig.text(0.5, 0.01, f"Integral: {self.area}\nAmp: {self.peak_y}\nCenter: {self.peak_x}\noffset: {self.background}")
            ax.set_xlabel(r"$\lambda$ (nm)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"Peak Fit @ {np.round(self.peak_x, 3)}")
            ax.legend()
            #print(data_bounds)
            #print(xdata)
            #print(self.xdata)
            plt.show()

def get_alamgir_peak(img, data_bounds, background):
    #given some data, return an Alamgir peak with this data
    peak_locs, x_peaks, intensities, peak_props = img.find_peaks(threshold = background, width = 2, index_bounds = data_bounds)
    return AlamgirPeak(xdata = img.pixels[data_bounds[0]:data_bounds[1]],
                       ydata = img.lineout[data_bounds[0]:data_bounds[1]],
                       background = background,
                       peak_x = x_peaks[0],
                       peak_y = intensities[0],
                       min_wavelength = min(img.wavelengths)
                       )

class Plotter:
    def __init__(self, fname, calibration:LinearCalibration):
        self.fname = fname
        self.calibration = calibration
        self.has_fit = False #set has_fit to false
        self.has_auto_fit = False
        self.peaks_found = False #boolean control for auto peak finder
        self.num_peaks_measured = 0 #initialize w/ 0 measured peaks
        self.peak_colors = ["magenta", "orange", "lime", "gold", "mediumspringgreen"]
        self.measured_peak_lines = []
        self.measurement_labels = []
        self.auto_peaks = {}
        self.img = XUVImage(fname)
        self.img.take_lineout()
        self.img.apply_linear_calibration(calibration)
        self.setup_plot()
        self.initialize_plot()
        self.show()

    def setup_plot(self):
        self.fig = plt.figure(figsize = (15, 8))
        self.lineout_ax = self.fig.add_axes([0.05, 0.4, 0.55, 0.55])
        self.lineout_ax.set_title(f"{self.fname.split('/')[-1]}")
        self.lineout_ax.set_xlabel(r"$\lambda$ $(nm)$")
        self.lineout_ax.set_ylabel(r"$Intensity$ $(a.u.)$")
        self.setup_table()
        self.setup_manual_sliders()
        self.setup_manual_section()
        self.setup_auto_section()
        self.setup_measurement_section()
        self.setup_labeler_section()
        self.setup_finalization_section()

    def setup_manual_sliders(self):
        outline = patches.Rectangle((0.02, 0.05), 0.58, 0.2, fill = True, facecolor = "mintcream", edgecolor = "k", figure = self.fig, zorder = 0)
        self.fig.add_artist(outline)
        self.fig.text(0.3, 0.22, "Manual Locator", ha = "center", va = "center", fontsize = 14, fontweight = "bold")

        self.x_zoom_slider_ax = self.fig.add_axes([0.1, 0.15, 0.42, 0.05])
        self.x_zoom_slider = RangeSlider(self.x_zoom_slider_ax, "X range", min(self.img.wavelengths), max(self.img.wavelengths), valinit = (min(self.img.wavelengths), max(self.img.wavelengths)))

        self.peak_loc_slider_ax = self.fig.add_axes([0.1, 0.1, 0.42, 0.05])
        self.peak_loc_slider = RangeSlider(self.peak_loc_slider_ax, "Peak\nLoc", min(self.img.wavelengths), max(self.img.wavelengths), valinit = (min(self.img.wavelengths), max(self.img.wavelengths)))

        self.background_slider_ax = self.fig.add_axes([0.1, 0.05, 0.42, 0.05])
        self.background_slider = Slider(self.background_slider_ax, "Background", valmin = -max(self.img.lineout), valmax = max(self.img.lineout), valinit=0)

    def setup_table(self):
        table_ax = self.fig.add_axes([0.62, 0.35, 0.35, 0.5])
        table_ax.set_axis_off()
        table_ax.set_title("Measured Peaks")
        self.peak_table = table_ax.table(cellText = [["", "", "", "", "", ""]],
                                    rowLabels = [""],
                                    colLabels = ["#", "Label", "Loc", "Amp", "Area", "Measurement\nMethod"],
                                    loc = "center"
                                    )
        
        for (row, col_idx), cell in self.peak_table.get_celld().items():
            height = 0.14 if row == 0 else 0.07
            cell.set_height(height)
            width = 0.15 if col_idx in [0, 2, 3, 4] else 0.2
            width = 0.08 if col_idx == 0 else width
            cell.set_width(width)
        self.peak_table.set_fontsize(14)

    def setup_manual_section(self):
        #outline section
        outline = patches.Rectangle((0.62, 0.05), 0.11, 0.3, fill = True, facecolor = "lightcyan", edgecolor = "k", figure = self.fig, zorder = 0)
        self.fig.add_artist(outline)
        self.fig.text(0.675, 0.31, "Manual\nMeasure", ha = "center", va = "center", fontsize = 14, fontweight = "bold")

        #checkboxes for fit type
        fit_types = ["voight"]
        initial_states = [True]
        check_ax = self.fig.add_axes([0.66, 0.17, 0.11, 0.16])
        check_ax.set_axis_off()
        self.fit_check = CheckButtons(check_ax, fit_types, initial_states)

        #add slider for integral width
        integral_slider_ax = self.fig.add_axes([0.63, 0.08, 0.02, 0.15])
        self.integral_slider = Slider(integral_slider_ax, "int\nWidth", valmin = 0, valmax = 30, valinit = 0, orientation = "vertical", valstep = 1)

        #buttons
        fit_button_ax = self.fig.add_axes([0.66, 0.13, 0.05, 0.05])
        self.fit_button = Button(fit_button_ax, "Fit")

        peak_button_ax = self.fig.add_axes([0.66, 0.07, 0.05, 0.05])
        self.peak_button = Button(peak_button_ax, "Add Peak")
    
    def setup_auto_section(self):
        outline = patches.Rectangle((0.74, 0.2), 0.16, 0.15, fill = True, facecolor = "bisque", edgecolor = "k", figure = self.fig, zorder = 0)
        self.fig.add_artist(outline)
        self.fig.text(0.82, 0.33, "Auto Locator", ha = "center", va = "center", fontsize = 14, fontweight = "bold")

        find_peak_button_ax = self.fig.add_axes([0.77, 0.21, 0.07, 0.03])
        self.find_peak_button = Button(find_peak_button_ax, "Find Peaks")

        auto_background_slider_ax = self.fig.add_axes([0.79, 0.25, 0.08, 0.03])
        self.auto_background_slider = Slider(auto_background_slider_ax, "Threshold", valmin = 0, valmax = max(self.img.lineout), valinit = 0)
        auto_width_slider_ax = self.fig.add_axes([0.79, 0.28, 0.08, 0.03])
        self.auto_width_slider = Slider(auto_width_slider_ax, "Width ", valmin = 1, valmax = 30, valinit = 1, valstep = 1)

    def setup_measurement_section(self):
        outline = patches.Rectangle((0.74, 0.05), 0.16, 0.14, fill = True, facecolor = "lavender", alpha = 1, edgecolor = "k", figure = self.fig, zorder = 0)
        self.fig.add_artist(outline)
        self.fig.text(0.82, 0.17, "Auto Measure", ha = "center", va = "center", fontsize = 14, fontweight = "bold")

        measurement_width_slider_ax = self.fig.add_axes([0.78, 0.1, 0.11, 0.03])
        self.measurement_width_slider = Slider(measurement_width_slider_ax, "Width ", valmin = 0, valmax = 30, valinit = 0, valstep = 1)
        peak_num_slider_ax = self.fig.add_axes([0.78, 0.13, 0.11, 0.03])
        self.peak_num_slider = Slider(peak_num_slider_ax, "Peak # ", valmin = 0, valmax = 30, valinit = 1, valstep = 1)

        #checkbox
        self.fig.text(0.745, 0.07, "Method: ")
        #checkboxes for fit type
        fit_types = ["voight", "alamgir"]
        initial_states = [True, False]
        check_ax = self.fig.add_axes([0.78, 0.05, 0.06, 0.05])
        check_ax.set_axis_off()
        self.auto_fit_check = CheckButtons(check_ax, fit_types, initial_states)

        #buttons
        fit_button_ax = self.fig.add_axes([0.84, 0.06, 0.025, 0.03])
        self.auto_fit_button = Button(fit_button_ax, "Fit")
        add_button_ax = self.fig.add_axes([0.87, 0.06, 0.025, 0.03])
        self.auto_add_button = Button(add_button_ax, "Add")

    def setup_labeler_section(self):
        outline = patches.Rectangle((0.02, 0.26), 0.58, 0.07, fill = True, facecolor = "palegreen", edgecolor = "k", figure = self.fig, zorder = 0)
        self.fig.add_artist(outline)
        self.fig.text(0.06, 0.295, "Labeling", ha = "center", va = "center", fontsize = 14, fontweight = "bold")

        self.label_slider_ax = self.fig.add_axes([0.14, 0.27, 0.22, 0.05])
        self.label_slider = Slider(ax  = self.label_slider_ax, label = "Peak no.", valmin = 0, valmax = 0, valinit = 0, valstep = 1)

        label_entry_ax = self.fig.add_axes([0.42, 0.27, 0.1, 0.05])
        self.label_entry = TextBox(label_entry_ax, "| Label ", initial = "0")

        update_label_ax = self.fig.add_axes([0.53, 0.27, 0.05, 0.05])
        self.update_label_button = Button(update_label_ax, "Update\nLabel", color = "salmon")

    def setup_finalization_section(self):
        finalize_button_ax = self.fig.add_axes([0.92, 0.06, 0.05, 0.08])
        self.finalize_button = Button(finalize_button_ax, "Finalize\nPlot", color = "crimson")
        self.finalize_button.label.set_color("azure")
        self.finalize_button.label.set_weight("bold")

    def initialize_plot(self):
        self.img.plot_lineout(self.lineout_ax)
        self.peak_min = self.lineout_ax.axvline(x = self.peak_loc_slider.val[0], c = "red", label = "Peak Fit Bounds")
        self.peak_max = self.lineout_ax.axvline(x = self.peak_loc_slider.val[1], c = "red")
        self.background_line = self.lineout_ax.axhline(y = 0, c = "green", label = "Background")
        self.lineout_ax.legend()

    def update_xrange_slider(self, val):
        self.lineout_ax.set_xlim(val[0], val[1])
        #reset the peak locations if they are out of the current x range
        peak_max = self.peak_loc_slider.val[1] if self.peak_loc_slider.val[1] < val[1] else val[1]
        peak_min = self.peak_loc_slider.val[0] if self.peak_loc_slider.val[0] > val[0] else val[0]
        self.peak_loc_slider.valmin = val[0]
        self.peak_loc_slider.valmax = val[1]
        self.peak_loc_slider_ax.set_xlim(val[0], val[1])
        self.peak_loc_slider.set_val((peak_min, peak_max))
        self.peak_min.set_xdata([peak_min])
        self.peak_max.set_xdata([peak_max])

    def auto_find_peaks(self, val):
        '''
        automatically find the peaks
        '''
        #remove any current peaks
        if self.peaks_found == True:
            for label in self.auto_peak_labels:
                label.remove()
        self.auto_peak_labels = []
        min_bound = np.argmin(np.abs(self.img.wavelengths - self.peak_loc_slider.val[0]))
        max_bound = np.argmin(np.abs(self.img.wavelengths - self.peak_loc_slider.val[1]))
        peak_locs, x_peaks, intensities, peak_props = self.img.find_peaks(threshold = self.auto_background_slider.val, width = self.auto_width_slider.val, index_bounds = [min_bound, max_bound])
        #st()
        self.peaks = {"pixel_locs": peak_locs, "x_locs": x_peaks, "y_locs": intensities, "props": peak_props}
        for num, intensity, peak, width, prominence in zip(np.arange(len(x_peaks)), intensities, x_peaks, peak_props["widths"], peak_props["prominences"]):
            print(peak, intensity, width)
            label = self.lineout_ax.text(peak, intensity, f"{num}", c = "white", bbox=dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.3'))
            self.auto_peak_labels.append(label)
            #show peak width
            #width = self.calibration.convert_pixels_to_wavelength(width)
            #self.lineout_ax.plot([peak - width/2, peak + width/2], [(prominence)/2 + intensity - prominence, (prominence)/2 + intensity - prominence], c = "red")
        self.peaks_found = True
        self.fig.canvas.draw_idle()

    def update_peak_slider(self, val):
        for c, peak_line in enumerate([self.peak_min, self.peak_max]):
            peak_line.set_xdata([val[c]])

    def update_background_slider(self, val):
        self.background_line.set_ydata([val])

    def update_labeler_slider(self, val):
        self.label_entry.set_val(self.auto_peaks[val]["label"])
        self.fig.canvas.draw_idle()

    def click_fit(self, val):
        min_bound = np.argmin(np.abs(self.img.wavelengths - self.peak_loc_slider.val[0]))
        max_bound = np.argmin(np.abs(self.img.wavelengths - self.peak_loc_slider.val[1]))
        xdata = self.img.wavelengths[min_bound:max_bound]
        ydata = self.img.lineout[min_bound:max_bound]
        peak = XUVPeak(xdata, ydata, background = self.background_slider.val)
        width = self.integral_slider.val
        width = None if width == 0 else width
        print(f"width: {width}")
        peak.fit_peak(width = width)
        self.peak = peak
        self.has_fit = True
        peak.show_fit()

    def add_peak_to_table(self, area, method):
        '''
        helper function for click_add_peak
        adds all peak info to the peak dictionary
        which will be saved as csv
        '''
        vals = [self.num_peaks_measured, self.num_peaks_measured, np.round(self.peak.x0, 2), np.round(self.peak.amp, 2), np.round(area, 2), method]
        if self.num_peaks_measured > 0:
            #if this is not the first peak, add a blank row
            add_row(self.peak_table, ["", "", "", "", "", ""])
        update_row(self.peak_table, self.num_peaks_measured + 1, vals)

        #ensure cell dimensions
        for (row, col_idx), cell in self.peak_table.get_celld().items():
            cell.set_height(0.07)
            width = 0.15 if col_idx in [0, 2, 3, 4] else 0.2
            width = 0.08 if col_idx == 0 else width
            cell.set_width(width)

    def click_add_peak(self, val):
        print(self.has_fit)
        if self.has_fit == True: # only add if we currently have the fit
            full_peak = self.peak.get_peak(self.img.wavelengths)
            area = np.trapz(full_peak, self.img.wavelengths)
            peak_line = self.lineout_ax.plot(self.img.wavelengths, full_peak, linestyle = "--", linewidth = 1, c = self.peak_colors[self.num_peaks_measured], label = self.num_peaks_measured)
            self.measured_peak_lines.append(peak_line)
            self.add_peak_to_table(area, method = "Manual Voight")
            self.num_peaks_measured += 1
            self.lineout_ax.legend()
            self.fig.canvas.draw_idle()
            print("peak added")

    def click_auto_fit(self, val):
        '''
        Function to execute upon clicking fit button in Measure box:
            1. Get peak num
            2. get Fit type
                for Voight:
                    1. define fit area
                    2. Get voight peak and make fit, generate report
                    3. save peak in case we click add button
        '''
        #method = self.auto_fit_check.labels[self.auto_fit_check.get_status()]
        #print(method)
        labels = self.auto_fit_check.labels
        checks = self.auto_fit_check.get_status()
        method = labels[checks.index(True)].get_text()
        self.method = method
        print(method)
        peak_num = self.peak_num_slider.val
        peak_x = self.peaks["pixel_locs"][peak_num]
        peak_y = self.peaks["y_locs"][peak_num]
        peak_width = self.peaks["props"]["widths"][peak_num]
        min_loc = int(peak_x - peak_width*1.2)
        max_loc = int(peak_x + peak_width*1.2)
        xdata = self.img.pixels[min_loc:max_loc]
        ydata = self.img.lineout[min_loc:max_loc]
        self.has_auto_fit = True
        if method == "voight":
            peak = XUVPeak(xdata, ydata, background = self.background_slider.val)
            width = self.measurement_width_slider.val
            if width == 0:
                width = None
            peak.fit_peak(width = width)
            self.peak = peak
            self.has_fit = True
            self.auto_peak = peak
            peak.show_fit()
        elif method == "alamgir":
            peak = AlamgirPeak(xdata, ydata, background = self.background_slider.val, peak_x = peak_x, peak_y = peak_y, min_wavelength = self.img.wavelengths[min_loc])
            peak.x0 = self.img.wavelengths[peak_x]
            peak.amp = peak_y
            print(self.measurement_width_slider.val, type(self.measurement_width_slider.val))
            width = self.measurement_width_slider.val
            width = width + 1 if width%2 == 1 else width
            peak.get_area(width = width)
            #peak.area = self.calibration.convert_numpixels_to_wavelength(peak.area)
            self.auto_peak = peak
            self.peak = peak
            peak.show_fit()

    def click_auto_add(self, val):
        if self.has_auto_fit == True:
            if self.method == "voight":
                full_peak = self.auto_peak.get_peak(np.arange(len(self.img.wavelengths)))
                self.lineout_ax.plot(self.img.wavelengths, full_peak, c = "gold", linestyle = "--")
                if self.measurement_width_slider.val == 0:
                    self.lineout_ax.fill_between(self.img.wavelengths, np.ones_like(full_peak)*self.background_slider.val, full_peak, color = "gold", alpha = 0.1)
                    area = np.trapz(full_peak, self.img.wavelengths)
                else:
                    peak_loc = np.argmin(np.abs(self.peak.xdata - self.peak.x0))
                    min_loc = peak_loc - int(self.peak.width/2)
                    max_loc = peak_loc + int(self.peak.width/2)
                    x_data = self.peak.xdata[min_loc:max_loc]
                    fit_data = self.peak.fit_line[min_loc:max_loc]
                    area = np.trapz(fit_data, x_data)
                    background_data = np.ones_like(fit_data)*self.peak.background
                    fill_between_xdata = self.peak.xdata[min_loc:max_loc]*self.calibration.m + self.calibration.b
                    self.lineout_ax.fill_between(x = fill_between_xdata, y1 = background_data, y2 = fit_data, color  = "gold", alpha = 0.1)
                self.add_peak_to_table(area, method = "Auto Voight")
            elif self.method == "alamgir":
                self.peak.show_fit(ax = self.lineout_ax, wavelengths = self.img.wavelengths, color = "cyan")
                self.add_peak_to_table(self.peak.area, "Auto Alamgir")
            peak_num = self.peak_num_slider.val
            peak_x = self.peaks["pixel_locs"][peak_num]
            peak_y = self.peaks["y_locs"][peak_num]
            label = self.lineout_ax.text(peak_x, peak_y, f"{self.num_peaks_measured}", c = "white", bbox=dict(facecolor='green', edgecolor='black', boxstyle='round,pad=0.3'))
            self.measurement_labels.append(label)
            adjust_text(self.measurement_labels, ax = self.lineout_ax, only_move = "y")
            self.auto_peaks[self.num_peaks_measured] = {"loc": [peak_x, peak_y], "label": f"{self.num_peaks_measured}"}
            self.num_peaks_measured += 1

            #update the peak no. slider for labeling
            self.label_slider_ax.set_xlim(0, self.num_peaks_measured - 1)
            self.label_slider.valmin = 0
            self.label_slider.valmax = self.num_peaks_measured - 1

            self.lineout_ax.legend()
            self.fig.canvas.draw_idle()
            print("peak added")

    def click_update_label(self, val):
        label = self.measurement_labels[int(self.label_slider.val)]
        label.set_text(self.label_entry.text)
        self.auto_peaks[int(self.label_slider.val)]["label"] = self.label_entry.text
        for c, label in enumerate(self.measurement_labels):
            loc = self.auto_peaks[c]["loc"]
            label.set_position((loc[0], loc[1]))
        adjust_text(self.measurement_labels, ax = self.lineout_ax, only_move = "y")
        #update label in table
        self.peak_table[(self.label_slider.val + 1, 1)].get_text().set_text(self.label_entry.text)
        self.fig.canvas.draw_idle()

    def set_widgets(self):
        self.x_zoom_slider.on_changed(self.update_xrange_slider)
        self.peak_loc_slider.on_changed(self.update_peak_slider)
        self.background_slider.on_changed(self.update_background_slider)
        self.fit_button.on_clicked(self.click_fit)
        self.peak_button.on_clicked(self.click_add_peak)
        self.find_peak_button.on_clicked(self.auto_find_peaks)
        self.auto_fit_button.on_clicked(self.click_auto_fit)
        self.auto_add_button.on_clicked(self.click_auto_add)
        self.label_slider.on_changed(self.update_labeler_slider)
        self.update_label_button.on_clicked(self.click_update_label)

    def show(self):
        self.set_widgets()
        plt.show()