from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, Button
from PeakFitter import *
from scipy.optimize import curve_fit
from scipy.special import wofz


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

    def fit_peak(self):
        '''
        Fits peak to a Voigt profile
        '''
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

    def show_fit(self):
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(bottom = 0.2)
        ax.plot(self.xdata, self.ydata, label = "Raw Data", c = "blue")
        ax.plot(self.xdata, self.fit_line, label = "Voigt Fit", c = "magenta")
        ax.set_xlabel(r"$\lambda$ (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Peak Fit @ {np.round(self.x0, 3)}")
        ax.legend()
        #get area under curve
        area = np.trapz(self.fit_line, self.xdata)
        #show values
        fig.text(0.5, 0.01, f"Integral: {area}\nAmp: {self.amp}\nCenter: {self.x0}\n$\sigma$: {self.sigma}\n$\gamma$: {self.gamma}\noffset: {self.background}")
        plt.show()

    def get_peak(self, new_xdata):
        #given a different x, this function returns the peak profile for this x
        return voigt(new_xdata, self.amp, self.x0, self.sigma, self.gamma, self.background)

class Plotter:
    def __init__(self, fname, calibration:LinearCalibration):
        self.fname = fname
        self.has_fit = False #set has_fit to false
        self.num_peaks_measured = 0 #initialize w/ 0 measured peaks
        self.peak_colors = ["magenta", "orange", "lime", "gold", "mediumspringgreen"]
        self.measured_peak_lines = []
        self.img = XUVImage(fname)
        self.img.take_lineout()
        self.img.apply_linear_calibration(calibration)
        self.setup_plot()
        self.initialize_plot()
        self.show()

    def setup_plot(self):
        self.fig = plt.figure(figsize = (13, 8))
        self.lineout_ax = self.fig.add_axes([0.1, 0.25, 0.567, 0.65])
        #self.fig.subplots_adjust(bottom = 0.25)
        #table for measured peaks
        self.lineout_ax.set_title(f"{self.fname.split('/')[-1]}")
        self.lineout_ax.set_xlabel(r"$\lambda$ $(nm)$")
        self.lineout_ax.set_ylabel(r"$Intensity$ $(a.u.)$")
        table_ax = self.fig.add_axes([0.7, 0.25, 0.25, 0.65])
        table_ax.set_axis_off()
        table_ax.set_title("Measured Peaks")
        self.peak_table = table_ax.table(cellText = [["", "", "", ""]],
                                    rowLabels = [""],
                                    colLabels = ["#", "Loc", "Amp", "Area"],
                                    loc = "center"
                                    )
        for key, cell in self.peak_table.get_celld().items():
            cell.set_height(0.1)

        #sliders
        self.x_zoom_slider_ax = self.fig.add_axes([0.1, 0.15, 0.65, 0.05])
        self.x_zoom_slider = RangeSlider(self.x_zoom_slider_ax, "X range", min(self.img.wavelengths), max(self.img.wavelengths), valinit = (min(self.img.wavelengths), max(self.img.wavelengths)))

        self.peak_loc_slider_ax = self.fig.add_axes([0.1, 0.1, 0.65, 0.05])
        self.peak_loc_slider = RangeSlider(self.peak_loc_slider_ax, "Peak\nLoc", min(self.img.wavelengths), max(self.img.wavelengths), valinit = (min(self.img.wavelengths), max(self.img.wavelengths)))

        self.background_slider_ax = self.fig.add_axes([0.1, 0.05, 0.65, 0.05])
        self.background_slider = Slider(self.background_slider_ax, "Background", valmin = -max(self.img.lineout), valmax = max(self.img.lineout), valinit=0)

        #buttons
        fit_button_ax = self.fig.add_axes([0.85, 0.13, 0.1, 0.05])
        self.fit_button = Button(fit_button_ax, "Fit")

        peak_button_ax = self.fig.add_axes([0.85, 0.07, 0.1, 0.05])
        self.peak_button = Button(peak_button_ax, "Add Peak")

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

    def update_peak_slider(self, val):
        for c, peak_line in enumerate([self.peak_min, self.peak_max]):
            peak_line.set_xdata([val[c]])

    def update_background_slider(self, val):
        self.background_line.set_ydata([val])

    def click_fit(self, val):
        min_bound = np.argmin(np.abs(self.img.wavelengths - self.peak_loc_slider.val[0]))
        max_bound = np.argmin(np.abs(self.img.wavelengths - self.peak_loc_slider.val[1]))
        xdata = self.img.wavelengths[min_bound:max_bound]
        ydata = self.img.lineout[min_bound:max_bound]
        peak = XUVPeak(xdata, ydata, background = self.background_slider.val)
        peak.fit_peak()
        self.peak = peak
        self.has_fit = True
        peak.show_fit()

    def add_peak_to_table(self, area):
        '''
        helper function for click_add_peak
        adds all peak info to the peak dictionary
        which will be saved as csv
        '''
        vals = [self.num_peaks_measured, np.round(self.peak.x0, 5), np.round(self.peak.amp, 5), np.round(area, 5)]
        if self.num_peaks_measured > 0:
            #if this is not the first peak, add a blank row
            add_row(self.peak_table, ["", "", ""])
        update_row(self.peak_table, self.num_peaks_measured + 1, vals)

    def click_add_peak(self, val):
        print(self.has_fit)
        if self.has_fit == True: # only add if we currently have the fit
            full_peak = self.peak.get_peak(self.img.wavelengths)
            area = np.trapz(full_peak, self.img.wavelengths)
            peak_line = self.lineout_ax.plot(self.img.wavelengths, full_peak, linestyle = "--", linewidth = 1, c = self.peak_colors[self.num_peaks_measured], label = self.num_peaks_measured)
            self.measured_peak_lines.append(peak_line)
            self.add_peak_to_table(area)
            self.num_peaks_measured += 1
            self.lineout_ax.legend()
            self.fig.canvas.draw_idle()
            print("peak added")

    def set_widgets(self):
        self.x_zoom_slider.on_changed(self.update_xrange_slider)
        self.peak_loc_slider.on_changed(self.update_peak_slider)
        self.background_slider.on_changed(self.update_background_slider)
        self.fit_button.on_clicked(self.click_fit)
        self.peak_button.on_clicked(self.click_add_peak)

    def show(self):
        self.set_widgets()
        plt.show()