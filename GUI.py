from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, RangeSlider, Button
from PeakFitter import *
from scipy.optimize import curve_fit
from scipy.special import wofz

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

class Plotter:
    def __init__(self, fname, calibration:LinearCalibration):
        self.img = XUVImage(fname)
        self.img.take_lineout()
        self.img.apply_linear_calibration(calibration)
        self.setup_plot()
        self.initialize_plot()
        self.show()

    def setup_plot(self):
        self.fig = plt.figure(figsize = (13, 8))
        self.lineout_ax = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(bottom = 0.25)

        #sliders
        self.x_zoom_slider_ax = self.fig.add_axes([0.1, 0.15, 0.65, 0.05])
        self.x_zoom_slider = RangeSlider(self.x_zoom_slider_ax, "X range", min(self.img.wavelengths), max(self.img.wavelengths), valinit = (min(self.img.wavelengths), max(self.img.wavelengths)))

        self.peak_loc_slider_ax = self.fig.add_axes([0.1, 0.1, 0.65, 0.05])
        self.peak_loc_slider = RangeSlider(self.peak_loc_slider_ax, "Peak\nLoc", min(self.img.wavelengths), max(self.img.wavelengths), valinit = (min(self.img.wavelengths), max(self.img.wavelengths)))

        self.background_slider_ax = self.fig.add_axes([0.1, 0.05, 0.65, 0.05])
        self.background_slider = Slider(self.background_slider_ax, "Background", valmin = -max(self.img.lineout), valmax = max(self.img.lineout), valinit=0)

        #button
        fit_button_ax = self.fig.add_axes([0.85, 0.1, 0.1, 0.05])
        self.fit_button = Button(fit_button_ax, "Fit")

    def initialize_plot(self):
        self.img.plot_lineout(self.lineout_ax)
        self.peak_min = self.lineout_ax.axvline(x = self.peak_loc_slider.val[0], c = "red")
        self.peak_max = self.lineout_ax.axvline(x = self.peak_loc_slider.val[1], c = "red")
        self.background_line = self.lineout_ax.axhline(y = 0, c = "green")

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
        peak.show_fit()

    def set_widgets(self):
        self.x_zoom_slider.on_changed(self.update_xrange_slider)
        self.peak_loc_slider.on_changed(self.update_peak_slider)
        self.background_slider.on_changed(self.update_background_slider)
        self.fit_button.on_clicked(self.click_fit)

    def show(self):
        self.set_widgets()
        plt.show()