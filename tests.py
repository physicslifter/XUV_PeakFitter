'''
tests to ensure peak fitting works correctly

test_basic_lineout: Plots a single lineout for the test img
test_GUI: pulls up the GUI for the test img
test_peak_fit: Fits a peak
'''
from PeakFitter import *
from matplotlib import pyplot as plt
from GUI import *

#=====
test_img = "../../../Downloads/400um_spot_size/500mJ/xuv/alfs6_sig.tif"
#=====

#=====
test_basic_lineout = 0
test_GUI = 1
test_peak_fit = 0
demo_x_axis = 0 #show x-axis in wavelength, pixels and eV 

#=====
if test_basic_lineout == True:
    #setup plot
    fig = plt.figure()
    img_ax = fig.add_subplot(2, 1, 1)
    lineout_ax = fig.add_subplot(2, 1, 2)
    lineout_ax.set_xlabel(r"$\lambda$")
    lineout_ax.set_ylabel("Intensity (a.u.)")
    fig.suptitle(f"{test_img} Data")

    #show data
    img = XUVImage(test_img)
    img.take_lineout()
    calibration = LinearCalibration(m = 0.00614, b = 6.832)
    img.apply_linear_calibration(calibration)
    img.plot_lineout(lineout_ax)
    img.plot_img(img_ax)
    plt.show()

#=====
if test_GUI == True:
    #calibration = LinearCalibration(m = 0.00614, b = 6.832)
    calibration = LinearCalibration(m = 1, b = 0)
    plotter = Plotter(test_img, calibration)

#=====
if test_peak_fit == True:
    img = XUVImage(test_img)
    img.take_lineout()
    calibration = LinearCalibration(m = 0.00614, b = 6.832)
    img.apply_linear_calibration(calibration)
    #get data about peak
    min = 460
    max = 505
    lineout = img.lineout[min:max]
    wavelengths = img.wavelengths[min:max]
    peak = XUVPeak(wavelengths, lineout, background = 0)
    peak.fit_peak()
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom = 0.2)
    ax.plot(wavelengths, lineout, label = "Raw Data", c = "blue")
    ax.plot(wavelengths, peak.fit_line, label = "Voigt Fit", c = "magenta")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Voigt Fit Test")
    ax.legend()
    fig.text(0.5, 0.01, f"Amp: {peak.amp}\nCenter: {peak.x0}\n$\sigma$: {peak.sigma}\n$\gamma$: {peak.gamma}")
    plt.show()

if demo_x_axis == True:
    fig = plt.figure(figsize = (6, 8))
    ax_pixels = fig.add_subplot(3, 1, 1)
    ax_wavelength = fig.add_subplot(3, 1, 2)
    ax_energy = fig.add_subplot(3, 1, 3)
    axes = [ax_pixels, ax_wavelength, ax_energy]
    titles = ["Pixels", "Wavelength", "Energy"]
    axis_types = ["pixels", "wavelength", "energy"]
    x_labels = ["Pixel no.", r"$\lambda$ (nm)", "E (eV)"]
    img = XUVImage(test_img)
    img.take_lineout()
    calibration = LinearCalibration(m = 0.00614, b = 6.832)
    img.apply_linear_calibration(calibration)
    diff_x_datas = [img.pixels, img.wavelengths, img.energy]
    for ax, title, axis_type, x_label, my_x in zip(axes, titles, axis_types, x_labels, diff_x_datas):
        #plot data
        #ax.set_title(title)
        ax.set_xlabel(x_label)
        img.plot_lineout(ax, x_axis = axis_type)

        #get & plot Voight fit to data
        peak_loc = [425, 445]
        x_data = my_x[peak_loc[0]:peak_loc[1]]
        y_data = img.lineout[peak_loc[0]:peak_loc[1]]
        voight_peak = XUVPeak(x_data, y_data, background = 0)
        voight_peak.fit_peak()
        full_peak = voight_peak.get_peak(my_x)
        ax.plot(my_x, full_peak, label = "Voight Fit", color = "magenta", linestyle = "--", linewidth = 1)
        ax.fill_between(x = my_x, y1 = np.zeros_like(my_x), y2 = full_peak, color = "magenta", alpha = 0.1)

        #get & plot Alamgir's method
        peak_loc = [710, 730]
        alamgir_peak = get_AP2(img = img, data_bounds = peak_loc, background = 0, xdata = axis_type)
        #print(alamgir_peak.real_x_data)
        alamgir_peak.get_area(width = 10)
        alamgir_peak.show_fit(ax = ax)
        
        ax.legend()
        print(title)
    plt.tight_layout()
    plt.show()
    