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
    