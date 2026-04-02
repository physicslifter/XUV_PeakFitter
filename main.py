"""
Entry point for XUV PeakFitter.
Run with:  python main.py
"""
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from GUI import Plotter
from PeakFitter import LinearCalibration


def main():
    root = tk.Tk()
    root.withdraw()  # hide blank root window

    # --- 1. Select image file ---
    fname = filedialog.askopenfilename(
        title="Select XUV TIFF image",
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
    )
    if not fname:
        messagebox.showinfo("Cancelled", "No file selected. Exiting.")
        return

    # --- 2. Calibration ---
    use_cal = messagebox.askquestion(
        "Wavelength Calibration",
        "Do you want to enter a wavelength calibration?\n\n"
        "The calibration converts pixels to wavelength (nm):\n"
        "  λ = m × pixel + b\n\n"
        "Select 'No' to use pixel units (m=1, b=0)."
    )
    if use_cal == "yes":
        m_str = simpledialog.askstring(
            "Calibration — slope",
            "Enter slope  m  (nm per pixel):",
            parent=root
        )
        b_str = simpledialog.askstring(
            "Calibration — intercept",
            "Enter intercept  b  (nm):",
            parent=root
        )
        try:
            m = float(m_str)
            b = float(b_str)
        except (TypeError, ValueError):
            messagebox.showerror(
                "Invalid input",
                "Could not parse calibration values. Using m=1, b=0 (pixel units)."
            )
            m, b = 1.0, 0.0
    else:
        m, b = 1.0, 0.0

    root.destroy()

    calibration = LinearCalibration(m=m, b=b)
    Plotter(fname, calibration)


if __name__ == "__main__":
    main()
