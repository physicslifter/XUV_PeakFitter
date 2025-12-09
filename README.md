# Example

```
from PeakFitter import LinearCalibration
from GUI import Plotter

my_img = "example.tif"
calibration = LinearCalibration(m = 0.00614, b = 6.832) #calibration for lambda = m*pixel + b
Plotter(fname = my_img, calibration = calibration) #Pull up the interactive plot
```
