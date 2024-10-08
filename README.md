# Auto PWLF
Piecewise Linear Fit with ability to automatically find the optimal number of line segments.
In addition outliers can be detected and avoided during the fitting process.

Full Documentation can be found here: [Documentation](https://autopwlf.readthedocs.io/en/latest/#)


Installation available through pip: https://pypi.org/project/autopwlf/
```
pip install autopwlf
```

![image](https://github.com/user-attachments/assets/c4f8bc1c-5287-431e-a3ea-abb289ca7fcb)



The piecewise fitting uses the [pwlf module](https://github.com/cjekel/piecewise_linear_fit_py) with full credit to all contributors to this package.

## How it works
The currently available Piecewise Linear Fit models in Python do not have a method to determine the optimal number of break points to use. To do so this package performs the following steps:

1. Smooth the data using a rolling median function
2. Fit a linear interpolation function on the smoothened data
3. Use scipy find_peaks function to find the number of extrema points
4. Filter by primenence to return an estimate on the number of break points


