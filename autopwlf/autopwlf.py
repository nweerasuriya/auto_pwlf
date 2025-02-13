"""
MIT License

Copyright (c) 2024 Nedeesha Weerasuriya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

__date__ = "2025-02-13"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.9.1"


from typing import Tuple

import numpy as np
import pwlf
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_prominences, savgol_filter
from sklearn.linear_model import ElasticNetCV


class AutoPWLF(object):
    """
    Class to find the optimal number of breaks for a piecewise linear fit using the Bayesian Information Criterion (BIC)
    and then fit the piecewise linear function to the given data set
    """

    def __init__(
        self,
        x_data: np.array,
        y_data: np.array,
        disp_res: bool = False,
        lapack_driver: str = "gelsd",
        degree: int = 1,
        weights: np.array = None,
        random_seed: int = None,
        smooth_polyorder: int = 0,
        peak_threshold: float = 0.25,
        prominence_threshold: float = 0.1,
    ):
        """
        Initialize the AutoPWLF class to find the optimal number of breaks and fit a continuous piecewise linear function.
        Supply x and y values which will be used to fit the piecewise linear function to where y(x) = f(x).

        Args:
            x_data: independent variable
            y_data: dependent variable
            disp_res: display results
            lapack_driver: LAPACK driver to use for the linear least squares problem
            degree: degree of the polynomial to fit
            weights: weights for the least squares problem
            random_seed: random seed for reproducibility
            smooth_polyorder: polynomial order for the Savitzky-Golay filter
            peak_threshold: threshold for identifying peaks and valleys
            prominence_threshold: threshold for identifying significant peaks and valleys
        """
        x, y = self._switch_to_np_array(x_data), self._switch_to_np_array(y_data)

        self.x, self.y = x, y
        self.disp_res = disp_res
        self.lapack_driver = lapack_driver
        self.degree = degree
        self.weights = weights
        self.random_seed = random_seed
        self.smooth_polyorder = smooth_polyorder
        self.peak_threshold = peak_threshold
        self.prominence_threshold = prominence_threshold

        if random_seed:
            np.random.seed(random_seed)

        # Initialise empty attributes
        self.stationary_points = None
        self.y_smooth = None
        self.optimal_breaks = None
        # Outliers attributes
        self.outliers = None
        self.outlier_detection_model = None
        self.outlier_detection_pred = None
        self.outlier_detection_residuals = None

    @staticmethod
    def _switch_to_np_array(input_):
        r"""
        Check the input, if it's not a Numpy array transform it to one.

        Args:
            input_: input data

        Returns:
            input_: input data as a Numpy array
        """
        if isinstance(input_, np.ndarray) is False:
            input_ = np.array(input_)
        return input_

    @staticmethod
    def _filter_outliers(
        x: np.ndarray, y: np.ndarray, outliers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from the data.

        Args:
            x: Independent variable.
            y: Dependent variable.
            outliers: Indices of outliers to be removed.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered x and y data without outliers.
        """
        mask = ~np.isin(np.arange(len(x)), outliers)
        return x[mask], y[mask]

    @staticmethod
    def _find_peaks_valleys(
        data, peak_threshold: float, prominence_threshold: float
    ) -> tuple:
        """
        Find the peaks and valleys of the smoothed data set

        Args:
            data: data set
            peak_threshold: threshold for identifying peaks and valleys
            prominence_threshold: threshold for identifying significant peaks and valleys

        Returns:
            significant_peaks: peaks of the data set
            significant_valleys: valleys of the data set
        """

        # Find maxima and minima of the smoothed linear interpolation
        peaks, _ = find_peaks(data, threshold=peak_threshold)
        valleys, _ = find_peaks(-data, threshold=peak_threshold)

        # Filter peaks and valleys by prominence
        peak_prominence = peak_prominences(data, peaks)[0]
        valley_prominence = peak_prominences(-data, valleys)[0]

        significant_peaks = peaks[peak_prominence > prominence_threshold]
        significant_valleys = valleys[valley_prominence > prominence_threshold]

        return significant_peaks, significant_valleys

    @staticmethod
    def _sav_gol_fit(x: np.array, y: np.array, poly_order: int) -> np.array:
        """
        Apply the Savitzky-Golay filter to the data set to smooth the data

        Args:
            x: independent variable
            y: dependent variable
            poly_order: polynomial order for the filter

        Returns:
            np.array: Smoothed y values
        """
        # Smoothing the data using Savitzky-Golay filter
        window_size = int(3 + len(x) / 50)
        # Making sure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        y_smooth = savgol_filter(y, window_size, poly_order)
        return y_smooth

    def auto_fit(
        self,
        fitfast: bool = True,
        buffer: int = 2,
        complexity_penalty=20,
        outliers: bool = False,
        outlier_threshold: int = 5,
    ) -> pwlf.PiecewiseLinFit:
        """
        Fit a piecewise linear function with automated number of breaks found from the stationary points
        Adding a buffer to the number of breaks to allow for more flexibility with the model chosen by the BIC

        Args:
            fitfast: if true, use the fast fitting method for the piecewise linear fit
            buffer: buffer for the number of breaks to allow for more flexibility
            complexity_penalty: complexity penalty for the BIC
            outliers: if true, remove outliers from the data
            outlier_threshold: threshold for identifying outliers: standard deviations from the mean residual

        Returns:
            pwlf.PiecewiseLinFit: PiecewiseLinFit model
        """
        self.stationary_points = self.find_num_stationary_points()
        if outliers:
            # Initial fit to detect outliers
            init_breaks = max(1, self.stationary_points)
            _, init_pwlf = self.fit(self.x, self.y, init_breaks, init_breaks)
            self.outliers = self.find_outliers(init_pwlf, outlier_threshold)
            # Remove outliers from the data
            x_filtered, y_filtered = self._filter_outliers(
                self.x, self.y, self.outliers
            )
        else:
            x_filtered, y_filtered = self.x, self.y

        min_breaks = self.stationary_points - buffer
        max_breaks = self.stationary_points + buffer

        optimal_breaks, my_pwlf = self.fit(
            x_filtered,
            y_filtered,
            min_breaks,
            max_breaks,
            complexity_penalty=complexity_penalty,
            fitfast=fitfast,
        )
        self.optimal_breaks = optimal_breaks
        return my_pwlf

    def find_num_stationary_points(self) -> int:
        """
        Find the number of stationary points in the data set to use as min and max breaks
        First fit a smoothened interpolation function on the data
        Then find the number of peaks and valleys in the data

        Returns:
            int: Number of stationary points
        """
        y_smooth = self._sav_gol_fit(self.x, self.y, poly_order=self.smooth_polyorder)
        # Linear interpolation for quicker runtime
        f = interp1d(self.x, y_smooth, kind="linear")
        xnew = np.linspace(self.x.min(), self.x.max(), num=100)
        ynew = f(xnew)
        # Find maxima and minima of the smoothened linear interpolation
        peaks, valleys = self._find_peaks_valleys(
            ynew, self.peak_threshold, self.prominence_threshold
        )
        stationary_points = (
            len(peaks) + len(valleys)
        ) // 2 + 2  # Adding 2 to account for the end points
        return stationary_points

    def fit(
        self,
        x: np.array,
        y: np.array,
        min_breaks: int,
        max_breaks: int = None,
        complexity_penalty: int = 20,
        fitfast: bool = True,
    ) -> tuple:
        """
        Finds the optimal number of breaks for a piecewise linear fit using the Bayesian Information Criterion (BIC)
        and then plots the piecewise linear fit of the given data set

        Args:
            x: Independent variable.
            y: Dependent variable.
            min_breaks: minimum number of breaks
            max_breaks: maximum number of breaks
            complexity_penalty: complexity penalty for the BIC
            fitfast: if true, use the fast fitting method for the piecewise linear fit

        Returns:
            tuple: Optimal number of breaks, PiecewiseLinFit model
        """
        if not max_breaks:
            max_breaks = min_breaks

        if min_breaks < 1:
            min_breaks = 1

        # BIC calculation
        bic_values = []
        pwlf_list = []
        n = len(y)
        for breaks in range(min_breaks, max_breaks + 1):
            my_pwlf = pwlf.PiecewiseLinFit(
                x,
                y,
                seed=self.random_seed,
                degree=self.degree,
                weights=self.weights,
                lapack_driver=self.lapack_driver,
            )
            if fitfast:
                my_pwlf.fitfast(breaks)
            else:
                my_pwlf.fit(breaks)
            # calculate the residual sum of squares
            yhat = my_pwlf.predict(x)
            rss = np.sum((y - yhat) ** 2)
            # calculate the BIC
            bic = n * np.log(rss / n) + complexity_penalty * breaks * np.log(n)
            bic_values.append(bic)
            pwlf_list.append(my_pwlf)

        optimal_index = np.argmin(bic_values)
        self.optimal_breaks = min_breaks + optimal_index
        return self.optimal_breaks, pwlf_list[optimal_index]

    def find_outliers(self, pwlf_model: pwlf.PiecewiseLinFit, outlier_threshold: int):
        """
        This function adjusts the outliers by fitting a new piecewise linear model without the outliers

        Args:
            pwlf_model: PiecewiseLinFit model
            outlier_threshold: threshold for identifying outliers: standard deviations from the mean residual

        Returns:
            outliers: array containing the outliers

        """
        # Remove the outliers from the data
        fit_breaks = pwlf_model.fit_breaks
        A = pwlf_model.assemble_regression_matrix(fit_breaks, self.x)
        en_model = ElasticNetCV(
            cv=5,
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1],
            fit_intercept=False,
            max_iter=1000,
            n_jobs=-1,
        )
        en_model.fit(A, self.y)

        # Calculate the residuals
        y_hat = pwlf_model.predict(self.x)
        residuals = self.y - y_hat

        # Store attributes for outlier detection
        self.outlier_detection_model = pwlf_model
        self.outlier_detection_pred = y_hat
        self.outlier_detection_residuals = residuals

        # Identify outliers as points that are more than 'outlier_threshold' standard deviations from the mean residual
        outlier_mask = np.abs(
            residuals - np.mean(residuals)
        ) > outlier_threshold * np.std(residuals)
        outliers = self.x[outlier_mask]
        return outliers
