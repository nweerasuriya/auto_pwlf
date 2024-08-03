"""
Helper functions
"""

__date__ = "2024-06-24"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def handle_date_col(df, date_col, val_col):
    """
    Handle the date column in the DataFrame
    """
    if date_col:
        x = df.index
    else:
        # Convert the date column to datetime if it's not already
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.dropna(subset=[date_col, val_col])

        # Convert the datetime to Unix timestamp (number of seconds since 1970-01-01 00:00:00 UTC)
        x = (df[date_col] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    return x


def plot_piecewise_linear_fit(
    x: np.array, y: np.array, y_hat: np.array, optimal_breaks: int,
):
    """
    Plots the piecewise linear fit of the given data set

    Args:
        x: x values
        y: y values
        y_hat: predicted y values
        optimal_breaks: optimal number of breaks

    Returns: 
        fig: matplotlib figure object of the piecewise linear fit  
    """
    print(
        "Plotting piecewise linear fit with optimal number of breaks: ", optimal_breaks
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, c="r")
    ax.plot(x, y_hat)

    ax.set_ylabel("Value")
    ax.set_xlabel("Index")
    ax.set_title("Piecewise Linear Fit")
    ax.legend(["Data", "Piecewise Linear Fit"])
    return fig


def plot_savgol_fit(x: np.array, y: np.array, y_smooth: np.array):
    """
    Plots the Savitzky-Golay filter fit of the given data set

    Args:
        x: x values
        y: y values
        y_smooth: smoothed y values

    Returns: 
        fig: matplotlib figure object of the Savitzky-Golay fit  
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, c="r")
    ax.plot(x, y_smooth)

    ax.set_ylabel("Value")
    ax.set_xlabel("Index")
    ax.set_title("Savitzky-Golay Filter Fit")
    ax.legend(["Data", "Savitzky-Golay Fit"])
    return fig
