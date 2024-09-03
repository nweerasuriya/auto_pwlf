AutoPWLF Documentation
==========================

Piecewise Linear Fit with automated selection of the number of segments.


Installation
----------------

.. code-block:: bash

    $ pip install -U autopwlf


Examples
-----------

Default usage involves calling auto_fit() method with the x and y data. fastfit is set to True by default.

.. code-block:: python

    import numpy as np
    import autopwlf

    # Initialize variables
    x = np.arange(0, 150)

    y = np.array([ 
            0.1,  0.4, -0.2,  0.7,  1.7,  1.6,  3.1,  1.9,  4.4,  5.3,  3.3,
            5.6,  3.7,  5.8,  6. ,  5.1,  5.9,  5.7, 10.3,  8. ,  7.5,  8.7,
            7.9,  9.2, 11. , 15.2, 18.1, 28.4, 29.3, 29.1, 30.4, 38.4, 37.1,
            40.5, 42.1, 47.4, 49.1, 50.2, 51.9, 52.5, 53.3, 54.2,
            54.1, 58.3, 59.1, 60.2, 54.9, 55.1, 49.3, 48.1, 47.4, 44.1,
            43.7, 42.1, 41.5, 40.3, 39.1, 38.9, 37.7, 36.1, 35.5, 34.3,
            42.1, 36.8, 33.3, 30.1, 28.9, 27.8, 26.9, 26.1, 25.3, 24.6,
            27. , 25.7, 25.7, 24.8, 23.9, 23.4, 22.5, 21.6, 20.7, 19.8, 18.9,
            16.1, 15.4, 14.8, 13.9, 14.9, 12.3, 11.4, 10.7,  9.8,  8.9,  8.1,
            15.3, 18.0, 14.5 , 11.3,  9.4, 10.5, 9.8, 6.8, 6.4, 5.4, 2.9,
            3.8, 2.4, 1.1, 2.5, 3.9, 2.7, 3.7, 4.3, 6.3, 4.9, 6.5 ,
            7.7, 6.4 , 7.8, 7.9, 8.7, 8.3, 10.5, 19.7, 13.5, 19.7, 19.6,
            28.5, 38. , 39. , 36.6, 38.2, 38.3, 37. , 36.2, 37. , 35.3, 34.9,
            33.8, 34.7, 33.1, 33.1, 32.3, 33.9, 30.9, 31.2, 31.7, 30.8, 30.2,
            30.3
            ])

    apwlf = autopwlf.AutoPWLF(x, y)
    model_fit = apwlf.auto_fit() # default complexity penlty is 20

.. figure:: https://github.com/nweerasuriya/auto_pwlf/blob/main/examples/example_default_params.png

Lowering complexity penalty will often result in more segments.

.. code-block:: python

    apwlf = autopwlf.AutoPWLF(x, y)
    model_fit = apwlf.auto_fit(complexity=5)

.. figure:: https://github.com/nweerasuriya/auto_pwlf/blob/main/examples/example_low_complex_penalty.png

For better model perfomance (at the cost of speed), set fastfit=False. Recommended to use buffer of 0 or 1 to prevent large runtime.

.. code-block:: python

    my_pwlf = apwlf.auto_fit(fastfit=False, buffer=1)


If an upper and lower limit is known for the number of segments, use the following:

.. code-block:: python

    apwlf = autopwlf.AutoPWLF(x, y)
    model_fit = apwlf.fit(x, y, min_breaks=2, max_breaks=5)

If outliers are present in the data, they can be detected and ignored in the fitting, using the following:

.. code-block:: python

    x = np.arange(0, 152)
    y = np.array([
        0.1,  0.4, -0.2,  0.7,  1.7,  1.6,  3.1,  1.9,  4.4,  5.3,
        3.3,  5.6,  3.7,  5.8,  6.0,  5.1,  5.9,  5.7, 10.3,  8.0,
        7.5,  8.7,  7.9,  9.2, 11.0, 15.2, 18.1, 28.4, 29.3, 29.1,
        30.4, 38.4, 37.1, 40.5, 42.1, 47.4,  0.2, 49.1, 50.2, 51.9,
        52.5, 53.3, 54.2, 54.1, 58.3, 59.1, 60.2, 54.9, 55.1, 49.3,
        48.1, 47.4, 44.1, 43.7, 42.1, 41.5, 40.3, 39.1, 38.9, 37.7,
        36.1, 35.5, 34.3, 42.1, 36.8, 33.3, 30.1, 28.9, 27.8, 26.9,
        26.1, 25.3, 24.6, 27.0, 25.7, 25.7, 24.8, 23.9, 23.4, 22.5,
        21.6, 20.7, 19.8, 18.9, 16.1, 15.4, 14.8, 13.9, 14.9, 12.3,
        11.4, 10.7,  9.8,  8.9,  8.1, 15.3, 18.0, 14.5, 11.3,  9.4,
        10.5,  9.8,  6.8,  6.4,  5.4,  2.9,  3.8,  2.4,  1.1, 65.0,
        2.5,  3.9,  2.7,  3.7,  4.3,  6.3,  4.9,  6.5,  7.7,  6.4,
        7.8,  7.9,  8.7,  8.3, 10.5, 19.7, 13.5, 19.7, 19.6, 28.5,
        38.0, 39.0, 36.6, 38.2, 38.3, 37.0, 36.2, 37.0, 35.3, 34.9,
        33.8, 34.7, 33.1, 33.1, 32.3, 33.9, 30.9, 31.2, 31.7, 30.8,
        30.2, 30.3
        ])

    apwlf = autopwlf.AutoPWLF(x, y)
    model_fit = apwlf.auto_fit(outliers=True, outlier_threshold=4)

.. figure:: https://github.com/nweerasuriya/auto_pwlf/blob/main/examples/example_with_outliers.png

    
Contents
-----------

.. autoclass:: autopwlf.AutoPWLF
    :members:
    :undoc-members:
    :show-inheritance:



