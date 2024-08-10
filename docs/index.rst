AutoPWLF Documentation
==========================

Piecewise Linear Fit with automated selection of the number of segments.


Installation
----------------

.. code-block:: bash

    $ pip install -U autopwlf


Contents
-----------

.. autoclass:: autopwlf.AutoPWLF
    :members:
    :undoc-members:
    :show-inheritance:


Examples
-----------

Default usage involves calling auto_fit() method with the x and y data. fastfit is set to True by default.

.. code-block:: python

    apwlf = autopwlf.AutoPWLF(x_data=data.index, y_data=data.values)

    optimal_breaks, my_pwlf = apwlf.auto_fit() # default fastfit=True & buffer=2

.. figure:: https://github.com/nweerasuriya/auto_pwlf/blob/main/examples/Climate_autofit.png

For better model perfomance (at the cost of speed), set fastfit=False. Recommended to use buffer of 0 or 1 to prevent large runtime.

.. code-block:: python

    apwlf = autopwlf.AutoPWLF(x_data=data.index, y_data=data.values)

    optimal_breaks, my_pwlf = apwlf.auto_fit(fastfit=False, buffer=1)

