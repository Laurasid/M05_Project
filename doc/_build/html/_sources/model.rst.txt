model package
=============

This package gather two modules used to construct different AI models

model.linear\_regression module
-------------------------------

.. automodule:: model.linear_regression
   :members:
   :undoc-members:
   :show-inheritance:

Full use example
----------------
.. code-block:: python
   :linenos:

   from model import linear_regression as lr

   regressor = lr.train(x_train, y_train)
   y_pred = lr.predict(regressor, x_test)

model.regression\_tree module
-----------------------------

.. automodule:: model.regression_tree
   :members:
   :undoc-members:
   :show-inheritance:

Full use example
----------------
.. code-block:: python
   :linenos:

   from model import regression_tree as rt

   regressor = rt.train(x_train, y_train)
   y_pred = rt.predict(regressor, x_test)
