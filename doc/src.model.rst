src.model package
=================

src.model.linear\_regression module
-----------------------------------

.. automodule:: src.model.linear_regression
   :members:
   :undoc-members:
   :show-inheritance:

Use example
-----------

.. code-block:: python

   from model import linear_regression as lr

   regressor = lr.train(x_train, y_train)
   y_pred = lr.predict(regressor, x_test)


src.model.regression\_tree module
---------------------------------

.. automodule:: src.model.regression_tree
   :members:
   :undoc-members:
   :show-inheritance:

Use example
-----------

.. code-block:: python

   from model import regression_tree as rt

   regressor = rt.train(x_train, y_train)
   y_pred = rt.predict(regressor, x_test)

