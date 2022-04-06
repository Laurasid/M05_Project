analysis package
================

analysis.analyse module
-----------------------

.. automodule:: analysis.analyse
   :members: mae, r2, rmse, correlation_matrix
   :undoc-members:
   :show-inheritance:

Full use example
----------------

.. code-block:: python
   :linenos:

   from analysis import analyse

   ...  # see full use example of preprocessor module

   x_train, x_test, y_train, y_test = pp.preprocessing(dataset, n_split, n_normalization)

   mae = analyse.mae(y_test, y_pred)
   r2 = analyse.r2(y_test, y_pred)
   rmse = analyse.rmse(y_test, y_pred)

   analyse.correlation_matrix(dataset)
