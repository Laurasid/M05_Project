preprocessor package
====================

preprocessor.preprocessing module
---------------------------------

.. automodule:: preprocessor.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Full use example
----------------
.. code-block:: python
   :linenos:

   from preprocessor import preprocessing as pp

   dataset = pp.import_dataset("Data/winequality-red.csv")

   x_train, x_test, y_train, y_test = pp.preprocessing(dataset, n_split, n_normalization)
