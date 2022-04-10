src.preprocessor package
========================

src.preprocessor.preprocessing module
-------------------------------------

.. automodule:: src.preprocessor.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Use example
-----------

.. code-block:: python

   from preprocessor import preprocessing as pp
   import pkg_resources
   DATAFILE = pkg_resources.resource_filename(__name__, "data.csv")
   
   url = pkg_resources.resource_filename(__name__, "path/to/data/to/import")
   dataset = pp.import_dataset(url)
   x_train, x_test, y_train, y_test = pp.preprocessing(dataset, n_split, n_normalization)
