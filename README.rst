[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
===========
M05_Project
===========

This project has been developed during the M05 module of Msc AI at Idiap. Its objective is to experiment with the principles of reproductibility in science.

Datasets
========
For this project we use three different datasets. The datasets are already imported into project under the folder data.
  - Two dataset for wine quality (red-white) that could be found here : https://archive.ics.uci.edu/ml/datasets/wine+quality
  - One dataset for housing price that can be found here : https://archive.ics.uci.edu/ml/machine-learning-databases/housing/ 

Installation
============
You can install our program in two ways : 
  1. from git clone if you just want to run the main program
  2. from pip install <package_name> if you want to use our modules in a custom way

Installation from git clone
---------------------------
.. Note:: Make sure you have at least Python 3.8.9 to run the program

We encourage you to make a virtual environnement before installing the dependencies.

Do the following : 
  - ``$ git clone `git@github.com:Laurasid/M05_Project.git```
  - go to the folder
  - ``$ pip install -r requirements.txt``
  - ``$ python main.py``
  - follow the instructions

Installation from pip
---------------------
Do the following : 
  - ``$ pip install -i https://test.pypi.org/simple/ repro-m05==1.1.1``
  - ``$ python``
  - ``$ import <wanted_package>``
  - or ``$ from <wanted_package> import <wanted_method>``

Enjoy !

The use of each methods is specified in the dedicated documentation index.

Project structure
=================
This project is composed of four packages and one main. 
  - Data : package containing the datasets and data info
  - Preprocessor : package containing the preprocessing methods
  - Model : package containing the different regression models
  - Analysis : package containing the loss function methods
  - main.py : run the program
  
.. image:: /src/doc/tree_image.png

