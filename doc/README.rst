.. image:: https://github.com/Laurasid/M05_Project/actions/workflows/python-app.yml/badge.svg?branch=main
   :target: https://github.com/Laurasid/M05_Project/actions/workflows/python-app.yml
.. image:: https://img.shields.io/badge/github-project-0000c0.svg
   :target: https://github.com/Laurasid/M05_Project/tree/dist
.. image:: https://img.shields.io/github/license/Naereen/StrapDown.js.svg
   :target: https://github.com/Laurasid/M05_Project/blob/dist/LICENSE.txt
.. image:: https://coveralls.io/repos/github/Laurasid/M05_Project/badge.svg?branch=main
   :target: https://coveralls.io/github/Laurasid/M05_Project?branch=main

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
.. Note:: Make sure you have at least Python 3.8.9 to run the program

We encourage you to make a virtual environnement before installing the dependencies.

Do the following : 
  - ``$ git clone `https://github.com/Laurasid/M05_Project.git```
  - go to the folder
  - ``$ pip install .``
  - ``$ m05-run``
  - follow the instructions

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
