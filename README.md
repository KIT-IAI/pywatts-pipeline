![Pipeline status](https://github.com/KIT-IAI/pywatts-pipeline/workflows/Python%20application/badge.svg?branch=master)
[![DOI](https://zenodo.org/badge/297579218.svg)](https://zenodo.org/badge/latestdoi/297579218)
[![Documentation](https://readthedocs.org/projects/pywatts/badge/)](https://pywatts.readthedocs.io/en/latest/)


# pyWATTS Pipeline - Now Deprecated

**Please note that pyWATTS Pipeline is no longer actively maintained. Therefore, the contents of this repository are now read-only!**

**The functionality of pyWATTS Pipeline has now been integrated into the open source python package [sktime](https://www.sktime.net/en/stable/).**

We would like to thank everybody who has helped develop, test, and use pyWATTS Pipeline in the last few years and strongly advise all past pyWATTS users to apply [sktime](https://www.sktime.net/en/stable/) in the future.


# Legacy Information
This repository contains the functionality to define and execute a Graphpipeline for machine learning workflows.
Such workflows may include all tasks in time series analysis ranging from outlier handling as preprocessing to rescaling as postprocessing.

Compatible machine learning modules can be found in the [pyWATTTS-Repo](https://github.com/KIT-IAI/pyWATTS).

## Installation

To install and use pyWATTS you probably want to install the pipeline alongside with the modules. Therefore,
refer to  [pyWATTTS-Repo](https://github.com/KIT-IAI/pyWATTS).

If you want to **only** install the pipeline without machine learning modules, you have to perform the following steps. 

1. Clone the project
2. Open a terminal of the virtual environment where you want to use the project
2. cd pywatts
3. ``pip install .`` or ``pip install -e .``
   if you want to install the project editable. If you aim to develop code for pyWATTS, you should
   use:  ``pip install -e .[dev]``

   
## How to use
See [pyWATTTS-Repo](https://github.com/KIT-IAI/pyWATTS)


## Goals

The goals of pyWATTS (Python Workflow Automation Tool for Time-Series) are

* to support researchers in conducting automated time series experiments independent of the execution environment and
* to make methods developed during the research easily reusable for other researchers.

Therefore, pyWATTS is an automation tool for time series analysis that implements three core ideas:

* pyWATTS provides a pipeline to support the execution of experiments. This way, the execution of simple and often
  recurring tasks is simplified. For example, a defined preprocessing pipeline could be reused in other experiments.
  Furthermore, the execution of defined pipelines is independent of the execution environment. Consequently, for the
  repetition or reuse of a third-party experiment or pipeline, it should be sufficient to install pyWATTS and clone the
  third-party repository.
* pyWATTS allows the definition of end-to-end pipelines for experiments. Therefore, experiments can be easily executed that
  comprise the preprocessing, models and benchmark training, evaluation, and comparison of the models with the
  benchmark.
* pyWATTS defines an API that forces the different methods (called modules) to have the same interface in order to make
  newly developed methods more reusable.

## Features

* Reuseable modules
* Plug-and-play architecture to insert modules into the pipeline
* End-to-end pipeline for experiments such that pipeline performs all necessary steps from preprocessing to evaluation
* Conditions within the pipeline
* Saving and loading of the entire pipeline including the pipeline modules
* Adapters and wrappers for existing machine learning libraries

## Programming Guidelines

* Implement new features on a new, separate branch (see "Module Implementation Worflow" below). If everything works,
  open a pull request to merge the branch with the master. Note that you should name your branch with the following
  convention: <feature|docs|bugfix>/<issue_number>_<descriptive_name>
* Provide tests for your module (see "Tests" below).
* Provide proper logging information (see "Logging" below).
* Use a linter, follow pep8, and add docstrings to your classes and methods<br>
  To do so in PyCharm, activate in Settings -> Editor -> Inspections -> Python:
    * "PEP8 coding style violation"
    * "missing or empty docstring"
    * "missing type hinting for function parameter"
    * "package requirement"
* Use typing (see https://docs.python.org/3/library/typing.html).

## Current Development Status

pyWATTS Pipeline is deprecated and will no longer be maintained.

### Standing assumptions

* The graph representing the pipeline is a directed acyclic graph (DAG).
* The first coordinate is always the time coordinate.

# Funding

This project is supported by the Helmholtz Association under the Program “Energy System Design”, by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the Helmholtz Association under the Joint Initiative "Energy System 2050 - A Contribution of the Research Field Energy", and by the German Research Foundation (DFG) Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation".

# Citation
If you use this framework in a scientific publication please cite the corresponding paper:

>Benedikt Heidrich, Andreas Bartschat, Marian Turowski, Oliver Neumann, Kaleb Phipps, Stefan Meisenbacher, Kai Schmieder, Nicole Ludwig, Ralf Mikut, Veit Hagenmeyer. “pyWATTS: Python Workflow Automation Tool for Time Series.” (2021). ). arXiv:2106.10157. http://arxiv.org/abs/2106.10157
