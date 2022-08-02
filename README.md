![Pipeline status](https://github.com/KIT-IAI/pywatts_pipeline/workflows/Python%20application/badge.svg?branch=master)
<!-- TODO Adapt path to readthedocs if new website is available -->
[![Documentation](https://readthedocs.org/projects/pywatts/badge/)](https://pywatts.readthedocs.io/en/latest/)


# WIP: Extract pyWATTS Pipeline in new repository.

For executing this pipeline please clone and checkout the pipeline_extraction branch from the main pyWATTS Repo.

# pyWATTS Pipeline

<!-- Insert Text about the aim of this repository -->

## Installation

<!-- Link to pyWATTS repo -->

## How to use

## Contributing

### Testing

#### Writing tests

To write tests, see the test template where the different methods are explained. When writing tests, please consider the
following basic rules for testing:

* In general, tests should not only cover the normal case but also edge cases such as exceptions when wrong inputs are
  used.
* After writing tests, please also check if the tests sufficiently cover the code of your module. However, be aware that
  the line-coverage metric provided will only help you to identify uncovered source code. Please also note that 100%
  line coverage does not mean that your code is free of bugs.
* If you fix a bug, implement a test case which repeats the bug.

If you are looking for some test guidelines, have a look at https://docs.python-guide.org/writing/tests/

#### Test libraries

* **[unittest](https://docs.python.org/3/library/unittest.html)** We use unittests to define our test cases.
    * **[unittest.mock](https://docs.python.org/3/library/unittest.mock.html)** We use this mock object library to mock
      calls of other methods. In mocks, it is possible to check whether a method is called correctly.
    * **[unittest.mock.patch](https://docs.python.org/3/library/unittest.mock.html#the-patchers)** We use this library
      to replace objects imported in the module under test with mock objects.
* **[pytest](https://docs.pytest.org/en/latest/)** We use pytest to run our tests.
* **[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)** We use pytest-coverage to calculate the test coverage
  of our source code.

### Run tests

To run tests, use your preferred editor, IDE, or the following CLI command

``
pytest tests
``

This command executes all tests and prints the results. It also provides the code coverage of the project.




### Documentation
  

# Funding

This project is supported by the Helmholtz Association under the Program “Energy System Design”, by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the Helmholtz Association under the Joint Initiative "Energy System 2050 - A Contribution of the Research Field Energy", and by the German Research Foundation (DFG) Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation".

# Citation
If you use this framework in a scientific publication please cite the corresponding paper:

>Benedikt Heidrich, Andreas Bartschat, Marian Turowski, Oliver Neumann, Kaleb Phipps, Stefan Meisenbacher, Kai Schmieder, Nicole Ludwig, Ralf Mikut, Veit Hagenmeyer. “pyWATTS: Python Workflow Automation Tool for Time Series.” (2021). ). arXiv:2106.10157. http://arxiv.org/abs/2106.10157

# Contact
If you have any questions and want to talk to the pyWATTS Team directly, feel free to [contact us](mailto:pywatts-team@iai.kit.edu).
For more information on pyWATTSvisit the [project website](https://www.iai.kit.edu/english/1266_4162.php).
