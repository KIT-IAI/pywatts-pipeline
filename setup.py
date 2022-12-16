import setuptools

dev = [
    "pytest",
    "sphinx>=4",
    "pylint",
    "pytest-cov"
]

visualize = ["schemdraw"]

setuptools.setup(
    name="pywatts-pipeline",
    version="0.1.0",
    packages=setuptools.find_packages(),

    install_requires=['cloudpickle', 'distlib', 'xarray>=0.19', 'numpy', 'pandas', 'tabulate'],
    extras_require={
        'dev': dev + visualize,
        'visualize': visualize
    },
    author="pyWATTS-TEAM",
    author_email="pywatts-team@iai.kit.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Natural Language :: English",
        "Operating System :: OS Independent",

    ],
    description="A python time series pipelining project",
    keywords="preprocessing time-series machine-learning",
)
