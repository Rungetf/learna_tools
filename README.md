# LEARNA-tools
RNA Design with Automated Reinforcement Learning

`LEARNA-tools` is a Python package that implements the commandline interface of
LEARNA and libLEARNA as documented in the following publications

- [Learning to Design RNA](https://openreview.net/pdf?id=ByfyHh05tQ)
- [Towards Automated Design of Riboswitches](https://icml-compbio.github.io/2023/papers/WCBICML2023_paper42.pdf)

---
## Installation


---
### Requirements

`LEARNA-tools` requires

- Python 3.6
- RNAfold from the ViennaRNA package

However, we provide a `conda` environment for a more convenient installation of `LEARNA-tools`.

### Install conda environment

To install the current version of `LEARNA-tools` from the github repository, first clone the repo as follows

```
git clone https://github.com/Rungetf/learna_tools.git
```

And `cd` into the cloned directory

```
cd learna_tools
```

You can setup the conda environment to include all requirements with

```
conda env create -f environment.yml
```

and

```
conda activate learna_tools
```

### Installation from github repository

When your system satisfies all requirements, you can install `LEARNA-tools` via pip, either directly within the `learna_tools` by running

```
pip install .
```

or from the PyPi package

```
pip install learna_tools
```


## Usage

We provide simple command line interfaces for the following algorithms

- LEARNA
- Meta-LEARNA
- Meta-LEARNA-Adapt
- libLEARNA

In the following, we provide some information about the different approaches for RNA design as well as on how to run each individual tool.

### LEARNA

The easiest way of running `LEARNA` from commandline is to simply type

```
$ learna --target-structure <RNA structure in dot-bracket format>
```

a real example of a `LEARNA` call then looks as follows

```
$ learna --target-structure ...(((((....)))))...
```






To install `LEARNA-tools`
