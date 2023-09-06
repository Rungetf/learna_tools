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






To install `LEARNA-tools`
