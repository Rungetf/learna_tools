# LEARNA-tools
Generative RNA Design with Automated Reinforcement Learning

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

These tools run with the default parameters of each of the algorithms. However, it is also possible to change the parameters.
You can run

```
$ <tool> -h
```
, where `<tool>` is one of `learna, meta-learna, meta-learna-adapt, liblearna`, to see a list of available options for the respective tool.

In the following, we provide some information about the different approaches for RNA design as well as on how to run each individual tool.

### LEARNA

The LEARNA algorithm takes a secondary structure in dot-bracket notation as input to generate a RNA sequence that folds into the desired structure.
The algorithm updates its policy each time it has generated a new sequence and, thus, gets better and better over time by successively updating its weights based on previous predictions.
We provide a version of *LEARNA* with tuned hyperparameters as described in our ICLR'19 paper [Learning to Design RNA](https://openreview.net/pdf?id=ByfyHh05tQ).

#### Input
*LEARNA* either reads a secondary structure directly from the commandline, or from an input file, starting with a structure Id, followed by the desired structure in dot-bracket notation.

An example input file might look as follows:

```
> Test structure 1
....((((....))))....
> Test structure 2
..((....))..
> Test structure 3
....((((....))))...............
```

If multiple structures are provided with the input file, *LEARNA* will try to solve one structure after the other.

The easiest way of running `LEARNA` from commandline is to simply type

```
$ learna --target-structure <RNA structure in dot-bracket format>
```

This will run the *LEARNA* algorithm on the secondary structure in dot-bracket notation.

**Note:** LEARNA Does not support pseudoknots. The input structure has to be in standard dot-bracket notation, i.e. the input may only contain `'.', '(', and ')'`.

A real example of a `LEARNA` call then looks as follows

```
$ learna --target-structure ...(((((....)))))...
```

You can use the `--min_solutions` argument to define the number of (optimal) solutions that LEARNA should provide.
Using the `hamming_tolerance` argument, you can further define a distance (Hamming distance between the input structure and the folded candidate sequence) threshold to ask LEARNA to additionally output all sub-optimal solutions with a distance below the given threshold.

For example, the output of the call
```
$ learna --target-structure ...(((((....)))))... --min_solutions 10 --hamming_tolerance 10
```
could then look as follows

|    |   Id |      time |   hamming_distance |   rel_hamming_distance | sequence             | structure            |
|---:|-----:|----------:|-------------------:|-----------------------:|:---------------------|:---------------------|
|  0 |    1 | 0.0187199 |                  0 |                    0   | GUCUACAGCUCUCUGUAUUG | ...(((((....)))))... |
|  1 |    1 | 0.0293458 |                  0 |                    0   | AUUCGAUCCUGCGAUCGCGC | ...(((((....)))))... |
|  2 |    1 | 0.033498  |                  0 |                    0   | GCCGGCGUGCUGACGCCCAA | ...(((((....)))))... |
|  3 |    1 | 0.0387537 |                  0 |                    0   | AAUACUACACCCGUAGUGAA | ...(((((....)))))... |
|  4 |    1 | 0.0474875 |                  0 |                    0   | CUCGAUGACCCCUCAUCCAC | ...(((((....)))))... |
|  5 |    1 | 0.0523767 |                  0 |                    0   | CGGCCAUCAUAUGAUGGACG | ...(((((....)))))... |
|  6 |    1 | 0.116002  |                  0 |                    0   | GCACUAGCUGGAGCUAGCUC | ...(((((....)))))... |
|  7 |    1 | 0.120159  |                  0 |                    0   | ACCAGUUUGUUUAAACUCAC | ...(((((....)))))... |
|  8 |    1 | 0.124296  |                  0 |                    0   | GGAGAAGCUCGGGCUUCGGC | ...(((((....)))))... |
|  9 |    1 | 0.128402  |                  0 |                    0   | AAUUGGAGCGCUCUCCAUCC | ...(((((....)))))... |
| 10 |    1 | 0.0246227 |                  6 |                    0.3 | CUGGGCACUGCGGUGCCCAG | ((((((((....)))))))) |
| 11 |    1 | 0.0428925 |                  6 |                    0.3 | GAUAUGAUGACAAUCAUCAC | ....((((((...)))))). |

**Note:** The last two predictions are sub-optimal with a Hamming distance of 6 each. The output is sorted by Hamming distance.


### Meta-LEARNA


