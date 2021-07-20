# recsys-final-proj
Final project of RecSys course BGU 2021 Semester 1

Improvement of the paper of Ziwei Zhu et al. [link to paper](http://people.tamu.edu/~jwang713/pubs/bias-sigir2020.pdf), Github [link](https://github.com/Zziwei/Item-Underrecommendation-Bias)

Based on the work of Ludovico Boratto et al. [link to paper](https://www.sciencedirect.com/science/article/pii/S0306457320308827?via%3Dihub)

## Installation
1. Install Python (>= **3.X**)
2. Install the package requirements:

`pip install -r all_reqs.txt`

## Usage
In order to run any of the BPR / BPR-DPR-RSP / **BPR-DPR-RSP-DEBIAS** algorithms, please run:
> `python <NAME-OF-ALGORITHM>.py [**args]`

In order to see the available arguments (i.e. number of epochs, hyper-parameters etc.), please run:
> `python <NAME-OF-ALGORITHM>.py -h`

## Outputs
Outputs (summary plots) will be available in the "outputs" directory which will be created in the main project root directory
