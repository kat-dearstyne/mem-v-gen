#!/bin/bash
#$ -M kdearsty@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu_card=1

module load python

python3.12 -m pip install --user virtualenv

~/.local/bin/virtualenv -p python3.12 mem-v-gen

source mem-v-gen/bin/activate

python3.12 -m pip install -r requirement.txt

export PATH=${HOME}/.local/bin:${PATH}

python test_error_hypothesis.py