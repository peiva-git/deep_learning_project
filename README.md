# Deep Learning Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peiva-git/deep_learning_project/blob/main/deep_learning_project.ipynb)

[![build and test CPU](https://github.com/peiva-git/deep_learning_project/actions/workflows/build-and-test-cpu.yml/badge.svg)](https://github.com/peiva-git/deep_learning_project/actions/workflows/build-and-test-cpu.yml)
[![build and test GPU](https://github.com/peiva-git/deep_learning_project/actions/workflows/build-and-test-gpu.yml/badge.svg)](https://github.com/peiva-git/deep_learning_project/actions/workflows/build-and-test-gpu.yml)
[![Build LaTeX report](https://github.com/peiva-git/deep_learning_project/actions/workflows/compile-report-pdf.yml/badge.svg)](https://github.com/peiva-git/deep_learning_project/actions/workflows/compile-report-pdf.yml)
![License](https://img.shields.io/github/license/peiva-git/deep_learning_project)

Exam project for the Deep Learning course @ UniTS

## Project setup

To install the GPU version, in order to use the project with a CUDA enabled GPU, run:
```shell
conda create --name dl-tf --file conda/tf-gpu.yml
```

Alternatively, to install the CPU version, run:
```shell
conda create --name dl-tf --file conda/tf-cpu.yml
```

## Compile report

The project report is provided as a set of LaTeX source files.
The build process is managed by [latexmk](https://mg.readthedocs.io/latexmk.html).

The following packages are required for a successful compilation and can be installed with the following commands
on a Linux system using the apt package manager:
```shell
apt install texmaker texlive-babel-italian texlive-hyphen-italian texlive-subfigmat texlive-appendix
apt install latexmk
```
In order to compile the output `report/out/main.pdf` file, using the [provided configuration file](report/.latexmkrc),
simply run the following commands:
```shell
cd report/
latexmk
```
