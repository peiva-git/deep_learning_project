# Deep Learning Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peiva-git/deep_learning_project/blob/main/simple_model.ipynb)

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

## Project description

Image denoising is a fundamental problem in computer vision, with a wide range of real-world applications, including improving the quality of images for object segmentation, detection, tracking, and more. This project aims to explore and implement state-of-the-art deep learning models for image denoising, with a particular focus on CBDNet, a convolutional blind denoising network.

Deep Learning has demonstrated remarkable success in this field [1], and our project would leverage this technology to address the issue of noisy images. We’d delve into the principles of deep learning, especially Convolutional Neural Networks (CNNs), and understand how these networks can be trained to remove noise and enhance image quality.

One of the key highlights of this project is the exploration of CBDNet, a novel approach to image denoising [2]. CBDNet focuses on improving the generalization ability of deep CNN denoisers by training with realistic noise models and real-world noisy-clean image pairs. It considers signal-dependent noise and in-camera signal processing pipelines to synthesize realistic noisy images. Additionally, a noise estimation subnetwork is embedded within CBDNet to rectify denoising results conveniently.

We would implement both CBDNet and simpler deep learning models for image denoising using Python. After this stage, we would perform an extensive analysis of the results obtained. Evaluation will include qualitative and quantitative metrics. Qualitatively, aspects such as edge preservation, texture, uniformity, and smoothness will be considered. Quantitative metrics like Peak Signal-to-Noise Ratio (PSNR)[3], Structural Similarity Index Measurement (SSIM)[4], and Mean Square Error (MSE) will be used to compare the performance of the deep learning-based denoising methods with traditional techniques. The project will provide a clear understanding of the strengths and limitations of different models.

References

[1] Image Denoising using Deep Learning: Convolutional Neural Network, Shreyasi Ghose et al., 2020

[2] Toward Convolutional Blind Denoising of Real Photographs, Shi Guo et al., 2019

[3] Peak Signal-to-Noise Ratio as an Image Quality Metric, www.ni.com, National Instruments

[4] Image quality assessment: from error visibility to structural similarity, Zhou Wang et al., 2004
