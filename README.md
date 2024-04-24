# Gradient Convolution Kernel Compensation Applied to Surface Electromyograms

This repository contains a personal, unofficial implementation of the algorithm described in the paper [Gradient Convolution Kernel Compensation Applied to Surface Electromyograms](https://link.springer.com/chapter/10.1007/978-3-540-74494-8_77).  It is written in Python and is intended for educational and research purposes.

## Description

This project implements the methods discussed in the referenced paper using Python. The main focus is on applying gradient convolution kernel compensation to surface electromyograms (sEMG) data to enhance the signal processing pipeline.

## Installation

To run this project, you need to have Python installed on your system. The code has been tested on Python 3.12, but other versions might also work.

### Prerequisites

Ensure you have Python installed. You can download Python from [python.org](https://www.python.org/downloads/). It's recommended to use [pyenv](https://github.com/pyenv/pyenv) for managing Python versions.

### Setup

Clone this repository to your local machine using:

```sh
git clone https://github.com/panoanx/gckc-emg.git
cd gckc-emg
```

Install the required Python packages using:

```sh
pip install -r requirements.txt
```

## Usage

To run the notebook that contains the implementation, use the following command:

```sh
jupyter notebook test/main.ipynb
```

This will open the Jupyter Notebook in your browser where you can run the cells interactively.

## Disclaimer

This is a personal, unofficial implementation of the paper "Gradient Convolution Kernel Compensation Applied to Surface Electromyograms" and is not affiliated with the original authors of the paper. This implementation was created for educational purposes and should be used accordingly.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- Original authors of the paper for their significant contributions to the field.
- Contributors and maintainers of the Python packages used in this project.