
# Subject-Specific EEG Modeling

This repository contains code for exploring **subject-specific layers (private encoding layers)** in EEG classification tasks. It builds on top of MOABB and Braindecode to implement and compare various architectures that aim to capture individual differences in EEG signals while still leveraging shared representations.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)

---

## Project Overview

EEG signals exhibit significant inter-subject variability, making it challenging for standard models to generalize well across different individuals. This project proposes **subject-specific layers** to isolate subject-unique parameters, potentially improving model personalization and balancing performance across subjects. The code includes:

- Baseline models such as **ShallowFBCSPNet** and **CollapsedShallowNet**.
- Subject-specific dictionary-based implementations (e.g., `SubjectDicionaryFCNet`).
- Training and evaluation scripts using **MOABB** and **Braindecode**.

---

## Installation


 **Install required packages**:

    *Important*: This code requires Python 3.10.11 (not sure if it works with any other versions, but this one should work).

   ```bash
   pip install -r requirements.txt
   ```

   This will install MOABB, Braindecode, MNE, PyTorch, and other dependencies needed for running the scripts.

---

## Usage


1. **Run the training script**:  
   Look at `MainRunTrainAndCompare.ipynb` or `MainRun.py`.
   Adjust paths and parameters (e.g., epochs, batch size, learning rate) as desired.

   By default, it will use MOABB’s MotorImagery paradigm and the BNCI2014\_001 dataset.

2. **Monitor training**:  
    The script logs training and validation metrics at each epoch. 

3. **Evaluate models**:  
   After training, the script automatically evaluates the trained models. Results are saved to CSV and visualized in PNG plots.
   For further analysis the `Training_Illustration.ipynb` notebook can be used.

---
