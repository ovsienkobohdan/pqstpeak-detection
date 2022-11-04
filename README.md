# PQST peaks detection

This repo is a companion for a [medium article](https://medium.com) (proper link will be added after publication).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ovsienkobohdan/pqstpeak-detection/blob/main/pt_detection.ipynb)

The goal of the "PQST peaks detection" project is to build a simple algorithm to detect P, Q, S, T points based on lead-1 ECG and annotated R-peaks. R peaks can be easily and accurately detected by known Pan-Tompkins algorithm. Q and S peaks are simply minimums in 60 milliseconds around R [R-60ms, R+60ms] so it's also not the main point of this project. The main point of this project is P and T detection.

## Python code for ease use will be added soon:
- [x] Add a simple Jupyter notebook
- [x] Write dataset preparation script
- [x] Save a clean dataset
- [x] Write a script to get scores on manual and auto peak detection 
- [x] Write a class for PT detection
- [x] Save PT Detections to a file
- [x] Write a script to get scores on PT deteciton algorithms
- [x] Write a class for QS peak detection
- [x] Combine classes to ECG peaks annotation
- [x] ISSUE: Compare Annotation from jupyter and script (scores are not maching) (R peaks detection is wrong or PT Detection?)
- [x] Window PT peak detection with adaptive ab
- [ ] Adaptive AB
- [ ] Regression adaptive