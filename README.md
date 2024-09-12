# CodeXplain
This repo is a paper of python implementation : From Cryptic to Clear - Training on LLM Explanations to Detect Smart Contract Vulnerabilities

# Task Definition
Automated detection of smart contract vulnerabilities.

# Framework

![The overview of CodeXplain](figs/model.drawio.png)

The overview of our proposed method CodeXplain is illustrated in the Figure, which consists of three modules: 1) Multi-perspective Code Explanation Generation by LLM, and 2) Semantic Fusion for Code and Explanations.

# Datasets
We use the same dataset as [Ma et al., 202](https://sites.google.com/view/iaudittool/home). Further instructions on the dataset can be found on [Smart-Contract-Dataset](https://drive.google.com/drive/folders/1cAHxSu6dL3S21zz2iaQzSTABfSjY2vP8).

# Running
To run program, please use this command: python `run.py`.

Also all the hyper-parameters can be found in `run.py`.

Examples:

`
python run.py --epoch 50  --batch_szie 16 --max_length 1024
`
