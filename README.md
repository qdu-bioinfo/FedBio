# FedBio

## Content

- [Introduction](#introduction)
- [Package requirement](#package-requirement)
- [Installation environment](#installation-environment)
- [Model training and testing](#model-training-and-testing)
  - [a. Cohort-wise 5-fold split](#a-Cohort-wise 5-fold split)
  - [b. Centralized (pooled) 5-fold split](#b-Centralized (pooled) 5-fold split)
  - [c. FedBio (federated) model training ](#c-FedBio (federated) model training)
  - [d. Benchmark: local training per client (non-federated)](#d-Benchmark: local training per client (non-federated))
  - [e. Benchmark: centralized (pooled) setting](#e-Benchmark: centralized (pooled) setting)
- [Example](#Example)
    - [a. Running CTR vs. CRC analysis on WGS sequencing data.](#a-Running CTR vs. CRC analysis on WGS sequencing data.)
    - [b. Running CTR vs. ADA analysis on WGS sequencing data.](#b-Running CTR vs. ADA analysis on WGS sequencing data.)
    - [c. Running CTR vs. CRC analysis on 16S sequencing data.](#c-Running CTR vs. CRC analysis on 16S sequencing data.)
    - [d. Running CTR vs. ADA analysis on 16S sequencing data.](#d-Running CTR vs. CRC analysis on 16S sequencing data.)

## Introduction

FedBio, a privacy-preserving federated learning framework that enables collaborative, cross-institutional microbiome-based disease detection.

## Package requirement

* env.yml

## Installation environment

```
1.git clone https://github.com/qdu-bioinfo/FedBio.git
2.cd FedBio
3.conda env create -f env.yml
4.conda activate FedBio
```

###  	Download DistilBERT model files (first-time setup)

* On the **first run**, this project requires the pretrained **DistilBERT** model files:
   `distilbert/distilbert-base-uncased`

* Download link (Hugging Face):
    https://huggingface.co/distilbert/distilbert-base-uncased

* Please download the model from Hugging Face and place it into the following directory:↳

  ```
  FedBio_master/distilbert-base-uncased/After downloading, the folder should contain files such as:
  ```

  After downloading, the folder should contain files such as:

  * `config.json`
  * `model.safetensors`
  * `vocab.txt`
  * `tokenizer.json` 
  * `tokenizer_config.json`

  > **Important:** Keep the folder name exactly as `distilbert-base-uncased`, since the code loads the model from this local path.


## Model training and testing

### a. Cohort-wise 5-fold split
* Preserve cohort boundaries and avoid cross-cohort leakage. Each cohort is split **independently** into 5 folds.

```
python GeneratorData.py --data_type WGS --groups CTR_ADA
```
### b. Centralized 5-fold split

* Build a centralized baseline by first **merging all cohorts** into a single dataset and then performing a global 5-fold split.	

```
python GeneratorData_Central.py --data_type WGS --groups CTR_ADA
```

### c. FedBio model training

```
python FedBio.py --data_type WGS --groups CTR_ADA --num_clients 9
```

### d. Benchmark: local training per client (non-federated)

```
python BaseLine.py --data_type WGS --groups CTR_ADA --num_clients 9
```
### e. Benchmark: centralized (pooled) setting

```
python BaseLine_Central.py --data_type WGS --groups CTR_ADA --num_clients 9
```
## Example

### a. Running CTR vs. CRC analysis on WGS sequencing data.

```
python TestFedBio.py --data_type WGS --groups CTR_CRC --num_clients 12 --model_dir "FedBio_master/Models/WGS/CTR_CRC"
```

### b. Running CTR vs. ADA analysis on WGS sequencing data.

```
python TestFedBio.py --data_type WGS --groups CTR_ADA --num_clients 9 --model_dir "FedBio_master/Models/WGS/CTR_ADA"
```
### c. Running CTR vs. CRC analysis on 16S sequencing data.

```
python TestFedBio.py --data_type 16S --groups CTR_CRC --num_clients 6 --model_dir "FedBio_master/Models/16S/CTR_CRC"
```
### d. Running CTR vs. ADA analysis on 16S sequencing data.

```
python TestFedBio.py --data_type 16S --groups CTR_ADA --num_clients 5 --model_dir "FedBio_master/Models/16S/CTR_ADA"
```

