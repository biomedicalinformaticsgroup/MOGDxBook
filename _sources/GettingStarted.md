# Getting Started

## Introduction
Multi-omic Graph Diagnosis (MOGDx) is a tool for the integration of omic data and classification of heterogeneous diseases. MOGDx exploits a Patient Similarity Network (PSN) framework to integrate omic data using Similarity Network Fusion (SNF) [^fn1]. A single PSN is built per modality. The PSN is built using the most informative features of that modality. The most informative features are found by performing a contrastive analysis between classification targets. For example, in mRNA, differentially expressed genes will be used to construct the PSN. Where suitable, Pearson correlation, otherwise Euclidean distance is measured between these informative features and the network is constructed using the K Nearest Neighbours (KNN) algorithm. SNF is used to combine individual PSN's into a single network. The fused PSN and the omic datasets are input into the Graph Convultional Network with Multi Modal Encoder, the architecture of which is shown in below. Each omic measure is compressed using a two layer encoder. The compressed encoded layer of each modality is then decoded to a shared latent space using mean pooling. This encourages each modality to learn the same latent space. The shared latent space is the node feature matrix, required for training the GCN, with each row forming a node feature vector. Classification is performed on the fused network using the Graph Convolutional Network (GCN) deep learning algorithm [^fn2]. 

GCN is a novel paradigm for learning from both network structure and node features. Heterogeneity in diseases confounds clinical trials, treatments, genetic association and more. Accurate stratification of these diseases is therefore critical to optimize treatment strategies for patients with heterogeneous diseases. Previous research has shown that accurate classification of heterogenous diseases has been achieved by integrating and classifying multi-omic data [^fn3][^fn4][^fn5]. MOGDx improves upon this research. The advantages of MOGDx is that it can handle both a variable number of modalities and missing patient data in one or more modalities. Performance of MOGDx was benchmarked on the BRCA TCGA dataset with competitive performance compared to its counterparts. In summary, MOGDx combines patient similarity network integration with graph neural network learning for accurate disease classification. 

## Workflow
### Full pipeline overview


### Pre-preocessing and Graph Generation


### Graph Convolutionl Network with Multi Modal Encoder


### Step 1 - Data Download
Create a folder called `data` and a folder for each dataset with naming convention 'TCGA-' e.g. 'TCGA-BRCA' inside this folder.

Use the R script `data_download.R` to download all data changing the project to BRCA/LGG/KICH/KIRC/KICH

## Installation
A working version of R and Python is required. R version 4.2.2 and Python version 3.9.2 was used to obtain all results.


### Step 5 - Execute MOGDx.py
Ensure all expression, meta and graph files for all modalities are in a single folder with naming convention 'modality_datExpr.csv'/'modality_datMeta.csv'/'modality_graph.csv' e.g. 'mRNA_datExpr.csv'/'mRNA_datMeta.csv'/'modality_graph.csv'. \
If performing an analysis on integrated modalities ensure all expression and meta files for the integrated modalities are in the folder. \
e.g. if analysing mRNA & miRNA, ensure mRNA_miRNA_graph.csv , datMeta_mRNA.csv, datMeta_miRNA.csv, datExpr_mRNA.csv and datExper_miRNA.csv are in the same folder. \

This process will have been done automatically by the creation of the raw folder and running of SNF.R and it is easiest to retain this folder.

MOGDx is a command line tool. A sample command is : \
`python MOGDx.py -i "/raw_data/raw_BRCA" -o "./Output/BRCA/"  -snf "mRNA_miRNA_graph.csv" --n-splits 5 -ld 32 16 --target "paper_BRCA_Subtype_PAM50" --index-col "patient" --epochs 2000 --lr 0.001 --h-feats 64 --decoder-dim 64`

-i is the location of the raw data containing all datExpr, datMeta and graph.csv files \
-o is the location where the output will be printed \
-snf is the name of the fused psn  \
--n-splits is the number of cross validation splits \
-ld is the latent dimension per modality. It is order alphabetically thus, mRNA will be compressed to dim of 32 and miRNA to 16  \
--target is the phenotype being classified \
--index-col is the column containing the patient ids  

All other arguments are related to the Graph Convolutional Network with Multi Modal Encoder

## Requirements
All requirements are specified in requirements.txt 

To create virtual env execute :  \
 `conda create --name <env> --file requirements.txt` 

## Citations
```{bibliography}
```
