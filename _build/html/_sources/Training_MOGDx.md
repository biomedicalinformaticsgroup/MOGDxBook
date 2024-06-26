---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# MOGDx Main Functions and Classes
We provide a description of the main functions and classes used to train a MOGDx model. 

## Graph Neural Network with Multi Modal Encoder (GNNMME)
```{image} ./images/gcn-mme.png
:alt: fishy
:width: 1200px
:align: left
```

### Multi Modal Encoder

### Graph Convolutional Network 
```{image} ./images/gcn.png
:alt: fishy
:width: 1200px
:align: left
```

### GraphSage
```{image} ./images/gsage.png
:alt: fishy
:width: 1200px
:align: left
```

## Training
Functions used to train and evaluate the MOGDx model. The training implemented follows that outlined by {cite:p}`hamilton_graph_2020`.

The list of functions are : 
- train
- evaluate
- confusion_matrix
- AUROC

## Utility
Utility functions used to parse input data, load networks from csv or perform utility tasks. 

The list of functions are : 
- data_parsing
- get_gpu_memory
- indices_removal_adjust
- network_from_csv

## Citations
```{bibliography}
```
