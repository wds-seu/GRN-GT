# GRN-GT

We propose a gene regulatory network link prediction model, **GRN-GT**, based on graph neural networks. The GRN-GT model uses cDNA sequence information and single-cell gene expression data as features, with prior gene regulatory network links as labels, to perform node representation learning for gene regulatory networks. This ultimately enables link prediction as a downstream task. Specifically, the cDNA sequences are encoded using the gene sequence encoder **DeepGene**, and then feature fusion is performed with pre-processed single-cell gene expression data to initialize the nodes. During the training process, the impact of the loss function and decoding strategy on model performance is considered, and experiments are conducted across a wide range of datasets. The GRN-GT model demonstrates optimal average performance across 44 downstream task datasets.

## 1. Environment setup

Please see `GRN-GT/requirements.txt`.

## 2. Dataset

The original data is in `GRN-GT/Dataset`.

After being divided into training set, validation set and test set, they are in the `GRN-GT/train_validation_test`.

The code used to split the data is in `Train_Test_Split`.

## 3. Run

Please see `GRN-GT/run_sh`.

Download the DeepGene model parameters using this [link](https://drive.google.com/file/d/168GNy3zA8aqlZ1Wq8mR6aHthF3CLmczV/view?usp=drive_link) and put them in the directory `GRN-GT/gene_emb_model_params` after downloading.
