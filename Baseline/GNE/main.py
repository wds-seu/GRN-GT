from sklearn.linear_model import LogisticRegression
import LoadData as data
from evaluation import *
from GNE import GNE
import argparse
import os
import pandas as pd
import pickle


################################# Define parameters to train GNE model #######################################
parameters = {}
# Dimension of topological structure embeddings
parameters['id_embedding_size'] = 128
# Dimension of expression data embeddings
parameters['attr_embedding_size'] = 128
# Dimension of final representation after transformation of concatenated topological properties and expression data representation
parameters['representation_size'] = 128
# Importance of gene expression relative to topological properties
parameters['alpha'] = 1

# Number of negative samples for Negative Sampling
parameters['n_neg_samples'] = 10
# Number of epochs to run the model
parameters['epoch'] = 50
# Number of sample to consider in the batch
parameters['batch_size'] = 256
# Learning rate
parameters['learning_rate'] = 0.005

print(parameters)

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='STRING', help='network type')
parser.add_argument('--data', type=str, default='hESC', help='data type')
parser.add_argument('--num', type=int, default=500, help='network scale')

################################################################################################################


#################################### Define dataset and files ##################################################
# Define path
args = parser.parse_args()
path = os.getcwd() + '/train_validation_test/' + args.net + '/' + args.data + ' ' + str(args.num) + '/'

target_file = os.getcwd() + '/Dataset/' + args.net + ' Dataset/' + args.data + '/TFs+' + str(args.num) + '/Target.csv'

geneids = pd.read_csv(target_file)
num_genes = geneids.shape[0]

# Define the input to GNE model
feature_file = os.getcwd() + '/Dataset/' + args.net + ' Dataset/' + args.data + '/TFs+' + str(args.num) + "/BL--ExpressionData.csv"
################################################################################################################


################################# Load network and split to train and test######################################
# Perform train-test split
df = pd.read_csv(path + 'Train_set.csv')
train_dataset = [np.array(df['TF'].values), np.array(df['Target'].values), np.array(df['Label'].values)]
train_dataset = np.stack(train_dataset, axis=1)
train_dataset_pos = train_dataset[train_dataset[:, 2] == 1]
train_dataset_neg = train_dataset[train_dataset[:, 2] == 0]


df = pd.read_csv(path + 'Validation_set.csv')
val_dataset = [np.array(df['TF'].values), np.array(df['Target'].values), np.array(df['Label'].values)]
val_dataset = np.stack(val_dataset, axis=1)
val_labels = val_dataset[:, 2]


df = pd.read_csv(path + 'Test_set.csv')
test_dataset = [np.array(df['TF'].values), np.array(df['Target'].values), np.array(df['Label'].values)]
test_dataset = np.stack(test_dataset, axis=1)
test_labels = test_dataset[:, 2]
test_dataset_pos = test_dataset[test_dataset[:, 2] == 1]
test_dataset_neg = test_dataset[test_dataset[:, 2] == 0]

################################################################################################################

################## load interaction and expression data to fit GNE model and learn embeddings ##################
# load dataset to fit GNE model

# 只有正例
Data = data.LoadData(path, train_links=train_dataset_pos, features_file=feature_file)

# Define GNE model with data and parameters
model = GNE(path, Data, 2018, parameters)

# learn embeddings
embeddings, attr_embeddings = model.train(val_dataset, val_labels)

################################################################################################################


################## Create feature matrix and true labels for training and randomize the rows  ##################
# Train-set edge embeddings
pos_train_edge_embs = get_edge_embeddings(embeddings, train_dataset_pos)
neg_train_edge_embs = get_edge_embeddings(embeddings, train_dataset_neg)
train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

# Create train-set edge labels: 1 = real edge, 0 = false edge
train_edge_labels = np.concatenate([np.ones(len(train_dataset_pos)), np.zeros(len(train_dataset_neg))])


# Randomize train edges and labels
index = np.random.permutation([i for i in range(len(train_edge_labels))])
train_data = train_edge_embs[index, :]
train_labels = train_edge_labels[index]

################################################################################################################


################## Train the logistic regression on training data and predict on test dataset ##################
# Train logistic regression on train-set edge embeddings
edge_classifier = LogisticRegression(random_state=0)
edge_classifier.fit(train_data, train_labels)

# Test-set edge embeddings, labels
pos_test_edge_embs = get_edge_embeddings(embeddings, test_dataset_pos)
neg_test_edge_embs = get_edge_embeddings(embeddings, test_dataset_neg)
test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

# Randomize test edges and labels
index = np.random.permutation([i for i in range(len(test_labels))])
test_data = test_edge_embs[index, :]
test_labels = test_labels[index]

# Predict the probabilty for test edges by trained classifier
test_preds = edge_classifier.predict_proba(test_data)[:, 1]
test_roc = roc_auc_score(test_labels, test_preds)
test_ap = average_precision_score(test_labels, test_preds)

msg = "Alpha: {0:>6}, GNE Test ROC Score: {1:.4f}, GNE Test AP score: {2:.4f}"
print(msg.format(parameters['alpha'], test_roc, test_ap))

################################################################################################################


########################################## Save the embedding to a file ########################################

embeddings_file = open(path + "embeddings_trainsize_alpha_" + str(parameters['alpha']) + ".pkl", 'wb')
pickle.dump(embeddings, embeddings_file)
embeddings_file.close()

################################################################################################################
