import random
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas
from tqdm import tqdm #進度條, 用法: tqdm(iter)
import mynode2vec
import networkx as nx
from gensim.models import Word2Vec

def randomly_choose_false_edges(nodes, true_edges):
    tmp_list = list()
    all_edges = list()
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            all_edges.append((i, j))
    random.shuffle(all_edges)
    for edge in all_edges:
        if edge[0] == edge[1]:
            continue
        if (nodes[edge[0]], nodes[edge[1]]) not in true_edges and (nodes[edge[1]], nodes[edge[0]]) not in true_edges:
            tmp_list.append((nodes[edge[0]], nodes[edge[1]]))
        if len(tmp_list) >= L:
            return tmp_list
    return tmp_list

def divide_data(input_list, group_number):
    local_division = len(input_list) / float(group_number)
    random.shuffle(input_list)
    return [input_list[int(round(local_division * i)): int(round(local_division * (i + 1)))] for i in
            range(group_number)]

def get_G_from_edges(edges):
    edge_dict = dict()
    # calculate the count for all the edges
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.DiGraph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        # add edges to the graph
        tmp_G.add_edge(edge_key.split('_')[0], edge_key.split('_')[1])
        # add weights for all the edges
        tmp_G[edge_key.split('_')[0]][edge_key.split('_')[1]]['weight'] = weight
    return tmp_G

def get_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model.wv.syn0[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.syn0[local_model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return 0

def get_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)

directed = True
p = 0.5
q = 1
num_walks = 50
walk_length = 26
dimension = 200
window_size = 14
num_workers = 6
iterations = 70
number_of_groups = 5

train_edges = list()
raw_train_data = pandas.read_csv('train.csv')
for i, record in raw_train_data.iterrows():
    train_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the train data.')

# Start to load the valid/test data

valid_positive_edges = list()
valid_negative_edges = list()
raw_valid_data = pandas.read_csv('valid.csv')
for i, record in raw_valid_data.iterrows():
    if record['label']:
        valid_positive_edges.append((str(record['head']), str(record['tail'])))
    else:
        valid_negative_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the valid/test data.')


G = mynode2vec.Graph(get_G_from_edges(train_edges), directed, p, q)
# Calculate the probability for the random walk process
G.preprocess_transition_probs()
# Conduct the random walk process
walks = G.simulate_walks(num_walks, walk_length)
# Train the node embeddings with gensim word2vec package
model = Word2Vec(walks, size=dimension, window=window_size, min_count=0, sg=1, workers=num_workers, iter=iterations)
# Save the resulted embeddings (you can use any format you like)
resulted_embeddings = dict()
for i, w in enumerate(model.wv.index2word):
    resulted_embeddings[w] = model.wv.syn0[i]

# replace 'your_model' with your own model and use the provided evaluation code to evaluate.
tmp_AUC_score = get_AUC(model, valid_positive_edges, valid_negative_edges)
print('tmp_accuracy:', tmp_AUC_score)


raw_test = open("test.csv", "r")
output_done = model.wv
test_output = ""
for tt in raw_test:
    temp = tt.split(',')
    if temp[3]=='0':
        try:
            temp[3] = str(output_done.similarity(temp[1],temp[4][:-2]))
        except:
            a=1
    test_output += ','.join(temp)
a = open("test_output.csv","w")
a.write(test_output)


print('end')