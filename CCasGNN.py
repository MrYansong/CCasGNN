#encoding: utf-8

import torch
import json

import numpy as np
import copy
import time
import sys
import math
from layers import Positional_GCN, MultiHeadGraphAttention, dens_Net, Positional_GAT, fuse_gate
import scipy.stats as sci

class CCasGNN(torch.nn.Module):
    def __init__(self, args):
        super(CCasGNN, self).__init__()
        self.args = args
        self.number_of_features = self.args.number_of_features
        self.number_of_nodes = self.args.number_of_nodes
        self._setup_layers()

    def _setup_GCN_layers(self):
        self.GCN_layers = Positional_GCN(in_channels=self.args.user_embedding_dim + self.args.location_embedding_dim,
                                 out_channels=self.args.gcn_out_channel,
                                 location_embedding_dim=self.args.location_embedding_dim,
                                 filters_1=self.args.gcn_filters_1,
                                 filters_2=self.args.gcn_filters_2,
                                 dropout=self.args.gcn_dropout)  #self.args.user_embedding_dim

    def _setup_GAT_layers(self):
        self.GAT_layers = Positional_GAT(in_channels=self.args.user_embedding_dim + self.args.location_embedding_dim,
                                 out_channels=self.args.gcn_out_channel,
                                 n_heads=self.args.gat_n_heads,
                                 location_embedding_dim=self.args.location_embedding_dim,
                                 filters_1=self.args.gcn_filters_1,
                                 filters_2=self.args.gcn_filters_2,
                                 dropout=self.args.gcn_dropout)  #self.number_of_features + self.args.location_embedding_dim

    def _setup_MultiHead_att_layers(self):
        self.MultiHead_att_layers = MultiHeadGraphAttention(num_heads=self.args.att_num_heads,
                                                            dim_in=self.args.gcn_out_channel,
                                                            dim_k=self.args.att_dim_k,
                                                            dim_v=self.args.att_dim_v)

    def _setup_dens_layers (self):
        self.dens_layers = dens_Net(dens_inputsize=self.args.gcn_out_channel,
                                    dens_hiddensize=self.args.dens_hiddensize,
                                    dens_dropout=self.args.dens_dropout,
                                    dens_outputsize=self.args.dens_outsize
                                    ) #self.args.attn_out_dim
    def _setup_fuse_layers(self):
        self.fuse_layers = fuse_gate(batch_size=1,
                                     in_dim=2)
    def _setup_layers(self):
        self._setup_GCN_layers()
        self._setup_MultiHead_att_layers()
        self._setup_dens_layers()
        self._setup_GAT_layers()
        self._setup_fuse_layers()

    def forward(self,data):
        true_nodes_num = data["true_nodes_num"]
        features = data['features'][:true_nodes_num]
        edges = data['edges']
        undirected_edges = data['undirected_edges']
        location_embedding = data['location_embedding'][:true_nodes_num]
        GAT_representation = torch.nn.functional.relu(self.GAT_layers(edges, features, location_embedding))
        GAT_representation = GAT_representation[:true_nodes_num]    #nodes_num * feature_num
        GAT_representation = torch.mean(GAT_representation, dim=0, keepdim=False)

        GCN_representation = torch.nn.functional.relu(self.GCN_layers(undirected_edges, features, location_embedding))
        GCN_representation = GCN_representation[:true_nodes_num]    #nodes_num * feature_num
        GCN_representation = GCN_representation.unsqueeze(dim=0)  #batch_size * nodes_num * feature_num
        #
        GCN_att_representation = self.MultiHead_att_layers(GCN_representation)   #batch_size * nodes_num * feature_num
        GCN_att_representation = torch.mean(GCN_att_representation, dim=1, keepdim=False)  #batch_size * feature_num
        GCN_squeeze_att_representation = GCN_att_representation.squeeze(dim=0)

        GAT_pred, GCN_pred = self.dens_layers(GAT_representation, GCN_squeeze_att_representation)

        model_predict = torch.cat((GAT_pred, GCN_pred), dim=0)
        prediction, omega1, omega2 = self.fuse_layers(model_predict)

        return prediction, omega1, omega2, GAT_pred, GCN_pred


class CCasGNN_Trainer(torch.nn.Module):
    def __init__(self, args):
        super(CCasGNN_Trainer, self).__init__()
        self.args = args
        self.setup_model()

    def setup_model(self):
        self.load_graph_data()
        self.model = CCasGNN(self.args)

    def load_graph_data(self):
        self.number_of_nodes = self.args.number_of_nodes
        self.number_of_features = self.args.number_of_features
        self.graph_data = json.load(open(self.args.graph_file_path, 'r'))
        N = len(self.graph_data)    #the number of graphs
        train_start, valid_start, test_start = \
            0, int(N * self.args.train_ratio), int(N * (self.args.train_ratio + self.args.valid_ratio))
        train_graph_data = self.graph_data[0:valid_start]   #list type [dict,dict,...]
        valid_graph_data = self.graph_data[valid_start:test_start]
        test_graph_data = self.graph_data[test_start:N]
        self.train_batches, self.valid_batches, self.test_batches = [], [], []
        for i in range(0, len(train_graph_data), self.args.batch_size):
            self.train_batches.append(train_graph_data[i:i+self.args.batch_size])
        for j in range(0, len(valid_graph_data), self.args.batch_size):
            self.valid_batches.append(valid_graph_data[j:j+self.args.batch_size])
        for k in range(0, len(test_graph_data), self.args.batch_size):
            self.test_batches.append(test_graph_data[k:k+self.args.batch_size])

    def create_edges(self,data):
        """
        create an Edge matrix
        :param data:
        :return: Edge matrix
        """
        self.nodes_map = [str(nodes_id) for nodes_id in data['nodes']]
        self.true_nodes_num = len(data['nodes'])
        edges = [[self.nodes_map.index(str(edge[0])), self.nodes_map.index(str(edge[1]))] for edge in data['edges']]
        undirected_edges = edges + [[self.nodes_map.index(str(edge[1])), self.nodes_map.index(str(edge[0]))] for edge in data['edges']]
        return torch.t(torch.LongTensor(edges)), torch.t(torch.LongTensor(undirected_edges))

    def create_location_embedding(self, omega=0.001):
        location_dim = self.args.location_embedding_dim
        location_emb = torch.zeros(self.number_of_nodes, location_dim)
        for i in range(self.number_of_nodes):
            for j in range(location_dim):
                if j % 2 == 0:
                    location_emb[i][j] = math.sin(i * math.pow(omega, j / location_dim))
                else:
                    location_emb[i][j] = math.cos(i * math.pow(omega, (j - 1) / location_dim))
        return location_emb

    def create_target(self,data):
        return torch.tensor([data['activated_size']])

    def create_features(self,data):
        features = np.zeros((self.number_of_nodes, self.args.user_embedding_dim))
        # features = np.zeros((self.number_of_nodes, self.number_of_features))
        for nodes_id in data['nodes']:
            features[self.nodes_map.index(str(nodes_id))][:self.args.user_embedding_dim] = data['nodes_embedding'][str(nodes_id)]
            # features[self.nodes_map.index(str(nodes_id))][self.args.user_embedding_dim:] = data['location_embedding'][str(nodes_id)]
        features = torch.FloatTensor(features)
        return features

    def create_user_embedding(self,data):
        user_embedding = np.zeros((self.number_of_nodes, self.args.user_embedding_dim))
        for nodes_id in data['nodes']:
            user_embedding[self.nodes_map.index(str(nodes_id))] = data['nodes_embedding'][str(nodes_id)]
        return torch.FloatTensor(user_embedding)

    def create_input_data(self, data):
        """
        :param data: one data in the train/valid/test graph data
        :return: to_pass_forward: Data dictionary
        """
        to_pass_forward = dict()
        activated_size = self.create_target(data)
        edges, undirected_edges = self.create_edges(data)
        features = self.create_features(data)
        user_embedding = self.create_user_embedding(data)
        location_embedding = self.create_location_embedding(omega=0.001)
        to_pass_forward["edges"] = edges
        to_pass_forward["undirected_edges"] = undirected_edges
        to_pass_forward["features"] = features
        to_pass_forward["user_embedding"] = user_embedding
        to_pass_forward["true_nodes_num"] = self.true_nodes_num
        to_pass_forward['location_embedding'] = location_embedding
        return to_pass_forward, activated_size

    def create_forward_data(self, data_batches):
        data_x, data_y = [], []
        for data_batch in data_batches:
            data_x_tmp, data_y_tmp = [], []
            for each_data in data_batch:
                input_data, target = self.create_input_data(each_data)
                data_x_tmp.append(input_data)
                data_y_tmp.append(target)
            data_x.append(copy.deepcopy(data_x_tmp))
            data_y.append(copy.deepcopy(data_y_tmp))
        return data_x, data_y

    def fit(self):
        print('\nLoading data.\n')
        self.model.train()
        train_data_x, train_data_y = self.create_forward_data(self.train_batches)
        valid_data_x, valid_data_y = self.create_forward_data(self.valid_batches)
        test_data_x, test_data_y = self.create_forward_data(self.test_batches)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        time_start = time.time()
        print('\nTraining started.\n')
        for epoch in range(self.args.epochs):
            losses = 0.
            average_loss = 0.
            for step, (train_x_batch, train_y_batch) in enumerate(zip(train_data_x, train_data_y)):
                optimizer.zero_grad()
                GAT_prediction_tensor = torch.tensor([])
                GCN_prediction_tensor = torch.tensor([])
                target_tensor = torch.tensor([])
                for k, (train_x, train_y) in enumerate(zip(train_x_batch, train_y_batch)):
                    prediction = self.model(train_x)
                    GAT_prediction_tensor = torch.cat((GAT_prediction_tensor, prediction[3].float()), 0)
                    GCN_prediction_tensor = torch.cat((GCN_prediction_tensor, prediction[4].float()), 0)
                    target_tensor = torch.cat((target_tensor, torch.log2(train_y.float() + 1)), 0)
                omega1 = prediction[1].data.float()
                omega2 = prediction[2].data.float()
                GAT_loss = torch.nn.functional.mse_loss(target_tensor,GAT_prediction_tensor)
                GCN_loss = torch.nn.functional.mse_loss(target_tensor,GCN_prediction_tensor)
                loss = omega1 * GAT_loss + omega2 * GCN_loss
                loss.backward()

                optimizer.step()
                losses = losses + loss.item()
                average_loss = losses / (step + 1)
            print('CCasGNN train MSLE loss in ', epoch + 1, ' epoch = ', average_loss)
            time_now = time.time()
            print('the rest of running time about:', (((time_now-time_start)/ (epoch+1)) * (self.args.epochs - epoch)) / 60, ' minutes')
            print('\n')

            if (epoch + 1) % self.args.check_point == 0:
                print('epoch ',epoch + 1, ' evaluating.')
                self.evaluation(valid_data_x, valid_data_y)
                self.test(test_data_x, test_data_y)

    def evaluation(self, valid_x_batches, valid_y_batches):
        self.model.eval()
        losses = 0.
        average_loss = 0.
        for step, (valid_x_batch, valid_y_batch) in enumerate(zip(valid_x_batches, valid_y_batches)):
            loss = 0.
            prediction_tensor = torch.tensor([])
            target_tensor = torch.tensor([])
            for (valid_x, valid_y) in zip(valid_x_batch, valid_y_batch):
                prediction = self.model(valid_x)
                prediction_tensor = torch.cat((prediction_tensor, prediction[0].float()), 0)
                target_tensor = torch.cat((target_tensor, torch.log2(valid_y.float() + 1)), 0)
            loss = torch.nn.functional.mse_loss(target_tensor, prediction_tensor)
            losses = losses + loss.item()
            average_loss = losses / (step + 1)
        print('#####CCasGNN valid MSLE loss in this epoch = ', average_loss)
        print('\n')

    def test(self, test_x_batches, test_y_batches):
        print("\n\nScoring.\n")
        self.model.eval()
        losses = 0.
        average_loss = 0.
        all_test_tensor = torch.tensor([])
        all_true_tensor = torch.tensor([])
        for step, (test_x_batch, test_y_batch) in enumerate(zip(test_x_batches, test_y_batches)):
            loss = 0.
            prediction_tensor = torch.tensor([])
            target_tensor = torch.tensor([])
            for (test_x, test_y) in zip(test_x_batch, test_y_batch):
                prediction = self.model(test_x)
                prediction_tensor = torch.cat((prediction_tensor, prediction[0].float()), 0)
                all_test_tensor = torch.cat((all_test_tensor, prediction[0].float()), dim=0)
                target_tensor = torch.cat((target_tensor, torch.log2(test_y.float() + 1)), 0)
                all_true_tensor = torch.cat((all_true_tensor, torch.log2(test_y.float() + 1)), dim=0)
            loss = torch.nn.functional.mse_loss(target_tensor, prediction_tensor)
            losses = losses + loss.item()
            average_loss = losses / (step + 1)
        all_test_np = all_test_tensor.detach().numpy()
        all_true_np = all_true_tensor.detach().numpy()
        sub_np = all_test_np - all_true_np
        print('correlation: ', sci.pearsonr(sub_np, all_true_np))
        print('#####CCasGNN test MSLE loss = ', average_loss)
        print('\n')
