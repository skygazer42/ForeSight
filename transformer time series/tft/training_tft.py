# -*- coding: utf-8 -*-
# Converted from: transformer time series/tft/training_tft.ipynb

# %%
import pandas as pd
import numpy as np
import pyunpack
import math
import json

from data.data_download import Config, download_electricity
from data_formatters.electricity import ElectricityFormatter
from data_formatters.base import DataTypes, InputTypes

from data.custom_dataset import TFTDataset
from models import GatedLinearUnit
from models import GateAddNormNetwork
from models import GatedResidualNetwork 
from models import ScaledDotProductAttention
from models import InterpretableMultiHeadAttention
from models import VariableSelectionNetwork
from models import ManualLSTM

from quantile_loss import QuantileLossCalculator
from quantile_loss import NormalizedQuantileLossCalculator

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from argparse import ArgumentParser

import matplotlib.pyplot as plt

# %% [markdown]
# ## Download Data

# %%
config = Config('data','data/electricity.csv')

# %%
download_electricity(config)

# %% [markdown]
# ## Creating Datasets

# %%
electricity = pd.read_csv('data/electricity.csv', index_col = 0)
data_formatter = ElectricityFormatter()
train, valid, test = data_formatter.split_data(electricity)

# %%
train.shape, valid.shape, test.shape

# %%
train.days_from_start.value_counts().to_frame().reset_index().sort_values(by=['index'])

# %%
valid.days_from_start.value_counts().to_frame().reset_index().sort_values(by=['index'])

# %%
test.days_from_start.value_counts().to_frame().reset_index().sort_values(by=['index'])

# %% [markdown]
# ### Reviewing Test dataset error

# %%
test = test.reset_index(drop=True)
test

# %%
test[test.categorical_id == 0]

# %%
test.groupby(['categorical_id']).apply(lambda x: x.shape[0]).mean()

# %%
g = test.groupby(['categorical_id'])

# %%
data_formatter.get_time_steps()

# %%
df_index_abs = g[['categorical_id']].transform(lambda x: x.index+data_formatter.get_time_steps()) \
                        .reset_index() \
                        .rename(columns={'index':'init_abs',
                                         'categorical_id':'end_abs'})
df_index_abs

# %%
df_index_rel_init = g[['categorical_id']].transform(lambda x: x.reset_index(drop=True).index) \
                        .rename(columns={'categorical_id':'init_rel'})
df_index_rel_init

# %%
df_index_rel_end = g[['categorical_id']].transform(lambda x: x.reset_index(drop=True).index+data_formatter.get_time_steps()) \
                .rename(columns={'categorical_id':'end_rel'})
df_index_rel_end

# %%
336 - 192 + 1

# %%
df_total_count = g[['categorical_id']].transform(lambda x: x.shape[0] - data_formatter.get_time_steps() + 1) \
                .rename(columns = {'categorical_id':'group_count'})
df_total_count

# %%
new_test = pd.concat([df_index_abs, 
                       df_index_rel_init,
                       df_index_rel_end,
                       test[['id']], 
                       df_total_count], axis = 1).reset_index(drop = True)
new_test

# %%
new_test[new_test.end_rel < test.groupby(['categorical_id']).apply(lambda x: x.shape[0]).mean()].reset_index()

# %% [markdown]
# ## Loading Datasets

# %%
train_dataset = TFTDataset(train)
valid_dataset = TFTDataset(valid)
test_dataset = TFTDataset(test)

# %%
len(train_dataset), len(valid_dataset), len(test_dataset)

# %%
len(train_dataset), len(valid_dataset), len(test_dataset)

# %%
test_dataset[0][0].shape, test_dataset[0][1].shape, test_dataset[0][2].shape

# %% [markdown]
# ## Temporal Fusion Transformer

# %%
class TemporalFusionTransformer(pl.LightningModule):
    def __init__(self, hparams):
        super(TemporalFusionTransformer, self).__init__()
        
        self.hparams = hparams
        
        self.name = self.__class__.__name__

        # Data parameters
        self.time_steps = int(hparams.total_time_steps)#int(params['total_time_steps'])
        self.input_size = int(hparams.input_size)#int(params['input_size'])
        self.output_size = int(hparams.output_size)#int(params['output_size'])
        self.category_counts = json.loads(str(hparams.category_counts))#json.loads(str(params['category_counts']))
        self.num_categorical_variables = len(self.category_counts)
        self.num_regular_variables = self.input_size - self.num_categorical_variables
        self.n_multiprocessing_workers = int(hparams.multiprocessing_workers) #int(params['multiprocessing_workers'])

        # Relevant indices for TFT
        self._input_obs_loc = json.loads(str(hparams.input_obs_loc))#json.loads(str(params['input_obs_loc']))
        self._static_input_loc = json.loads(str(hparams.static_input_loc))#json.loads(str(params['static_input_loc']))
        self._known_regular_input_idx = json.loads(str(hparams.known_regular_inputs))#json.loads(str(params['known_regular_inputs']))
        self._known_categorical_input_idx = json.loads(str(hparams.known_categorical_inputs))#json.loads(str(params['known_categorical_inputs']))
        
        self.num_non_static_historical_inputs = self.get_historical_num_inputs()
        self.num_non_static_future_inputs = self.get_future_num_inputs()
        
        self.column_definition = [
                                  ('id', DataTypes.REAL_VALUED, InputTypes.ID),
                                  ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
                                  ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
                                  ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                                  ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                                  ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                                  ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
                                ]

        # Network params
        self.quantiles = [0.1, 0.5, 0.9]
#         self.use_cudnn = use_cudnn  # Whether to use GPU optimised LSTM
        self.hidden_layer_size = int(hparams.hidden_layer_size)#int(params['hidden_layer_size'])
        self.dropout_rate = float(hparams.dropout_rate)#float(params['dropout_rate'])
        self.max_gradient_norm = float(hparams.max_gradient_norm)#float(params['max_gradient_norm'])
        self.learning_rate = float(hparams.learning_rate)#float(params['learning_rate'])
        self.minibatch_size = int(hparams.minibatch_size)#int(params['minibatch_size'])
        self.num_epochs = int(hparams.num_epochs)#int(params['num_epochs'])
        self.early_stopping_patience = int(hparams.early_stopping_patience)#int(params['early_stopping_patience'])

        self.num_encoder_steps = int(hparams.num_encoder_steps)#int(params['num_encoder_steps'])
        self.num_stacks = int(hparams.stack_size)#int(params['stack_size'])
        self.num_heads = int(hparams.num_heads)#int(params['num_heads'])

        # Serialisation options
#         self._temp_folder = os.path.join(params['model_folder'], 'tmp')
#         self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        print('*** {} params ***'.format(self.name))
        for k in vars(hparams):
            print('# {} = {}'.format(k, vars(hparams)[k]))
            
        self.train_criterion = QuantileLossCalculator(self.quantiles, self.output_size)
        self.test_criterion = NormalizedQuantileLossCalculator(self.quantiles, self.output_size)

        # Build model
        ## Build embeddings
        self.build_embeddings()
        
        ## Build Static Contex Networks
        self.build_static_context_networks()
        
        ## Building Variable Selection Networks
        self.build_variable_selection_networks()
        
        ## Build Lstm
        self.build_lstm()
        
        ## Build GLU for after lstm encoder decoder and layernorm
        self.build_post_lstm_gate_add_norm()
        
        ## Build Static Enrichment Layer
        self.build_static_enrichment()
        
        ## Building decoder multihead attention
        self.build_temporal_self_attention()
        
        ## Building positionwise decoder
        self.build_position_wise_feed_forward()
        
        ## Build output feed forward
        self.build_output_feed_forward()
        
        ## Initializing remaining weights
        self.init_weights()
        
    def init_weights(self):
        for name, p in self.named_parameters():
            if ('lstm' in name and 'ih' in name) and 'bias' not in name:
                #print(name)
                #print(p.shape)
                torch.nn.init.xavier_uniform_(p)
#                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif ('lstm' in name and 'hh' in name) and 'bias' not in name:
        
                 torch.nn.init.orthogonal_(p)
            
            elif 'lstm' in name and 'bias' in name:
                #print(name)
                #print(p.shape)
                torch.nn.init.zeros_(p)
        
    def get_historical_num_inputs(self):
        
        obs_inputs = [i for i in self._input_obs_loc]
        
        known_regular_inputs = [i for i in self._known_regular_input_idx
                                if i not in self._static_input_loc]
            
        known_categorical_inputs = [i for i in self._known_categorical_input_idx
                                    if i + self.num_regular_variables not in self._static_input_loc]
        
        wired_embeddings = [i for i in range(self.num_categorical_variables)
                            if i not in self._known_categorical_input_idx 
                            and i not in self._input_obs_loc]

        unknown_inputs = [i for i in range(self.num_regular_variables)
                          if i not in self._known_regular_input_idx
                          and i not in self._input_obs_loc]

        return len(obs_inputs+known_regular_inputs+known_categorical_inputs+wired_embeddings+unknown_inputs)
    
    def get_future_num_inputs(self):
            
        known_regular_inputs = [i for i in self._known_regular_input_idx
                                if i not in self._static_input_loc]
            
        known_categorical_inputs = [i for i in self._known_categorical_input_idx
                                    if i + self.num_regular_variables not in self._static_input_loc]

        return len(known_regular_inputs + known_categorical_inputs)
    
    def build_embeddings(self):
        self.categorical_var_embeddings = nn.ModuleList([nn.Embedding(self.category_counts[i], 
                                                                      self.hidden_layer_size) 
                                                     for i in range(self.num_categorical_variables)])

        self.regular_var_embeddings = nn.ModuleList([nn.Linear(1, 
                                                              self.hidden_layer_size) 
                                                  for i in range(self.num_regular_variables)])

    def build_variable_selection_networks(self):
        
        self.static_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,
                                                   input_size = self.hidden_layer_size * len(self._static_input_loc),
                                                   output_size = len(self._static_input_loc),
                                                   dropout_rate = self.dropout_rate)
        
        self.temporal_historical_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,
                                                                input_size = self.hidden_layer_size *
                                                                        self.num_non_static_historical_inputs,
                                                                output_size = self.num_non_static_historical_inputs,
                                                                dropout_rate = self.dropout_rate,
                                                                additional_context=self.hidden_layer_size)
        
        self.temporal_future_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,
                                                            input_size = self.hidden_layer_size *
                                                                        self.num_non_static_future_inputs,
                                                            output_size = self.num_non_static_future_inputs,
                                                            dropout_rate = self.dropout_rate,
                                                            additional_context=self.hidden_layer_size)
        
    def build_static_context_networks(self):
        
        self.static_context_variable_selection_grn = GatedResidualNetwork(self.hidden_layer_size,
                                                                          dropout_rate=self.dropout_rate)
        
        self.static_context_enrichment_grn = GatedResidualNetwork(self.hidden_layer_size,
                                                              dropout_rate=self.dropout_rate)

        self.static_context_state_h_grn = GatedResidualNetwork(self.hidden_layer_size,
                                                           dropout_rate=self.dropout_rate)
        
        self.static_context_state_c_grn = GatedResidualNetwork(self.hidden_layer_size,
                                                           dropout_rate=self.dropout_rate)
        
    def build_lstm(self):
        self.historical_lstm = ManualLSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
        )
        self.future_lstm = ManualLSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
        )
        
    def build_post_lstm_gate_add_norm(self):
        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                                 self.hidden_layer_size,
                                                                 self.dropout_rate,
                                                                 activation = None)
        
    def build_static_enrichment(self):
        self.static_enrichment = GatedResidualNetwork(self.hidden_layer_size,
                                                      dropout_rate = self.dropout_rate,
                                                      additional_context=self.hidden_layer_size)
        
    def build_temporal_self_attention(self):
        self.self_attn_layer = InterpretableMultiHeadAttention(n_head = self.num_heads, 
                                                               d_model = self.hidden_layer_size,
                                                               dropout = self.dropout_rate)
        
        self.post_attn_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                           self.hidden_layer_size,
                                                           self.dropout_rate,
                                                           activation = None)
        
    def build_position_wise_feed_forward(self):
        self.GRN_positionwise = GatedResidualNetwork(self.hidden_layer_size,
                                                     dropout_rate = self.dropout_rate)
        
        self.post_tfd_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                         self.hidden_layer_size,
                                                         self.dropout_rate,
                                                         activation = None)
        
    def build_output_feed_forward(self):
        self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size, 
                                                   self.output_size * len(self.quantiles))
         
    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.
        Args:
        self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1]
        bs = self_attn_inputs.shape[0]
        mask = torch.cumsum(torch.eye(len_s), 0)
        mask = mask.repeat(bs,1,1).to(torch.float32)

        return mask.to(DEVICE)
    
    def get_tft_embeddings(self, regular_inputs, categorical_inputs):
        # Static input
        if self._static_input_loc:
            static_regular_inputs = [self.regular_var_embeddings[i](regular_inputs[:, 0, i:i + 1]) 
                                    for i in range(self.num_regular_variables)
                                    if i in self._static_input_loc]
            #print('static_regular_inputs')
            #print([print(emb.shape) for emb in static_regular_inputs])
            
            static_categorical_inputs = [self.categorical_var_embeddings[i](categorical_inputs[Ellipsis, i])[:,0,:] 
                                         for i in range(self.num_categorical_variables)
                                         if i + self.num_regular_variables in self._static_input_loc]
            #print('static_categorical_inputs')
            #print([print(emb.shape) for emb in static_categorical_inputs])
            static_inputs = torch.stack(static_regular_inputs + static_categorical_inputs, axis = 1)
        else:
            static_inputs = None
            
        # Target input
        obs_inputs = torch.stack([self.regular_var_embeddings[i](regular_inputs[Ellipsis, i:i + 1])
                                     for i in self._input_obs_loc], axis=-1)
        
        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(self.num_categorical_variables):
            if i not in self._known_categorical_input_idx \
            and i not in self._input_obs_loc:
                e = self.categorical_var_embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(self.num_regular_variables):
            if i not in self._known_regular_input_idx \
            and i not in self._input_obs_loc:
                e = self.regular_var_embeddings[i](regular_inputs[Ellipsis, i:i + 1])
                unknown_inputs.append(e)
                
        if unknown_inputs + wired_embeddings:
            unknown_inputs = torch.stack(unknown_inputs + wired_embeddings, axis=-1)
        else:
            unknown_inputs = None
            
        # A priori known inputs
        known_regular_inputs = [self.regular_var_embeddings[i](regular_inputs[Ellipsis, i:i + 1])
                                for i in self._known_regular_input_idx
                                if i not in self._static_input_loc]
        #print('known_regular_inputs')
        #print([print(emb.shape) for emb in known_regular_inputs])
        
        known_categorical_inputs = [self.categorical_var_embeddings[i](categorical_inputs[Ellipsis, i])
                                    for i in self._known_categorical_input_idx
                                    if i + self.num_regular_variables not in self._static_input_loc]
       #print('known_categorical_inputs')
       #print([print(emb.shape) for emb in known_categorical_inputs])

        known_combined_layer = torch.stack(known_regular_inputs + known_categorical_inputs, axis=-1)
        
        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs
        
    def forward(self, all_inputs):

        regular_inputs = all_inputs[:, :, :self.num_regular_variables].to(torch.float)
        #print('regular_inputs')
        #print(regular_inputs.shape)
        categorical_inputs = all_inputs[:, :, self.num_regular_variables:].to(torch.long)
        #print('categorical_inputs')
        #print(categorical_inputs.shape)
        
        unknown_inputs, known_combined_layer, obs_inputs, static_inputs \
            = self.get_tft_embeddings(regular_inputs, categorical_inputs)
        
        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
              historical_inputs = torch.cat([
                  unknown_inputs[:, :self.num_encoder_steps, :],
                  known_combined_layer[:, :self.num_encoder_steps, :],
                  obs_inputs[:, :self.num_encoder_steps, :]
              ], axis=-1)
        else:
              historical_inputs = torch.cat([
                  known_combined_layer[:, :self.num_encoder_steps, :],
                  obs_inputs[:, :self.num_encoder_steps, :]
              ], axis=-1)
                
        #print('historical_inputs')
        #print(historical_inputs.shape)
        
        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, self.num_encoder_steps:, :]
        #print('future_inputs')
        #print(future_inputs.shape)
              
        #print('static_inputs')
        #print(static_inputs.shape)
        
        static_encoder, sparse_weights = self.static_vsn(static_inputs)
        
        #print('static_encoder')
        #print(static_encoder.shape)
        
        #print('sparse_weights')
        #print(sparse_weights.shape)
        
        static_context_variable_selection = self.static_context_variable_selection_grn(static_encoder)
        #print('static_context_variable_selection')
        #print(static_context_variable_selection.shape)
        static_context_enrichment = self.static_context_enrichment_grn(static_encoder)
        #print('static_context_enrichment')
        #print(static_context_enrichment.shape)
        static_context_state_h = self.static_context_state_h_grn(static_encoder)
        #print('static_context_state_h')
        #print(static_context_state_h.shape)
        static_context_state_c = self.static_context_state_c_grn(static_encoder)
        #print('static_context_state_c')
        #print(static_context_state_c.shape)
        
        historical_features, historical_flags \
        = self.temporal_historical_vsn((historical_inputs,
                                        static_context_variable_selection))
        #print('historical_features')
        #print(historical_features.shape)
        #print('historical_flags')
        #print(historical_flags.shape)
        
        future_features, future_flags \
        = self.temporal_future_vsn((future_inputs,
                                    static_context_variable_selection))
        #print('future_features')
        #print(future_features.shape)
        #print('future_flags')
        #print(future_flags.shape)
        
        history_lstm, (state_h, state_c) \
        = self.historical_lstm(historical_features,
                               (static_context_state_h.unsqueeze(0),
                                static_context_state_c.unsqueeze(0)))
        #print('history_lstm')
        #print(history_lstm.shape)
        #print('state_h')
        #print(state_h.shape)
        #print('state_c')
        #print(state_c.shape)
        
        future_lstm, _ = self.future_lstm(future_features,
                                          (state_h,
                                           state_c))
        #print('future_lstm')
        #print(future_lstm.shape)
        
        # Apply gated skip connection
        input_embeddings = torch.cat((historical_features, future_features), axis=1)
        #print('input_embeddings')
        #print(input_embeddings.shape) 
        
        lstm_layer = torch.cat((history_lstm, future_lstm), axis=1)
        #print('lstm_layer')
        #print(lstm_layer.shape) 
        
        temporal_feature_layer = self.post_seq_encoder_gate_add_norm(lstm_layer, input_embeddings)
        #print('temporal_feature_layer')
        #print(temporal_feature_layer.shape)  
        
        # Static enrichment layers
        expanded_static_context = static_context_enrichment.unsqueeze(1)
        
        enriched = self.static_enrichment((temporal_feature_layer, expanded_static_context))
        #print('enriched')
        #print(enriched.shape)    
        
        # Decoder self attention
        #self.mask = self.get_decoder_mask(enriched)
        #print('enriched')
        #print(enriched.shape)
        x, self_att = self.self_attn_layer(enriched, 
                                           enriched, 
                                           enriched,
                                           mask = self.get_decoder_mask(enriched))
        #print('x')
        #print(x.shape)
        #print('self_att')
        #print(self_att.shape)
        
        x = self.post_attn_gate_add_norm(x, enriched)
        #print('x')
        #print(x.shape)
        
        # Nonlinear processing on outputs
        decoder = self.GRN_positionwise(x)
        #print('decoder')
        #print(decoder.shape)
        
        # Final skip connection
        transformer_layer = self.post_tfd_gate_add_norm(decoder, temporal_feature_layer)
        #print('transformer_layer')
        #print(transformer_layer.shape)
        
        outputs = self.output_feed_forward(transformer_layer[Ellipsis, self.num_encoder_steps:, :])
        #print('outputs')
        #print(outputs.shape)
        
        #ipdb.set_trace()
        
        return outputs
    
    def loss(self, y_hat, y):
        return self.train_criterion.apply(y_hat, y)
    
    def test_loss(self, y_hat, y):
        return self.test_criterion.apply(y_hat, y, self.quantiles[1])
    
    def training_step(self, batch, batch_nb):
        x, y, _ = batch
        
        x = x.to(torch.float)
        y = y.to(torch.float)
#         print('y')
#         print(y.shape)
        y_hat = self.forward(x)
#         print('y_hat')
#         print(y_hat.shape)
        loss = self.loss(y_hat, torch.cat([y, y, y], dim = -1))
        #print(loss.shape)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_nb):
        x, y, _ = batch
        x = x.to(torch.float)
        y = y.to(torch.float)
        y_hat = self.forward(x)
        #print(y_hat.shape)
        #print(torch.cat([y, y, y], dim = -1).shape)
        loss = self.loss(y_hat, torch.cat([y, y, y], dim = -1))
        #print(loss)
        return {'val_loss': loss}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y, _ = batch
        x = x.to(torch.float)
        y = y.to(torch.float)
        y_hat = self.forward(x)
        return {'test_loss': self.test_loss(y_hat[Ellipsis, 1], y[Ellipsis, 0])}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]
    
    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        layers = []
        for name, p in named_parameters:
            if p.grad is not None:
                if (p.requires_grad) and ("bias" not in name):
                    layers.append(name)
                    ave_grads.append(p.grad.abs().mean())
                    self.logger.experiment.add_histogram(tag=name, values=p.grad,
                                                         global_step=self.trainer.global_step)
            else:
                 print('{} - {}'.format(name, p.requires_grad))
            
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads), linewidth=1, color="k" )
        plt.xticks(list(range(0,len(ave_grads), 1)), layers, rotation='vertical')
        plt.xlim(left=0, right=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.rcParams["figure.figsize"] = (20, 5)
    
    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  
            self.plot_grad_flow(self.named_parameters())
    
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(train_dataset, batch_size = self.minibatch_size, shuffle=True, drop_last=True, num_workers=1)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(valid_dataset, batch_size = self.minibatch_size, shuffle=True, drop_last=True, num_workers=1)
    
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(test_dataset, batch_size = self.minibatch_size, shuffle=True, drop_last=True, num_workers=1)

# %% [markdown]
# ## Training

# %% [markdown]
# ### Setting Device

# %%
DEVICE = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
DEVICE

# %% [markdown]
# ### Input Parameters

# %%
params = data_formatter.get_experiment_params()
params.update(data_formatter.get_default_model_params())

parser = ArgumentParser(add_help=False)

# %%
for k in params:
    if type(params[k]) in [int, float]:
        #if k == 'minibatch_size':
        #    parser.add_argument('--{}'.format(k), type=type(params[k]), default = 256)
        #else:
        parser.add_argument('--{}'.format(k), type=type(params[k]), default = params[k])
    else:
        parser.add_argument('--{}'.format(k), type=str, default = str(params[k]))
hparams = parser.parse_known_args()[0]

# %% [markdown]
# ### Model Instance

# %%
tft = TemporalFusionTransformer(hparams)#.to(DEVICE)
tft

# %%
early_stop_callback = EarlyStopping(monitor = 'val_loss',
                                    min_delta = 1e-4,
                                    patience = tft.early_stopping_patience,
                                    verbose=False,
                                    mode='min')

# %%
trainer = pl.Trainer(max_nb_epochs = tft.num_epochs,
                     gpus = 1, 
                     track_grad_norm = 2, 
                     gradient_clip_val = tft.max_gradient_norm,
                     early_stop_callback = early_stop_callback,
                     #train_percent_check = 0.01,
                     #val_percent_check = 0.01,
                     #test_percent_check = 0.01,
                     overfit_pct=0.01,
                     #fast_dev_run=True,
                     profiler=True,
                     #print_nan_grads = True,
                     #distributed_backend='dp'
                    )    
trainer.fit(tft)

# %%
trainer.test()

# %% [markdown]
# ### Testing

# %%
# !pwd

# %%
model = tft.load_from_metrics(
                             weights_path='lightning_logs/version_18/checkpoints/epoch=6.ckpt',
                             tags_csv='lightning_logs/version_18/meta_tags.csv',
                             #on_gpu=True,
                             map_location=None
                            )

# %%
q_risk = NormalizedQuantileLossCalculator([0.1, 0.5, 0.9], 1)
test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=True, drop_last=True)

# %%
loss = []
batches = 0
for i, (batch, target, _ )in enumerate(test_dataloader):
    if i < 5:
        t = target
        batches += 1
        output = tft(batch)
        loss.append(q_risk.apply(output[Ellipsis, 1], target[Ellipsis, 0], 0.5))
    else:
        break
mean_loss = sum(loss) / batches
mean_loss

# %%
loss

# %%
output[Ellipsis, 1:2].shape

# %%
t.shape, output.shape

# %% [markdown]
# ## Testing implementation

# %%
def get_decoder_mask(self_attn_inputs):
    """Returns causal mask to apply for self-attention layer.
    Args:
    self_attn_inputs: Inputs to self attention layer to determine mask shape
    """
    len_s = self_attn_inputs.shape[1]
    bs = self_attn_inputs.shape[0]
    mask = torch.cumsum(torch.eye(len_s), 0)
    mask = mask.repeat(bs,1,1).to(torch.float32)

    return mask

# %%
a = torch.randn((2,6,4))
a

# %%
mask = get_decoder_mask(a)
mask

# %%
mask.shape

# %%
linear = nn.Linear(4, 2, bias = False)

# %%
a_lin = linear(a)
a_lin.shape

# %%
to_attn = torch.bmm(a_lin, a_lin.permute(0,2,1))

# %%
masked_attn = to_attn.masked_fill(mask == 0, -1e9)
masked_attn

# %%
softmax = nn.Softmax(dim = 2)

# %%
sft_attn = softmax(masked_attn)
sft_attn

# %%
torch.bmm(sft_attn, a_lin).shape

# %%
scaled_att = ScaledDotProductAttention()

# %%
scaled_att(a_lin, a_lin, a_lin, mask = get_decoder_mask(a))

# %%

