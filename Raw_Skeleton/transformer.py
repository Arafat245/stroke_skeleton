import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Transformer(nn.Module):

    def __init__(self, N_gaits, input_dim=96, d_model=12, d_ff=128, n_heads=4, 
                 e_layers=2, output_dim=1, dropout=0.3, activation='gelu'):
        super(Transformer, self).__init__()

        self.seq_len = 100*N_gaits    
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.activation = activation
        
        self.enc_embedding = DataEmbedding(input_dim, d_model, embed_type='fixed', freq='h', dropout=dropout)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=dropout,
                                      output_attention=False), 
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.d_model * self.seq_len, self.output_dim)

    def forward(self, x):

        B, G, T, C = x.shape
        
        x_enc = x.view(B, G * T, C) 
        
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out) 
        output = self.dropout(output)
        
        output = output.reshape(output.shape[0], -1) 
        output = self.projection(output) 
        
        return output