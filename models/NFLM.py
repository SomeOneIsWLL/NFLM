import torch
import torch.nn as nn


class IEBlock(nn.Module):
    def __init__(self, input_dim, hid_dim, f1_size,output_dim, num_node):
        super(IEBlock, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node
        self.f1_size=f1_size
        self._build()
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_node))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_node))

    def _build(self):

        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.f1_size)

        )

        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)
        self.output_proj = nn.Linear(self.f1_size, self.output_dim)

    def forward(self, x):
        means = x.mean(1, keepdim=True).detach()
        # mean
        x = x - means
        # var
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev
        x = x * self.affine_weight + self.affine_bias

        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))
        x = x.permute (0, 2, 1)

        ## reverse RIN ###
        x = x - self.affine_bias
        x = x / (self.affine_weight )
        x = x * stdev
        x = x + means
        return x


class Model(nn.Module):


    def __init__(self, configs):
        """
        chunk_size: int, reshape T into [num_chunks, chunk_size]
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len=configs.pred_len
        self.chunk_size = min(configs.pred_len, configs.seq_len, configs.chunk_size)
        assert (self.seq_len % self.chunk_size == 0)
        self.num_chunks = self.seq_len // self.chunk_size
        self.enc_in = configs.enc_in
        self.f1_size=configs.f1_size
        self.hid_dim=configs.hid_dim
        self.output_dim=configs.output_dim
        self._build()
    def _build(self):
        self.layer_1 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.hid_dim,
            f1_size=self.f1_size,
            output_dim=self.output_dim,
            num_node=self.num_chunks,
        )

        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.hid_dim,
            f1_size=self.f1_size,
            output_dim=self.output_dim,
            num_node=self.num_chunks,
        )

        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = IEBlock(
            input_dim=self.output_dim* 2,
            hid_dim=self.hid_dim*2,
            f1_size=self.f1_size*2,
            output_dim=self.pred_len,
            num_node=self.enc_in,

        )

        self.ar = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, N = x_enc.size()

        highway = self.ar(x_enc.permute(0, 2, 1))
        highway = highway.permute(0, 2, 1)

        x1 = x_enc.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1)

        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)

        x1 = self.layer_1(x1)

        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)


        x2 = x_enc.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)


        x3 = torch.cat([x1, x2], dim=-1)
        x3 = x3.reshape(B, N, -1)
        x3 = x3.permute(0, 2, 1)


        out = self.layer_3(x3)
        out = out + highway
        return out[:, -self.pred_len:, :]  # [B, L, D]
        # return None
