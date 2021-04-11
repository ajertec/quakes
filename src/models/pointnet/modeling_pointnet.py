from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from configuration_pointnet import PointNetConfig


def conv_block(in_ch, out_ch, kernel_size, *args, **kwargs):

    "Generic 1D convolution block with batchnorm and relu."

    return nn.Sequential(
        nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            *args,
            **kwargs
        ),
        nn.BatchNorm1d(num_features=out_ch),
        nn.ReLU(),
    )


def linear_block(in_feat, out_feat, bias=True):

    "Generic 1D linear block with batchnorm and relu."

    return nn.Sequential(
        nn.Linear(in_features=in_feat, out_features=out_feat, bias=bias),
        nn.BatchNorm1d(num_features=out_feat),
        nn.ReLU(),
    )


class Tnet(nn.Module):
    def __init__(self, config, tnet_type: str):
        super().__init__()

        assert tnet_type in ["input", "feature"]

        self.num_points = config.num_points
        self.config = config
        self.tnet_type = tnet_type

        if self.tnet_type == "input":
            self.input_dim = self.config.encoder_post_inp_dims[0]
            self.config.encoder_dims[0] = self.config.encoder_post_inp_dims[0]
        elif self.tnet_type == "feature":
            self.input_dim = self.config.encoder_post_feat_dims[0]
            self.config.encoder_dims[0] = self.config.encoder_post_feat_dims[0]
        else:
            raise ValueError

        self.encoder = nn.Sequential(
            *[
                conv_block(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(
                    self.config.encoder_dims, config.encoder_dims[1:]
                )
            ]
        )

        self.maxpool = nn.MaxPool1d(self.num_points)

        self.linears = nn.Sequential(
            *[
                linear_block(in_feat, out_feat, bias=True)
                for in_feat, out_feat in zip(
                    self.config.linear_dims, config.linear_dims[1:]
                )
            ]
        )

        self.fc_out = nn.Linear(config.linear_dims[-1], self.input_dim ** 2)

    def forward(self, x):
        batch_size = x.size()[0]

        x = self.encoder(x)

        x = self.maxpool(x)
        x = x.view(batch_size, self.config.encoder_dims[-1])

        x = self.linears(x)
        x = self.fc_out(x)

        identity = (
            torch.eye(self.input_dim, requires_grad=True)
            .flatten()
            .repeat(batch_size, 1)
        )

        if x.is_cuda:
            identity = identity.cuda()

        x = x + identity

        return x.view(-1, self.input_dim, self.input_dim)


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.num_points = config.num_points
        self.input_dim = config.input_dim
        self.config = config

        self.config.encoder_post_inp_dims[0] = self.input_dim

        self.encoder_post_inp = nn.ModuleList(
            [
                conv_block(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(
                    self.config.encoder_post_inp_dims, config.encoder_post_inp_dims[1:]
                )
            ]
        )

        self.encoder_post_feat = nn.ModuleList(
            [
                conv_block(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(
                    self.config.encoder_post_feat_dims,
                    config.encoder_post_feat_dims[1:],
                )
            ]
        )

        self.maxpool = nn.MaxPool1d(self.num_points)

        if config.use_tnet:
            self.input_tnet = Tnet(config=config, tnet_type="input")
            self.feature_tnet = Tnet(config=config, tnet_type="feature")

    def forward(self, x):

        batch_size = x.size()[0]

        if self.config.use_tnet:
            x = torch.bmm(x.transpose(2, 1), self.input_tnet(x))
            x = x.transpose(2, 1)
            print("nj1", x.shape)

        for layer_block in self.encoder_post_inp:
            # potentially save point features
            x = layer_block(x)
            print("nj2", x.shape)

        if self.config.use_tnet:
            x = torch.bmm(x.transpose(2, 1), self.feature_tnet(x))
            x = x.transpose(2, 1)
            print("nj3", x.shape)

        for layer_block in self.encoder_post_feat:
            x = layer_block(x)
            print("nj4", x.shape)

        x = self.maxpool(x)
        x = x.view(batch_size, self.config.encoder_post_feat_dims[-1])

        return x


class PointNetDecoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        # jel je opce potreban config tnet?

        self.num_points = config.num_points
        self.input_dim = config.decoder_dims[0]
        self.output_dim = config.decoder_output_dim
        self.config = config

        self.config.decoder_dims[0] = self.input_dim
        self.linears = nn.Sequential(
            *[
                linear_block(in_feat, out_feat, bias=True)
                for in_feat, out_feat in zip(
                    self.config.decoder_dims, self.config.decoder_dims[1:]
                )
            ]
        )

        self.fc_out = nn.Linear(
            config.decoder_dims[-1], self.num_points * self.output_dim, bias=True
        )

    def forward(self, x):
        batch_size = x.size()[0]

        x = x.view(batch_size, self.input_dim)

        x = self.linears(x)
        x = self.fc_out(x)

        return x.view(batch_size, self.output_dim, self.num_points)


if __name__ == "__main__":
    # debugging

    # batch size, coord dim, num points
    x = torch.rand(32, 3, 300)

    config = PointNetConfig()

    print("\n ** input TNET **")
    tnet = Tnet(config=config, tnet_type="input")
    print(tnet)
    print(tnet(x).shape)

    x = torch.rand(32, 128, 300)

    print("\n ** feature TNET **")
    tnet = Tnet(config=config, tnet_type="feature")
    print(tnet)
    print(tnet(x).shape)

    x = torch.rand(32, 3, 300)
    print("\n ** Encoder **")
    pn_encoder = PointNetEncoder(config=config)

    print(pn_encoder)
    print(pn_encoder(x).shape)

    x = torch.rand(32, 64)

    print("\n ** Decoder **")
    pn_decoder = PointNetDecoder(
        config=config,
    )

    print(pn_decoder)
    print(pn_decoder(x).shape)
