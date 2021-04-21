import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertPredictionHeadTransform,
    SequenceClassifierOutput,
    BertEncoder,
    BertPooler,
)

from .custom_configuration import QuakeNetConfig


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
        nn.LayerNorm(normalized_shape=out_feat),
        nn.ReLU(),
    )


class PNConfig:
    def __init__(
        self,
        num_points=256,
        input_dim=3,
        hidden_size=768,
        seq_len=64,
        hidden_dropout_prob=0.1,
        encoder_points_features_dim=[3, 64, 128, 512, 768],
        encoder_points_reduction_dim=[256, 256, 64],
    ):
        self.num_points = num_points
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.hidden_dropout_prob = hidden_dropout_prob
        self.encoder_points_features_dim = encoder_points_features_dim
        self.encoder_points_reduction_dim = encoder_points_reduction_dim


class SimplePointNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.num_points = config.num_points
        self.input_dim = config.input_dim

        self.output_hidden_size = config.hidden_size
        self.output_seq_len = config.seq_len

        self.encoder_points_features_dim = config.encoder_points_features_dim
        self.encoder_points_reduction_dim = config.encoder_points_reduction_dim

        assert self.encoder_points_reduction_dim[0] == self.num_points
        assert self.encoder_points_reduction_dim[-1] == self.output_seq_len

        assert self.encoder_points_features_dim[0] == self.input_dim
        assert self.encoder_points_features_dim[-1] == self.output_hidden_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.maxpool = nn.MaxPool1d(self.num_points)
        self.linear = nn.Linear(1, self.encoder_points_reduction_dim[0])

        self.point_feature_blocks = nn.ModuleList(
            [
                conv_block(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(
                    self.encoder_points_features_dim,
                    self.encoder_points_features_dim[1:],
                )
            ]
        )

        # TODO replace with dense blocks?
        self.point_reduction_blocks = nn.ModuleList(
            [
                linear_block(in_ch, out_ch)
                for in_ch, out_ch in zip(
                    self.encoder_points_reduction_dim,
                    self.encoder_points_reduction_dim[1:],
                )
            ]
        )

    def forward(self, x):

        # ??
        x = self.dropout(x)

        for point_feature_block in self.point_feature_blocks:
            # potentially save point features
            x = point_feature_block(x)

        # TODO needs maxpool for order invariance! ?
        # is invariance something I really need? quake dataset will be order in time anyway
        # TODO after dense layers!

        x = self.maxpool(x)
        x = self.linear(x)
        # x = x.permute(0, 2, 1)

        for point_reduction_block in self.point_reduction_blocks:
            x = point_reduction_block(x)

        return x.permute(0, 2, 1)


class BertEmbeddingsCustom(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # special inputs, like CLS, UNK, SEP token, but in this case used as special aggregation positions
        # for various predictions
        self.special_inputs_embedding = nn.Embedding(
            config.num_special_inputs, config.hidden_size
        )
        self.num_special_inputs = config.num_special_inputs

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(self, position_ids=None, inputs_embeds=None):

        special_inputs = self.special_inputs_embedding(
            torch.arange(self.num_special_inputs).to(inputs_embeds.device)
        )

        bs, old_seq_len, hidden_size = inputs_embeds.size()
        seq_len = old_seq_len + self.num_special_inputs

        new_input_embeds = torch.zeros((bs, seq_len, hidden_size)).to(
            inputs_embeds.device
        )
        new_input_embeds[:, self.num_special_inputs :, :] = inputs_embeds
        new_input_embeds[:, : self.num_special_inputs] = special_inputs

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = new_input_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPoolerCustom(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        self.num_special_inputs = config.num_special_inputs

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        special_tensors = hidden_states[:, : self.num_special_inputs]
        pooled_outputs = self.dense(special_tensors)
        pooled_outputs = self.activation(pooled_outputs)
        return pooled_outputs


class BertModelCustom(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsCustom(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPoolerCustom(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        embedding_output = self.embeddings(
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        input_shape = embedding_output.size()[:-1]
        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_outputs = self.pooler(sequence_output)

        return (sequence_output, pooled_outputs) + encoder_outputs[1:]


class QuakeNet(BertPreTrainedModel):
    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.pointnet_encoder = SimplePointNetEncoder(config)

        # TODO create custom BertModel with removed word embeddings layer, position embeddings should be kept
        self.bert = BertModelCustom(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.transform_x = BertPredictionHeadTransform(config)
        self.transform_y = BertPredictionHeadTransform(config)

        # TODO why x and y size if I assume that they are equal ...

        self.x_size = config.x_size
        self.y_size = config.y_size

        assert self.x_size == self.y_size

        self.classifier_x = nn.Linear(config.hidden_size, config.x_size)
        self.classifier_y = nn.Linear(config.hidden_size, config.y_size)

    def forward(
        self,
        inputs=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        inputs_embeds = self.pointnet_encoder(inputs)

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        transformed_output_x = self.transform_x(pooled_output[:, 0])
        logits_x = self.classifier_x(transformed_output_x)

        transformed_output_y = self.transform_y(pooled_output[:, 1])
        logits_y = self.classifier_y(transformed_output_y)

        # assumes that x_size and y_size are equal
        logits = torch.cat([logits_x.unsqueeze(1), logits_y.unsqueeze(1)], dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss_x = loss_fct(logits_x.view(-1, self.x_size), labels[:, 0].view(-1))
            loss_y = loss_fct(logits_y.view(-1, self.y_size), labels[:, 1].view(-1))

            loss = loss_x + loss_y

        output = (logits,) + outputs[2:]
        # (loss), logits, other outputs
        return ((loss,) + output) if loss is not None else output


if __name__ == "__main__":

    print("** PN encoder **")
    config = PNConfig()

    pn_encoder = SimplePointNetEncoder(config=config)
    print(pn_encoder)

    x = torch.rand(32, 3, 256)

    print("input shape: ", x.shape)

    print("output shape", pn_encoder(x).shape)

    print(
        "number of params:",
        sum(p.numel() for p in pn_encoder.parameters() if p.requires_grad),
    )

    print("** QuakeNet **")
    qn_config = QuakeNetConfig()

    quake_net = QuakeNet(qn_config)

    print(quake_net)

    print(
        "number of params in QuakeNet:",
        sum(p.numel() for p in quake_net.parameters() if p.requires_grad),
    )

    x = torch.rand((32, 3, 256))

    outputs = quake_net(inputs=x)
    print(outputs[0].shape)
    print(outputs[1].shape)

    print("** QuakeNet from pretrained")
    qn_config = QuakeNetConfig()

    quake_net_pt = QuakeNet.from_pretrained("bert-base-cased", config=qn_config)