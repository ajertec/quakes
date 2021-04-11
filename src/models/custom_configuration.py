from transformers.configuration_utils import PretrainedConfig


class QuakeNetConfig(PretrainedConfig):

    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        position_embedding_type="absolute",
        use_cache=True,
        # PN params:
        num_points=256,
        input_dim=3,
        seq_len=64,
        encoder_points_features_dim=[3, 64, 128, 512, 768],
        encoder_points_reduction_dim=[256, 256, 64],
        x_size=2000,
        y_size=2000,
        num_special_inputs=4,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache

        # PN params:
        self.num_points = num_points
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.encoder_points_features_dim = encoder_points_features_dim
        self.encoder_points_reduction_dim = encoder_points_reduction_dim

        self.x_size = x_size
        self.y_size = y_size

        self.num_special_inputs = num_special_inputs