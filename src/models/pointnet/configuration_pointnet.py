class PointNetConfig:
    def __init__(
        self,
        num_points: int = 300,
        input_dim: int = 3,
        encoder_dims=[3, 64, 128, 1024],
        linear_dims=[1024, 512, 256],
        decoder_dims=[64, 64, 64],
        decoder_output_dim: int = 3,
        encoder_post_inp_dims=[3, 64, 128, 128],
        encoder_post_feat_dims=[128, 512, 2048],
        use_tnet: bool = True,
    ):

        self.num_points = num_points
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.linear_dims = linear_dims
        self.decoder_dims = decoder_dims
        self.decoder_output_dim = decoder_output_dim
        self.encoder_post_inp_dims = encoder_post_inp_dims
        self.encoder_post_feat_dims = encoder_post_feat_dims
        self.use_tnet = use_tnet

        assert encoder_post_inp_dims[-1] == encoder_post_feat_dims[0]