"""
Model settings
"""
class HARConfig:
    def __init__(self, model: str = "tiny_transformer"):
        # HAR settings
        self.model = model
        self.classes = 17
        self.features = 270
        self.window = 600
        self.shift = self.window // 2
        self.folds = 10
        self.ratio = 0.2
        self.device = "cuda"
        self.error = False

        # Tiny transformer encoder model settings
        if self.model == "tiny_transformer":
            self.patch = 8
            self.batch = 128
            self.dimension = 128
            self.depth = 2
            self.heads = 4
            self.learning_rate = 3e-4
            self.weight_decay = 1e-4
            self.epochs = 20
            self.mlp_ratio = 2.0
            self.drop = 0.1
        
        # Transformer encoder model settings
        elif self.model == "transformer_encoder":
            self.dimension = 256
            self.heads = 8
            self.depth = 3 
            self.dim_feedforward = 1024 
            self.dropout = 0.2 
            self.baseline_dropout = 0.1
            self.batch = 128
            self.learning_rate = 3e-4
            self.weight_decay =1e-04
            self.eps = 1e-8
            self.step_size = 10
            self.gamma = 0.75
            self.epochs = 30
        
        # CNN
        elif self.model == "cnn":
            self.batch_size = 128
            self.epochs = 10000
            self.learning_rate = 3e-4
            self.verbose = 0
        
        # GRU
        elif self.model == "gru":
            self.hidden_layer = 256
            self.l2_lambda = 0.001
            self.spatial_dropout = 0.3
            self.dropout = 0.4
            self.learning_rate = 0.001
            self.batch_size = 128
            self.epochs = 10000
            self.verbose = 0

        # Deep convolutional LSTM
        elif self.model == "deep_conv_lstm":
            self.batch_size = 128
            self.epochs = 10000
            self.learning_rate = 0.001
            self.verbose = 0
            self.lstm_units = 128
            self.dropout_rate = 0.5
            self.lstm_segments = 15
        
        # Model not defined
        else:
            self.error = True