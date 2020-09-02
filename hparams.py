from text import symbols

# Mel
num_mels = 80
n_mel_channels = num_mels
text_cleaners = ['english_cleaners']


# Model
# vocab_size = 300
# max_seq_len = 3000

# encoder_dim = 512
# encoder_num_conv = 3
# encoder_num_gru_layer = 1

# decoder_dim = 270
# decoder_head = 3
# decoder_n_layer = 4
# decoder_cb_num_conv = 3
# decoder_kernel_size = 3

# duration_predictor_filter_size = 256
# duration_predictor_kernel_size = 3

# n_frame_per_step = 3
dropout = 0.1

n_symbols = len(symbols)
symbols_embedding_dim = 512

# Encoder parameters
encoder_kernel_size = 5
encoder_n_convolutions = 3
encoder_embedding_dim = 512

# Decoder parameters
n_frames_per_step = 1
decoder_rnn_dim = 1024
prenet_dim = 256
max_decoder_steps = 1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1

# Attention parameters
attention_rnn_dim = 1024
attention_dim = 128

# Location Layer parameters
attention_location_n_filters = 32
attention_location_kernel_size = 31

# Mel-post processing network parameters
postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

mask_padding = True
fp16_run = False


# Train
checkpoint_path = "./model_new"
logger_path = "./logger"
mel_ground_truth = "./mels"
alignment_path = "./alignments"

batch_size = 64
epochs = 2000
n_warm_up_step = 4000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 3000
log_step = 5
clear_Time = 20

batch_expand_size = 32
