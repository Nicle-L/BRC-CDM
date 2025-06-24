# training config
n_step = 100000
scheduler_checkpoint_step = 5000
log_checkpoint_step = 200
gradient_accumulate_every = 1
lr = 1e-4
decay = 0.9
minf = 0.5
optimizer = "adam"  # adamw or adam
n_workers = 0

# load
load_model = False
load_step = False

# diffusion config
pred_mode = 'noise'
loss_type = "l2"
iteration_step = 20000
sample_steps = 2000
embed_dim = 64
dim_mults = (1, 2, 3, 4, 5, 6)
hyper_dim_mults = (4, 4, 4)
context_channels = 3
clip_noise = "full"
val_num_of_batch = 1
additional_note = ""
vbr = False
context_dim_mults = (1, 2, 3, 4)
sample_mode = "ddim"
var_schedule = "linear"
aux_loss_type = "l2"
compressor = "scale"

# data config
#data_config = {
#    "dataset_name": "vimeo",
#    "data_path": "E:/ljh/vimeo_septuplet",
#    "sequence_length": 1,
#    "img_size": 256,
#    "img_channel": 3,
#    "add_noise": False,
#    "img_hz_flip": False,
#}

batch_size =16
train_path="E:/ljh/video-test3/data/train.txt"

img_size = 256

train_size=7513
dataset_name = "HSI"
img_channel= 3

test_size= 1907
test_path="E:/ljh/video-test3/data/test.txt"

result_root = "E:/ljh/hsi_dm_compression_main/epsilonparam/result"
tensorboard_root = "E:/ljh/hsi_dm_compression/epsilonparam/tensorboard"

