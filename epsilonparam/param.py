import torch
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.model import ScaleSpaceFlow
from ptflops import get_model_complexity_info
import time

# Initialize device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model components without loading weights
denoise_fn = Unet(
    dim=64,
    channels=3,
    context_channels=3,
    dim_mults=(1, 2, 3, 4, 5, 6),
    context_dim_mults=(1, 2, 3, 4),
).to(device)

context_fn = ScaleSpaceFlow().to(device)

model = GaussianDiffusion(
    denoise_fn=denoise_fn,
    context_fn=context_fn,
    num_timesteps=20000,
    loss_type="l2",
    clip_noise="full",
    vbr=False,
    aux_loss_weight=1,
    pred_mode="noise",
    var_schedule="linear",
    aux_loss_type="l2"
).to(device)

# Define random input for testing (batch_size, channels, height, width)
input_image = torch.randn(1, 3, 256, 256).to(device)
ref_image = torch.randn(1, 3, 256, 256).to(device)

# Custom wrapper for ptflops to pass both x and x_ref
class DiffusionWrapper(torch.nn.Module):
    def __init__(self, model):
        super(DiffusionWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Wrap the model's forward to accept both inputs (x and x_ref)
        x_ref = torch.randn_like(x).to(x.device)  # Generate a dummy reference image
        return self.model(x, x_ref)

# Initialize the wrapper for the diffusion model
model_wrapper = DiffusionWrapper(model).to(device)

# Calculate FLOPs and Parameters using ptflops
macs, params = get_model_complexity_info(model_wrapper, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
print(f"FLOPs: {macs}")
print(f"Parameters: {params}")

# Measure FPS (Frames per Second) and encoding/decoding time
num_iters = 100

# Measure encoding time
start_time_encoding = time.time()
for _ in range(num_iters):
    with torch.no_grad():
        encoded_output = model(input_image, ref_image)
end_time_encoding = time.time()
encoding_time = (end_time_encoding - start_time_encoding) / num_iters

# Measure FPS (Frames per Second)
start_time_fps = time.time()
for _ in range(num_iters):
    with torch.no_grad():
        model(input_image, ref_image)
end_time_fps = time.time()
fps = num_iters / (end_time_fps - start_time_fps)

print(f"Encoding Time (per iteration): {encoding_time:.6f} seconds")
print(f"FPS: {fps:.2f}")
print(f"FLOPs: {macs}")
print(f"Parameters: {params}")
