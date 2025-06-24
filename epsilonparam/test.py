import argparse
import os
from datetime import datetime
import torch
import pathlib
import torchvision.transforms as transforms
from PIL import Image
import scipy.io as sio
from torch import nn
from torch.utils.data import DataLoader
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.model import ScaleSpaceFlow
from dataset import testDataSet1
import Util.torch_msssim as torch_msssim
import numpy as np
import lpips

# 初始化命令行解析器
parser = argparse.ArgumentParser(description="Script to validate the diffusion model")
parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
parser.add_argument("--n_denoise_step", type=int, default=2000, help="Number of denoising steps")
parser.add_argument("--device", type=int, default=0, help="GPU device index")
parser.add_argument("--val_path", type=str, default='E:/ljh/video-test3/data/view_001.txt',
                    help="Path to the validation dataset")
parser.add_argument("--out_dir", type=str,
                    default='E:/ljh/hsi_dm_compression_main_test(7)/epsilonparam/test_result',
                    help="Output directory for results")
args = parser.parse_args()


def calculate_psnr(input_image, compressed):
    mse = torch.mean((input_image - compressed) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def calculate_sam(input_image, x_rec):
    cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    sam = torch.acos(torch.clamp(cos_similarity(input_image, x_rec), -1 + 1e-7, 1 - 1e-7))
    return torch.mean(sam)

def save_compressed_as_image(compressed, output_path, batch_idx):
    compressed = compressed.clamp(0, 1)

    to_pil = transforms.ToPILImage()
    compressed_pil = to_pil(compressed.squeeze(0).cpu())
    output_image_path = os.path.join(output_path, f"compressed_batch_{batch_idx + 1}.png")
    compressed_pil.save(output_image_path)
    print(f"Compressed image saved to {output_image_path}")

# 模型验证函数
def validate_model(
        model,
        val_dl,
        weights_path,
        device,
        sample_steps,
        results_folder,
        val_num_of_batch,
        lpips_model,
        msssim_func
):

    model.load_state_dict(torch.load(weights_path, map_location=device)["model"])
    model.to(device)
    model.eval()


    results_folder = pathlib.Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)


    results_file = os.path.join(results_folder, "validation_results3.txt")

    with open(results_file, 'w') as f:
        f.write("Batch\tBPP\tPSNR\tMS-SSIM(dB)\tSAM\tLPIPS\n")

        with torch.no_grad():
            for i, batch in enumerate(val_dl):
                input_image, ref_image = batch[0].to(device), batch[1].to(device)

                if i >= val_num_of_batch:
                    break

                compressed, bpp = model.compress(
                    input_image,
                    ref_image,
                    sample_steps=sample_steps,
                    sample_mode="ddim",
                    bpp_return_mean=True,
                    init=None,
                    eta=0,
                )
                compressed = compressed.clamp(-1, 1) / 2.0 + 0.5


                save_compressed_as_image(compressed, results_folder, i)


                psnr = calculate_psnr(input_image, compressed)

                ms_ssim = msssim_func(compressed, input_image)
                ms_ssim_db = -10.0 * np.log10(max(1 - ms_ssim.item(), 1e-10))


                sam = calculate_sam(input_image, compressed)

                lpips_value = lpips_model(compressed, input_image)
                lpips_value = lpips_value.mean().item()

                f.write(f"{i + 1}\t{bpp.item():.4f}\t{psnr:.4f}\t{ms_ssim_db:.4f}\t{sam.item():.4f}\t{lpips_value:.4f}\n")

                print(f"Batch {i + 1}: BPP = {bpp.item():.4f}, PSNR = {psnr:.4f}, MS-SSIM(dB) = {ms_ssim_db:.4f}, SAM = {sam.item():.4f}, LPIPS = {lpips_value:.4f}")

if __name__ == "__main__":
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    val_dataset = testDataSet1(path=args.val_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # 初始化 MS-SSIM 函数
    msssim_func = torch_msssim.MS_SSIM(max_val=1.).to(device)
    # 初始化模型
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

    # 开始验证
    validate_model(
        model=model,
        val_dl=val_loader,
        weights_path=args.ckpt,
        device=device,
        sample_steps=args.n_denoise_step,
        results_folder=args.out_dir,
        val_num_of_batch=1,  # 假设要验证33个批次
        lpips_model=lpips_model,
        msssim_func=msssim_func
    )
