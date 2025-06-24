import numpy as np
import torch
import torch.nn as nn
from scipy.io import savemat
from model import ScaleSpaceFlow
from video.dataset3 import testDataSet1
from torch.utils.data import DataLoader
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
import Util.torch_msssim as torch_msssim
import lpips

def save_compressed_as_image(compressed, output_path, batch_idx):
    compressed = compressed.clamp(0, 1)
    to_pil = transforms.ToPILImage()
    compressed_pil = to_pil(compressed.squeeze(0).cpu())
    print(compressed_pil.size)
    output_image_path = os.path.join(output_path, f"compressed_batch_{batch_idx + 1}.png")
    compressed_pil.save(output_image_path)
    print(f"Compressed image saved to {output_image_path}")

def save_results_as_mat(x_hat, step, output_dir):
    try:
        x_hat = x_hat.squeeze().cpu().numpy()
        result_dict = {
            'data': x_hat
        }
        mat_file_path = os.path.join(output_dir, f'results_step_{step}.mat')
        savemat(mat_file_path, result_dict)
        print(f'Successfully saved {mat_file_path}')
    except Exception as e:
        print(f'Failed to save .mat file: {e}')


model = ScaleSpaceFlow()


model_path = 'E:\ljh/video-test3\output\weight\model_epoch_5700_lamda4096.pth' # 替换为你的模型权重路径
model.load_state_dict(torch.load(model_path))

model = model.cuda()


model.eval()


output_dir = "E:/ljh/video-test3/view_abl/rec"
os.makedirs(output_dir, exist_ok=True)
output_dir1 = "E:/ljh/video-test3/view_abl/input"
os.makedirs(output_dir1, exist_ok=True)

val_dataset = testDataSet1(path="E:/ljh/video-test3/data/view_abl.txt")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# 初始化变量
summse_test = 0
sum_bpp_test = 0
sum_ssim_test = 0
sum_sam_test = 0
sum_lpips_test = 0
log_dir = 'E:/ljh/video-test3/view_abl/test'
os.makedirs(log_dir, exist_ok=True)

msssim_func = torch_msssim.MS_SSIM(max_val=1.).to("cuda:0")

lpips_model = lpips.LPIPS(net='alex').to("cuda:0")

def calculate_sam(input_image, x_rec):
    cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    sam = torch.acos(torch.clamp(cos_similarity(input_image, x_rec), -1 + 1e-7, 1 - 1e-7))
    return torch.mean(sam)


for step, input in enumerate(val_loader):
    input_image, ref_image = input[0].cuda(), input[1].cuda()
    num_pixels = input_image.size()[0] * input_image.size()[1] * input_image.size()[2] * input_image.size()[3]
    lamb = 8192

    # 前向传播
    with torch.no_grad():
        output_dict = model(input_image, ref_image)
        x_rec_list = output_dict['output']
        bppb = output_dict['bppb']
        x_rec = x_rec_list[0]

    loss_func = nn.MSELoss()
    distortion = loss_func(input_image, x_rec)


    ms_ssim = msssim_func(input_image, x_rec)
    ms_ssim_db = -10.0 * np.log10(1 - ms_ssim.item())

    sam = calculate_sam(input_image, x_rec)

    lpips_value = lpips_model(input_image, x_rec).item()


    sum_bpp_test += bppb.item()
    summse_test += distortion.item()
    sum_ssim_test += ms_ssim_db
    sum_sam_test += sam.item()
    sum_lpips_test += lpips_value


    mse = distortion.item()
    psnr = 10.0 * np.log10(1 / mse)
    print(x_rec.size())
    save_compressed_as_image(x_rec , output_dir , step)
    print(f'step: {step}, MSE: {mse:.6f}, bppb: {bppb.item():.4f}, PSNR: {psnr:.2f}, MS-SSIM (dB): {ms_ssim_db:.4f}, SAM: {sam:.4f}, LPIPS: {lpips_value:.4f}')

    if (step + 1) % 1 == 0:
        mse_avg = summse_test / (step + 1)
        psnr_avg = 10.0 * np.log10(1 / mse_avg)
        bpp_total = sum_bpp_test / (step + 1)
        ssim_avg_db = sum_ssim_test / (step + 1)
        sam_avg = sum_sam_test / (step + 1)
        lpips_avg = sum_lpips_test / (step + 1)

        log_file_path = os.path.join(log_dir, f'val_001_{lamb}.log')
        with open(log_file_path, 'a') as fd:
            fd.write(f'step:{step + 1} MSE:{mse_avg:.6f} bppb_total:{bpp_total:.4f} PSNR:{psnr_avg:.2f} MS-SSIM (dB):{ssim_avg_db:.4f} SAM:{sam_avg:.4f} LPIPS:{lpips_avg:.4f}\n')

        print(f'step:{step + 1} MSE:{mse_avg:.6f} bppb_total:{bpp_total:.4f} PSNR:{psnr_avg:.2f} MS-SSIM (dB):{ssim_avg_db:.4f} SAM:{sam_avg:.4f} LPIPS:{lpips_avg:.4f}')
