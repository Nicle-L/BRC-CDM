import numpy as np
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from .utils import cycle
from torch.optim.lr_scheduler import LambdaLR
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.utils as utils


import os

def batch_psnr(imgs1, imgs2):
    with torch.no_grad():
        # 计算每个维度上的均方误差
        mse = torch.mean((imgs1 - imgs2) ** 2, axis=(1, 2, 3))
        # 计算每个维度上的 PSNR
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        # 计算每个维度上的平均 PSNR
        mean_psnr = torch.mean(psnr, axis=0)

        return mean_psnr


def adjust_learning_rate(optimizer, epoch, init_lr):
    """
    Adjusts the learning rate based on the epoch number.

    :param optimizer: The optimizer for which to adjust the learning rate.
    :param epoch: The current epoch number.
    :param init_lr: The initial learning rate.
    :return: The adjusted learning rate.
    """
    if epoch <= 2500:
        lr = init_lr
    elif epoch <= 3000:
        lr = init_lr / 2
    elif epoch <= 3500:
        lr = init_lr / 4
    elif epoch <= 4000:
        lr = init_lr / 8
    elif epoch <= 4500:
        lr = init_lr / 10
    elif epoch <= 5000:
        lr = init_lr / 12
    elif epoch <= 10000:
        lr = 1e-6
    else:
        lr = max(init_lr / 100, 1e-6)  # Ensure learning rate does not drop below 1e-6

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        sample_steps,
        diffusion_model,
        train_dl,
        val_dl,
        scheduler_function,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=10000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="E:/ljh/CDC_compression-main-test/epsilonparam/result3",
        tensorboard_dir="E:/ljh/CDC_compression-main-test/epsilonparam/tensorboard_logs/diffusion-test3",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        sample_mode="ddpm"
    ):
        super().__init__()
        self.model = diffusion_model
        # self.ema = EMA(ema_decay)
        # self.ema_model = copy.deepcopy(self.model)
        self.sample_mode = sample_mode
        # self.update_ema_every = update_ema_every
        self.val_num_of_batch = val_num_of_batch
        self.sample_steps = sample_steps

        # self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps
        self.train_dl = train_dl
        self.val_dl = val_dl
        if optimizer == "adam":
            self.opt = Adam(self.model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # if os.path.isdir(tensorboard_dir):
        #     shutil.rmtree(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     # self.ema_model.load_state_dict(self.model.state_dict())
    #     pass

    # def step_ema(self):
    #     # if self.step < self.step_start_ema:
    #     #     self.reset_parameters()
    #     # else:
    #     #     self.ema.update_model_average(self.ema_model, self.model)
    #     pass

    def save(self):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            # "ema": self.ema_model.module.state_dict(),
        }
        idx = (self.step // self.save_and_sample_every) % 3
        torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))

    def load(self, idx=0, load_step=True):
        data = torch.load(
            str(self.results_folder / f"{self.model_name}_{idx}.pt"),
            map_location=lambda storage, loc: storage,
        )

        if load_step:
            self.step = data["step"]
        try:
            self.model.module.load_state_dict(data["model"], strict=False)
        except:
            self.model.load_state_dict(data["model"], strict=False)
        # self.ema_model.module.load_state_dict(data["ema"], strict=False)

    def train(self):
        self.model.train()
        for step in range(self.train_num_steps):
            print("----begin training----")
            cur_lr1 = adjust_learning_rate(self.opt, step, self.train_lr)  # 调整第一个优化器的学习率
            for i, data in enumerate(self.train_dl):
                input_image, ref_image = data[0].to(self.device), data[1].to(self.device)
                # print("Input image size:", input_image.shape)
                # print("Reference image size:", ref_image.shape)
                bppb,x_output,loss = self.model(input_image, ref_image)#加载权重
                self.opt.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                loss.backward()
                self.writer.add_scalar("loss", loss.item(), self.step)
                self.opt.step()
            print("---- training end----")
            self.step = step + 1
            print("step:", step)

            if (self.step % self.save_and_sample_every == 0):
                self.model.eval()
                print("-----test started-----")
                for i, batch in enumerate(self.val_dl):
                    input_image, ref_image = batch[0].to(self.device), batch[1].to(self.device)
                    if i >= self.val_num_of_batch:
                        break
                    if self.model.vbr:
                        scaler = torch.zeros(batch.shape[1]).unsqueeze(1).to(self.device)
                    else:
                        scaler = None
                    with torch.no_grad():
                        compressed, bpp = self.model.compress(
                            input_image, ref_image, self.sample_steps, scaler, self.sample_mode
                        )
                        compressed = (compressed + 1.0) * 0.5
                        self.writer.add_scalar(
                            f"bppb/num{i}",
                            bpp,
                            self.step,
                        )
                        self.writer.add_scalar(
                            f"psnr/num{i}",
                            batch_psnr(compressed.clamp(0.0, 1.0).to(self.device), input_image),
                            self.step,
                        )
                        self.writer.add_images(
                            f"compressed/num{i}",
                            compressed.clamp(0.0, 1.0),
                            self.step,
                        )
                        self.writer.add_images(
                            f"original/num{i}",
                            input_image,
                            self.step,
                        )
                    self.save()
                    print("-----test end-----")


