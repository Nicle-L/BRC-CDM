import argparse, os
import torch
import numpy as np
import time
from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
from model import ScaleSpaceFlow
from video.dataset3 import DataSet1,testDataSet1
import torch.optim as optim
from torch.autograd import Variable
from collections import defaultdict
import math
from PIL import Image
from datetime import datetime
def adjust_learning_rate(optimizer, epoch, init_lr):
    if epoch <= 3000:
        lr = init_lr
    elif epoch <= 4000:
        lr = init_lr / 2
    elif 4000 < epoch <= 5000:
        lr = init_lr / 4
    else:
        lr = init_lr / 8
        if lr < 1e-6:
            lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def Var(x):
    return Variable(x.cuda())


def tensor_to_rgb_image(tensor, step, save_root_path):
    batch_size = tensor.size(0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_root_path, f'output_{timestamp}_step_{step}')
    os.makedirs(save_path, exist_ok=True)

    for i in range(batch_size):
        sample_tensor = tensor[i]
        scaled_data = (sample_tensor * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        if scaled_data.shape[0] == 3:
            scaled_data = np.transpose(scaled_data, (1, 2, 0))
        rgb_image = Image.fromarray(scaled_data)

        # 保存图像到指定路径
        output_path = os.path.join(save_path, f'output_image_{i}.png')
        rgb_image.save(output_path)

    return save_path

def min_max_normalize(tensor, epsilon=1e-8):
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    
    normalized_tensor = (tensor - min_val) / (max_val - min_val + epsilon)
    return normalized_tensor

net = ScaleSpaceFlow()
net.cuda()

def train(opt):
    # Loss:
    criterion_mse = nn.MSELoss()
    # Set module training
    net.train()
    train_dataset = DataSet1(path=opt.train_path)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=0)
    # Training iteration
    opt1 = optim.Adam(net.parameters(), lr=opt.lr)
    lamb = opt.lmbda
    for epoch in range(6000):
            last_time = time.time()
            sumloss = 0.0
            summse_train=0.0
            sum_bpp =0.0
            cur_lr1 = adjust_learning_rate(opt1, epoch, opt.lr)
            for step, input in enumerate(train_loader):
                  input_image, ref_image = input[0].cuda(), input[1].cuda()
                  output_dict  =net(input_image,ref_image)
                  bppb = output_dict["bppb"]
                  x_rec_list = output_dict["output"]
                  x_rec=x_rec_list[0]

                  distortion = criterion_mse(input_image,x_rec)
                  rd_loss = lamb* distortion + bppb
                  opt1.zero_grad()  # 清零第一个优化器的梯度
                  rd_loss.backward()
                  torch.nn.utils.clip_grad_norm_(net.parameters(),3)
                  opt1.step()
                  sumloss += rd_loss.item()
                  sum_bpp +=bppb.item()
                  summse_train  += distortion.item()
                  if  (step + 1)% 100 == 0:
                        if not os.path.exists('E:/ljh/video-test3/output/log'):
                                 os.mkdir('E:/ljh/video-test3/output/log')
                        with open(os.path.join(opt.out_dir_train, 'train_mse'+str((lamb))+'.log'), 'a') as fd:
                           mse = summse_train / (step + 1)
                           psnr = 10.0 * np.log10(1/ mse)
                           time_used = time.time()-last_time
                           last_time = time.time()
                           loss_total = sumloss / (step + 1)
                           bpp_total = sum_bpp / (step+1)
                           fd.write('ep:%d step:%d time:%.1f lr:%.6f loss:%.6f bppb_total:%.4f psnr:%.2f  \n'
                             %(epoch, step, time_used, cur_lr1, loss_total,  bpp_total,psnr))
                        fd.close()
                  print('epoch', epoch, 'step:', step,  'bppb:', bppb.item())
            if (epoch + 1) % 50 == 0:
                test(opt)
            if (epoch + 1) % opt.save_interval == 0:
                checkpoint_path = os.path.join(opt.checkpoint_dir,
                                               'model_epoch_{}_lamda{}.pth'.format(epoch + 1, (lamb)))
                torch.save(net.state_dict(), checkpoint_path)
                print('Model saved at', checkpoint_path)



 
def test(opt):
    test_dataset = testDataSet1(path=opt.test_path)
    test_loader = DataLoader(test_dataset, batch_size=opt.testBatchSize, shuffle=True, num_workers=4)
    net.eval()
    loss_func = nn.MSELoss()
    lamb = opt.lmbda
    with torch.no_grad():
        summse_test = 0
        sum_bpp_test = 0
        for step, input in enumerate(test_loader):
            input_image, ref_image = input[0].cuda(), input[1].cuda()
            output_dict = net(input_image, ref_image)
            bppb = output_dict["bppb"]
            x_rec_list = output_dict["output"]
            x_rec =x_rec_list[0]
            distortion = loss_func(input_image, x_rec)
            sum_bpp_test += bppb.item()
            summse_test += distortion.item()
            if (step + 1) % opt.test_size == 0:
                if not os.path.exists('E:/ljh/video-test3/output_test'):
                    os.mkdir('E:/ljh/video-test3/output_test')
                with open(os.path.join(opt.out_dir_test, 'test_mse' + str((lamb)) + '.log'), 'a') as fd:
                    mse = summse_test / (step + 1)
                    psnr = 10.0 * np.log10(1/ mse)
                    bpp_total = sum_bpp_test / (step + 1)
                    fd.write('step:%d MSE:%.6f bppb_total:%.4f psnr:%.2f \n'
                             % (step, mse, bpp_total, psnr))
                fd.close()
            print('step:', step, 'MSE:', distortion.item(), 'bppb:', bppb.item())
    net.train()

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='HSI Compression')
    # Model setting
    parser.add_argument('--bands', type=int, default=3)
    parser.add_argument('--train_size', type=int, default=7513)
    parser.add_argument('--test_size', type=int, default=1907)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--testBatchSize', type=int, default=1)
    parser.add_argument("--lambda", type=float, default=32768, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
    parser.add_argument('--out_dir_train', type=str, default='E:/ljh/video-test3/output/log')
    parser.add_argument('--out_dir_test', type=str, default='E:/ljh/video-test3/output_test')
    parser.add_argument('--checkpoint_dir', type=str, default='E:/ljh/video-test3/output/weight')
    parser.add_argument("--patchSize", type=int, default=256)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--train_path", type=str, default="E:/ljh/video-test3/data/train.txt")
    parser.add_argument("--test_path", type=str, default="E:/ljh/video-test3/data/test.txt")
    parser.add_argument('--cuda', action='store_true', help='use cuda?', default=True)
    parser.add_argument("--lr", type=float, default=4e-4, help="initial learning rate.")
    opt = parser.parse_args()
    print(opt)
    train(opt)
