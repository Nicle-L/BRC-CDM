import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.uniform import Uniform
from torch.nn.parameter import Parameter
    
#自动求导函数
class Low_bound(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x) # 保存张量
        x = torch.clamp(x, min=1e-6) #clamp截断 小于min的值赋min，其他不变
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        #对于 pass_through_if 数组的每个元素，如果在 x 对应位置的值大于等于 1e-6 或者在 g 对应位置的值小于 0.0，那么该位置的元素值为 True，否则为 False
        #这个数组将决定是否将对应位置的梯度传递到上一层的梯度计算中
        pass_through_if = np.logical_or( #logical_or逻辑或
            x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if+0.0).cuda()
        return grad1*t

class UniverseQuant(torch.autograd.Function):
    @staticmethod #装饰器，将一个方法转换为静态方法，意味着这个方法不依赖于类的实例，可以直接通过类名调用，而不需要创建对象
    #在定义静态方法时，不需要传递 self 或 cls 参数。静态方法不能访问或修改类的属性，也不能调用非静态的实例方法。
    def forward(ctx, x):
        #b = np.random.uniform(-1,1)
        b = 0
        #创建了一个-0.5*2^b到0.5*2^b的随机噪声uniform_distribution，并将其添加到输入张量 x 上，实现量化操作
        uniform_distribution = Uniform(-0.5*torch.ones(x.size())#torch.ones创建一个全为1的张量 Uniform()生成服从均匀分布的随机数
                                       * (2**b), 0.5*torch.ones(x.size())*(2**b)).sample().cuda()#sample()从概率分布中抽取一个随机样本
        return torch.round(x+uniform_distribution)-uniform_distribution#round()四舍五入 减去随机噪声张量uniform_distribution为了消除添加的随机噪声，使量化操作可逆

    @staticmethod
    def backward(ctx, g):

        return g

        
#熵瓶颈层 进一步提取特征
class Entropy_bottleneck(nn.Module):
    def __init__(self, channel, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-6, tail_mass=1e-9, optimize_integer_offset=True):
        super(Entropy_bottleneck, self).__init__()

        self.filters = tuple(int(t) for t in filters) # self.filters元组(3, 3, 3)
        self.init_scale = float(init_scale)# self.init_scale=10
        self.likelihood_bound = float(likelihood_bound) # self.likelihood_bound=1e-6
        self.tail_mass = float(tail_mass) # self.tail_mass=1e-9
        self.optimize_integer_offset = bool(optimize_integer_offset) # self.optimize_integer_offset = True
        if not 0 < self.tail_mass < 1:
            raise ValueError("`tail_mass` must be between 0 and 1")#检查tail_mass是否在0-1之间
        filters = (1,) + self.filters + (1,)#filters = (1,3,3,3,1) 元组(1,) 整数(1)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1)) # scale = 10^(1/6)
        self._matrices = nn.ParameterList([])   #_matrices矩阵 
        self._bias = nn.ParameterList([])       #_bias偏置
        self._factor = nn.ParameterList([])     #_factor因子
        #管理nn.Parameter类型的参数列表，该列表将会自动处理这些参数，而不需要我们手动指定要优化的参数
        # print ('scale:',scale)
        for i in range(len(self.filters) + 1): #0 1 2 3

            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))#log(e^(1.0 / scale / filters[i + 1])-1)

            self.matrix = Parameter(torch.FloatTensor( #创建一个大小为 (channel, filters[i + 1], filters[i]) 的 torch.FloatTensor 张量
                channel, filters[i + 1], filters[i]))  #并将其封装为 Parameter 对象，以便将其视为模型的可训练参数
            self.matrix.data.fill_(init)#matrix填充为init
            self._matrices.append(self.matrix)#将矩阵参数添加到 _matrices 参数列表中

            self.bias = Parameter(
                torch.FloatTensor(channel, filters[i + 1], 1))
            noise = np.random.uniform(-0.5, 0.5, self.bias.size())#np.random.uniform生成服从均匀分布的随机数
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)#copy_用一个张量赋值另一个张量
            self._bias.append(self.bias)

            if i < len(self.filters):
                self.factor = Parameter(
                    torch.FloatTensor(channel, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factor.append(self.factor)
    #计算累积的对数概率
    def _logits_cumulative(self, logits, stop_gradient): 
        for i in range(len(self.filters) + 1):
            matrix = f.softplus(self._matrices[i])#softplus函数f(x) = log(1 + exp(x))
            if stop_gradient:
                matrix = matrix.detach()#用于创建一个新的张量，其值与原始张量 matrix 相同，但是与计算图没有关联，即新的张量没有梯度信息
            logits = torch.matmul(matrix, logits)#matmul矩阵乘

            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self._factor):
                factor = f.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * f.tanh(logits)
        return logits # logits=factor*tanh(matrix*logits+bias)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x, training):
        x = x.permute(1, 0, 2, 3).contiguous() # x:(b,128,4,4)-> (128,b,4,4)
        #  contiguous()用于检查张量是否是内存中连续存储,如果一个张量是连续的，意味着它的数据在内存中是按照一维数组的方式存储的，没有间隔或跳跃
        shape = x.size() 
        x = x.view(shape[0], 1, -1) 
        if training == 0:
            x = self.add_noise(x)
        elif training == 1:
            x = UniverseQuant.apply(x)
        else:
            x = torch.round(x)
        lower = self._logits_cumulative(x - 0.5, stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5, stop_gradient=False)
        sign = -torch.sign(torch.add(lower, upper))# -1 0 1
        sign = sign.detach()
        likelihood = torch.abs(f.sigmoid(sign * upper) - f.sigmoid(sign * lower))
        if self.likelihood_bound > 0:
            likelihood = Low_bound.apply(likelihood)
        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0, 2, 3)
        x = x.view(shape) 
        x = x.permute(1, 0, 2, 3) 
        return x, likelihood 
    
    
#计算y的概率分布
class Distribution_for_entropy(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy, self).__init__()

    def forward(self, x, p_dec):
        
        c = p_dec.size()[1]
        mean  = p_dec[:, :c//2, :, :]
        scale = p_dec[:, c//2:, :, :]

    # to make the scale always positive
        scale = torch.clamp(scale,min = 1e-9)
        m1 = torch.distributions.normal.Normal(mean, scale)#创建了一个正态分布（Normal distribution）的概率分布对象
        #累积分布函数（CDF）是描述随机变量小于或等于给定值的概率的函数，这些概率值用于计算对数似然函数，从而用于量化和编码的概率建模过程
        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)#将小于阈值的概率值设为1e-6
        return x, likelihood


class Distribution_for_entropy3(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy3, self).__init__()

    def forward(self, x, p_dec):

        mean = p_dec[:, 0, :, :, :]
        scale = p_dec[:, 1, :, :, :]

    # to make the scale always positive
        scale[scale == 0] = 1e-9
    #scale1 = torch.clamp(scale1,min = 1e-6)
        m1 = torch.distributions.normal.Normal(mean, scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)

        #sign = -torch.sign(torch.add(lower, upper))
        #sign = sign.detach()
        #likelihood = torch.abs(f.sigmoid(sign * upper) - f.sigmoid(sign * lower))
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood


class Distribution_for_entropy2(nn.Module):
    def __init__(self):
        super(Distribution_for_entropy2, self).__init__()

    def forward(self, x, p_dec): # (b,192,16,16) (b,9,192,16,16)
        # you can use use 3 gaussian
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = [ # (b,1,192,16,16)
            torch.chunk(p_dec, 9, dim=1)[i].squeeze(1) for i in range(9)]
        # keep the weight  summation of prob == 1
        probs = torch.stack([prob0, prob1, prob2], dim=-1)
        probs = f.softmax(probs, dim=-1)
        # process the scale value to non-zero
        scale0 =scale0.clamp(1e-6,1e10)
        scale1 =scale1.clamp(1e-6,1e10)
        scale2 =scale2.clamp(1e-6,1e10)
        # scale1[scale1 == 0] = 1e-6
        # scale2[scale2 == 0] = 1e-6
        # 3 gaussian distribution
        m0 = torch.distributions.normal.Normal(mean0, scale0)
        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)

        likelihood0 = torch.abs(m0.cdf(x + 0.5)-m0.cdf(x-0.5))
        likelihood1 = torch.abs(m1.cdf(x + 0.5)-m1.cdf(x-0.5))
        likelihood2 = torch.abs(m2.cdf(x + 0.5)-m2.cdf(x-0.5))

        likelihoods = Low_bound.apply(
            probs[:, :, :, :, 0]*likelihood0+probs[:, :, :, :, 1]*likelihood1+probs[:, :, :, :, 2]*likelihood2)

        return likelihoods


class Laplace_for_entropy(nn.Module):
    def __init__(self):
        super(Laplace_for_entropy, self).__init__()

    def forward(self, x, p_dec):
        mean = p_dec[:, 0, :, :, :]
        scale = p_dec[:, 1, :, :, :]

    # to make the scale always positive
        scale[scale == 0] = 1e-9
    #scale1 = torch.clamp(scale1,min = 1e-6)
        m1 = torch.distributions.laplace.Laplace(mean, scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)
        likelihood = torch.abs(upper - lower)

        likelihood = Low_bound.apply(likelihood)
        return likelihood