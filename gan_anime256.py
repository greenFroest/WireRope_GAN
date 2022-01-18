# 哈尔滨工程大学
# 开发时间：2022/1/17 23:16

import torch
from torch import nn
import torchvision
from torchvision import transforms
import torch.utils.data as Data
import visdom
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os

to_tensor = ToTensor()  # 将图片转换成Tensor
to_pil = ToPILImage()  # 将Tensor转换成Image对象

'''基本配置'''


class Config(object):
    dataPath = 'E:/Code/PyCharm/WireRope/data/defect'
    numWorkers = 4
    imageSize = 256
    batchSize = 24  # batch size=128的梯度下降方法
    maxEpoch = 10000
    lr1 = 2e-4  # 生成器学习率
    lr2 = 2e-4  # 判别器学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    nz = 1000  # 随机操声维度
    ngf = 64  # generator feature map
    ndf = 64  # discriminator feature

    savePath = './saveimg/'

    useCuda = True
    vis = True  # 是否使用visdom
    env = 'GAN'  # visdom的env
    plotEvery = 2  # 每20batch，visdom画图一次

    dEvery = 1  # 判别器训练周期
    gEvery = 5  # 生成器训练周期
    decayEvery = 10  # 模型保存周期
    # netDpath = 'checkpoints/netD.pth'
    # netGpath = 'checkpoints/netG.pth'
    netDpath = 'E:/Code/PyCharm/WireRope/gan/checkpoint3/netd_180.pth'
    netGpath = 'E:/Code/PyCharm/WireRope/gan/checkpoint3/netg_180.pth'
    # netDpath = None
    # netGpath = None

    ganImg = 'result.png'
    ganNum = 64
    ganSearchNum = 512
    ganMean = 0  # 噪声均值
    ganStd = 1  # 噪声方差
    sol = 0.2  # LeakyReLU的斜率值
    pat = 0.5  # Momentum的patient


config = Config()
'''生成器'''


class Generator(nn.Module):
    def __init__(self, config):
        config = config
        super(Generator, self).__init__()
        self.out = nn.Sequential(
            # 100*1*1 --> (64*32)*4*4
            # ConvTranspose2d 是二维转置卷积
            nn.ConvTranspose2d(config.nz, config.ngf * 32, kernel_size=4, bias=False),
            nn.BatchNorm2d(config.ngf * 32),  # 批规范化  #如果不好加上0.5试试
            nn.ReLU(True),  # True为直接修改覆盖 ，节省内存

            # (64*32)*4*4 --> (64*16)*8*8
            nn.ConvTranspose2d(config.ngf * 32, config.ngf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ngf * 16),
            nn.ReLU(True),

            # (64*16)*8*8 --> (64*8)*16*16
            nn.ConvTranspose2d(config.ngf * 16, config.ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ngf * 8),
            nn.ReLU(True),

            # (64*8)*16*16 --> (64*4)*32*32
            nn.ConvTranspose2d(config.ngf * 8, config.ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ngf*4),
            nn.ReLU(True),

            # (64*4)*32*32 --> (64*2)*64*64
            nn.ConvTranspose2d(config.ngf * 4, config.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ngf * 2),
            nn.ReLU(True),

            # (64*2)*64*64 --> 64*128*128
            nn.ConvTranspose2d(config.ngf * 2, config.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ngf),
            nn.ReLU(True),

            # 64*128*128 --> 3*256*256
            nn.ConvTranspose2d(config.ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.out(x)


'''判别器'''


class Discriminator(nn.Module):
    def __init__(self, config):
        config = config
        super(Discriminator, self).__init__()
        self.out = nn.Sequential(
            # 3*256*256 --> 64*128*128
            nn.Conv2d(3, config.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(config.sol, True),

            # 64*128*128 --> (64*2)*64*64
            nn.Conv2d(config.ndf, config.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ndf * 2),
            nn.LeakyReLU(config.sol, True),

            # (64*2)*64*64 --> (64*4)*32*32
            nn.Conv2d(config.ndf * 2, config.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ndf * 4),
            nn.LeakyReLU(config.sol, True),

            # (64*4)*32*32 --> (64*8)*16*16
            nn.Conv2d(config.ndf * 4, config.ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ndf * 8),
            nn.LeakyReLU(config.sol, True),

            # (64*8)*16*16 --> (64*16)*8*8
            nn.Conv2d(config.ndf * 8, config.ndf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ndf * 16),
            nn.LeakyReLU(config.sol, True),

            # (64*16)*8*8 --> (64*32)*4*4
            nn.Conv2d(config.ndf * 16, config.ndf * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.ndf * 32),
            nn.LeakyReLU(config.sol, True),

            # (64*32)*4*4 --> 1 * 1 * 1
            nn.Conv2d(config.ndf * 32, 1, kernel_size=4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.out(x).view(-1)

'''准备数据'''
tfs = transforms.Compose([
    transforms.Resize(config.imageSize),  # 改成（size * size）
    # transforms.CenterCrop(config.imageSize),  # 中心切割
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

'''这里的数据放在./data/faces/下，注意dataPath = "./data" 
    这样ImageFolder判断faces下所有图片为一类'''
trainset = torchvision.datasets.ImageFolder(config.dataPath, transform=tfs)

trainloader = Data.DataLoader(
    trainset,
    batch_size=config.batchSize,
    shuffle=True,
    num_workers=config.numWorkers,
    drop_last=True
)

if __name__ == '__main__':
    map_location = lambda storage, loc: storage
    netG = Generator(config)  # 生成器
    netD = Discriminator(config)  # 判别器
    if config.vis:
        vis = visdom.Visdom(env=config.env)
    if config.netDpath:
        netD.load_state_dict(torch.load(config.netDpath, map_location=map_location))
    if config.netGpath:
        netG.load_state_dict(torch.load(config.netGpath, map_location=map_location))
    optG = torch.optim.Adam(netG.parameters(), config.lr1, betas=(config.beta1, 0.999))  # 生成器优化器
    optD = torch.optim.Adam(netD.parameters(), config.lr2, betas=(config.beta1, 0.999))  # 判别器优化器
    loss_func = torch.nn.BCELoss()

    true_labels = torch.ones(config.batchSize)  # 真图片为1
    false_labels = torch.zeros(config.batchSize)  # 假图片为0
    fix_noises = torch.randn(config.batchSize, config.nz, 1, 1)  # batch组 nz*1*1的数据
    noises = torch.randn(config.batchSize, config.nz, 1, 1)  # 随机生成噪声

    '''判断是否使用GPU'''
    if config.useCuda:
        netD.cuda()
        netG.cuda()
        loss_func.cuda()
        true_labels, false_labels = true_labels.cuda(), false_labels.cuda()
        fix_noises, noises = fix_noises.cuda(), noises.cuda()

    PATH = ''
    for i in range(100):
        PATH = 'E:/Code/PyCharm/WireRope/gan/checkpoint{}'.format(i)
        if os.path.exists(PATH):
            continue
        else:
            os.mkdir(PATH)
            break

    '''开始训练'''
    for epoch in range(config.maxEpoch):
        for i, (img, _) in enumerate(trainloader):
            print(10 * '*')
            print('start {} epochs'.format(epoch))
            real_img = img
            if config.useCuda:
                real_img = real_img.cuda()

            # 训练判别器
            if (i + 1) % config.dEvery == 0:
                optD.zero_grad()
                out = netD(real_img)  # 尽可能把真的图片判别为1
                loss_real = loss_func(out, true_labels)
                loss_real.backward()

                noises.data.copy_(torch.randn(config.batchSize, config.nz, 1, 1))
                fake_img = netG(noises).detach()  # 生成假图片 detach是切断求导关联
                fake_out = netD(fake_img)  # 尽可能把假的图片判别为0
                loss_fake = loss_func(fake_out, false_labels)
                loss_fake.backward()
                optD.step()

            # 训练生成器
            if i % config.gEvery == 0:
                optG.zero_grad()
                noises.data.copy_(torch.randn(config.batchSize, config.nz, 1, 1))
                fake_img = netG(noises)  # 尽可能让噪声为真，让判别器把假的图片判为1
                fake_out = netD(fake_img)
                loss_fake = loss_func(fake_out, true_labels)
                loss_fake.backward()
                optG.step()
            # '''
            # 这段代码不够成熟请忽略
            # if i %config.plotEvery == config.plotEvery - 1:
            #     #可视化
            #     fix_fake_imgs = netG(fix_noises)
            #     fix_fake_imgs = fix_fake_imgs.data.cpu()[:1] * 0.5 + 0.5
            #     check_real_img = real_img.data.cpu()[:1] * 0.5 + 0.5
            #
            #     to_pil(fix_fake_imgs.squeeze())
            #     to_pil(check_real_img.squeeze())
            # '''
            print('{} epochs {} times'.format(epoch + 1, i + 1))
        print('the {} epoches end'.format(epoch + 1))
        print(10 * '*')

        if epoch % config.decayEvery == 0:
            # 保存模型、图片
            # fix_fake_imgs = fix_fake_imgs.data.cpu()[:1] * 0.5 + 0.5
            # to_pil(fix_fake_imgs.squeeze()).save('%s/%s.png' % (config.savePath, epoch))

            torch.save(netD.state_dict(), PATH + '/netd_%s.pth' % epoch)
            torch.save(netG.state_dict(), PATH + '/netg_%s.pth' % epoch)

            optG = torch.optim.Adam(netG.parameters(), config.lr1, betas=(config.beta1, 0.999))
            optD = torch.optim.Adam(netD.parameters(), config.lr2, betas=(config.beta1, 0.999))
            fix_imgs = netG(fix_noises)
            vis.images(fix_imgs.data.cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')

    # netG = Generator(config)  # 生成器
    # netG.load_state_dict(torch.load('./netg_160.pth'))
    # rand_img = netG(torch.randn(1, config.nz, 1, 1))
    # rand_img = rand_img.data.cpu()[:1] * 0.5 + 0.5
    # to_pil(rand_img.squeeze()).show()
    # '''本段代码是在jupyter notebook上跑，因此会自动输出Image对象，
