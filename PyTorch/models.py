import torch
import torch.nn as nn
import torch.nn.functional as F
from config import opt
from torch.autograd import Variable

# define generator
class generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        self.conv1 = nn.Conv2d(input_nc, ngf, 4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(ngf * 4)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(ngf * 8)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(ngf * 8)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(ngf * 8)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(ngf * 8)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(ngf * 8)

        self.conv1d = nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1d = nn.BatchNorm2d(ngf * 8)
        self.conv2d = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2d = nn.BatchNorm2d(ngf * 8)
        self.conv3d = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3d = nn.BatchNorm2d(ngf * 8)
        self.conv4d = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4d = nn.BatchNorm2d(ngf * 8)
        self.conv5d = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm5d = nn.BatchNorm2d(ngf * 4)
        self.conv6d = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm6d = nn.BatchNorm2d(ngf * 2)
        self.conv7d = nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm7d = nn.BatchNorm2d(ngf)
        self.conv8d = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):  # x : batch * 3  * 256 * 256
        x1 = self.conv1(x)  # x1: batch * 64 * 128 * 128
        x2 = self.batchnorm2(self.conv2(F.leaky_relu(x1, 0.2)))  # x2: batch * 128 * 64 * 64
        x3 = self.batchnorm3(self.conv3(F.leaky_relu(x2, 0.2)))  # x3: batch * 256 * 32 * 32
        x4 = self.batchnorm4(self.conv4(F.leaky_relu(x3, 0.2)))  # x4: batch * 512 * 16 * 16
        x5 = self.batchnorm5(self.conv5(F.leaky_relu(x4, 0.2)))  # x5: batch * 512 *  8 *  8
        x6 = self.batchnorm6(self.conv6(F.leaky_relu(x5, 0.2)))  # x6: batch * 512 *  4 *  4
        x7 = self.batchnorm7(self.conv7(F.leaky_relu(x6, 0.2)))  # x7: batch * 512 *  2 *  2
        x8 = self.batchnorm8(self.conv8(F.leaky_relu(x7, 0.2)))  # x8: batch * 512 *  1 *  1

        #noise = torch.FloatTensor(opt.batch_size, opt.ngf*8, 1, 1)
        #noise = Variable(noise)
        #noise.data.resize_(x8.size())
        #noise.data.normal_(0, 1)
        #print(type(x8), x8.size())
        #print(type(noise), noise.size())
        #x8.data.resize_(x8.size(0), x8.size(1)*2, x8.size(2), x8.size(3))
        #x8.data[0, x8.size(1) // 2:, :, :] = noise.data
        #d0 = torch.cat((noise, x8), 1)
        d1_ = F.dropout(self.batchnorm1d(self.conv1d(F.relu(x8))), 0.5, training=True)  # d1_: batch * 512 *  2 *  2
        d1 = torch.cat((d1_, x7), 1)  # d1:  batch * 1024 *  2 *  2
        d2_ = F.dropout(self.batchnorm2d(self.conv2d(F.relu(d1))), 0.5, training=True)  # d2_: batch * 512 *  4 *  4
        d2 = torch.cat((d2_, x6), 1)  # d2:  batch * 1024 *  4 *  4
        d3_ = F.dropout(self.batchnorm3d(self.conv3d(F.relu(d2))), 0.5, training=True)  # d3_: batch * 512 *  8 *  8
        d3 = torch.cat((d3_, x5), 1)  # d3:  batch * 1024 *  8 *  8
        d4_ = self.batchnorm4d(self.conv4d(F.relu(d3)))  # d4_: batch * 512 *  16 *  16
        d4 = torch.cat((d4_, x4), 1)  # d4:  batch * 1024 *  16 *  16
        d5_ = self.batchnorm5d(self.conv5d(F.relu(d4)))  # d5_: batch * 256 *  32 *  32
        d5 = torch.cat((d5_, x3), 1)  # d5:  batch * 512 *  32 *  32
        d6_ = self.batchnorm6d(self.conv6d(F.relu(d5)))  # d6_: batch * 128 *  64 *  64
        d6 = torch.cat((d6_, x2), 1)  # d6:  batch * 256 *  64 *  64
        d7_ = self.batchnorm7d(self.conv7d(F.relu(d6)))  # d6_: batch * 64 *  128 *  128
        d7 = torch.cat((d7_, x1), 1)  # d7:  batch * 128 *  128 *  128
        d8 = self.conv8d(F.relu(d7))  # d8:  batch * 3   *  256 *  256

        out = F.tanh(d8)
        return out


# define discriminator
class disciminator_WGAN(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(disciminator_WGAN, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ndf = ndf

        main = nn.Sequential()
        main.add_module('initial.conv',
                        nn.Conv2d(input_nc + output_nc, ndf, 4, stride=2, padding=1, bias=False))  # ndf * 128*128
        main.add_module('initial.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('extra1.conv', nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))  # ndf * 128*128
        main.add_module('extra1.batchnorm', nn.BatchNorm2d(ndf))
        main.add_module('extra1.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid1.conv', nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))  # 2ndf * 64 * 64
        main.add_module('pyramid1.batchnorm', nn.BatchNorm2d(ndf * 2))
        main.add_module('pyramid1.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid2.conv', nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))  # 4ndf * 32 * 32
        main.add_module('pyramid2.batchnorm', nn.BatchNorm2d(ndf * 4))
        main.add_module('pyramid2.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid3.conv', nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))  # 8ndf * 16 * 16
        main.add_module('pyramid3.batchnorm', nn.BatchNorm2d(ndf * 8))
        main.add_module('pyramid3.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid4.conv', nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))  # 8ndf * 8 * 8
        main.add_module('pyramid4.batchnorm', nn.BatchNorm2d(ndf * 8))
        main.add_module('pyramid4.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid5.conv', nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))  # 8ndf * 4 * 4
        main.add_module('pyramid5.batchnorm', nn.BatchNorm2d(ndf * 8))
        main.add_module('pyramid5.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('final.conv', nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))  # 1 * 1 * 1
        self.main = main

    def forward(self, x):
        # output = self.main(x)
        # return output
        output = nn.parallel.data_parallel(self.main, x, [0])
        return output.view(-1, 1)

class disciminator_GAN(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(disciminator_GAN, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ndf = ndf

        main = nn.Sequential()
        main.add_module('initial.conv',
                        nn.Conv2d(input_nc + output_nc, ndf, 4, stride=2, padding=1, bias=False))  # ndf * 128*128
        main.add_module('initial.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('extra1.conv', nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))  # ndf * 128*128
        main.add_module('extra1.batchnorm', nn.BatchNorm2d(ndf))
        main.add_module('extra1.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid1.conv', nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))  # 2ndf * 64 * 64
        main.add_module('pyramid1.batchnorm', nn.BatchNorm2d(ndf * 2))
        main.add_module('pyramid1.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid2.conv', nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))  # 4ndf * 32 * 32
        main.add_module('pyramid2.batchnorm', nn.BatchNorm2d(ndf * 4))
        main.add_module('pyramid2.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid3.conv', nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))  # 8ndf * 16 * 16
        main.add_module('pyramid3.batchnorm', nn.BatchNorm2d(ndf * 8))
        main.add_module('pyramid3.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid4.conv', nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))  # 8ndf * 8 * 8
        main.add_module('pyramid4.batchnorm', nn.BatchNorm2d(ndf * 8))
        main.add_module('pyramid4.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('pyramid5.conv', nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))  # 8ndf * 4 * 4
        main.add_module('pyramid5.batchnorm', nn.BatchNorm2d(ndf * 8))
        main.add_module('pyramid5.relu', nn.LeakyReLU(0.2, inplace=True))

        main.add_module('final.conv', nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))  # 1 * 1 * 1
        main.add_module('sigmoid', nn.Sigmoid())
        self.main = main

    def forward(self, x):
        # output = self.main(x)
        # return output
        output = nn.parallel.data_parallel(self.main, x, [0])
        return output.view(-1, 1)

