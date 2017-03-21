#######__info__#######
# WGAN
# one D updates & one G update

import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.utils as vutils
from config import opt
from models import *

# prepare dataloader
dataset = datasets.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

# initialize generator and discriminator
## custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = generator(opt.input_nc, opt.output_nc, opt.ngf)
netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = disciminator(opt.input_nc, opt.output_nc, opt.ndf)
netD.apply(weights_init)
if opt.netD != '': # load checkpoint if needed
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# prepare data interface and cuda
realA = torch.FloatTensor(opt.batch_size, 3, 256, 256)
realB = torch.FloatTensor(opt.batch_size, 3, 256, 256)
noise = torch.FloatTensor(opt.batch_size, opt.ngf*8, 1, 1)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    realA, realB = realA.cuda(), realB.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise.cuda()

realA = Variable(realA)
realB = Variable(realB)
noise = Variable(noise)

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)

# training
gen_iterations = 0
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ############################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        # clamp parameters to a cube
        for p in netD.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        data, _ = data_iter.next()
        _realA, _realB = torch.chunk(data, 2, 3)
        realA.data.resize_(_realA.size()).copy_(_realA)
        realB.data.resize_(_realB.size()).copy_(_realB)
        i += 1

        # train with real
        netD.zero_grad()
        inputD = torch.cat((realA, realB), 1)
        errD_real = netD(inputD)
        errD_real.backward(one)

        # train with fake
        noise.data.normal_(0, 1)
        fakeA = netG(realB)
        print(type(fakeA), fakeA.size())
        inputD = torch.cat((fakeA, realB), 1)
        errD_fake = netD(inputD)
        errD_fake.backward(mone)
        errD = errD_real - errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network
        ############################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
        noise.data.resize_(opt.batch_size, opt.ngf*8, 1, 1)
        noise.data.normal_(0, 1)
        fakeA = netG(realB)
        inputD = torch.cat((fakeA, realB), 1)
        errG = netD(inputD)
        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
              % (epoch, opt.niter, gen_iterations, len(dataloader),
                 errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

        if gen_iterations % 100 == 0:
            vutils.save_image(_realA, '{0}/real_samples.png'.format(opt.experiment))
            fakeA = netG(realB)
            vutils.save_image(fakeA.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

    # do checkpointing
    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))