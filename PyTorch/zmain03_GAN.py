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

netD = disciminator_GAN(opt.input_nc, opt.output_nc, opt.ndf)
netD.apply(weights_init)
if opt.netD != '': # load checkpoint if needed
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# loss function
criterion = nn.BCELoss()
criterionAE = nn.L1Loss()

# prepare data interface and cuda
realA = torch.FloatTensor(opt.batch_size, 3, 256, 256)
realB = torch.FloatTensor(opt.batch_size, 3, 256, 256)
noise = torch.FloatTensor(opt.batch_size, opt.ngf*8, 1, 1)
label = torch.FloatTensor(opt.batch_size)
one = torch.FloatTensor([1])
mone = one * -1
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionAE.cuda()
    realA, realB = realA.cuda(), realB.cuda()
    label = label.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise.cuda()

realA = Variable(realA)
realB = Variable(realB)
noise = Variable(noise)
label = Variable(label)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

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

        data, _ = data_iter.next()
        _realA, _realB = torch.chunk(data, 2, 3)
        realA.data.resize_(_realA.size()).copy_(_realA)
        realB.data.resize_(_realB.size()).copy_(_realB)
        i += 1

        # train with real
        netD.zero_grad()
        inputD = torch.cat((realA, realB), 1)
        label.data.resize_(opt.batch_size).fill_(real_label)

        output = netD(inputD)
        errD_real = criterion(output, label)
        errD_real.backward()

        # train with fake
        fakeA = netG(realB)
        inputD = torch.cat((fakeA, realB), 1)
        label.data.fill_(fake_label)
        output = netD(inputD.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network
        ############################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
        label.data.fill_(real_label)
        output = netD(inputD)
        errG = criterion(output, label)
        errG.backward(retain_variables=True)
        #L1loss
        errL1 = criterionAE(fakeA, realA).mul(opt.lamb)
        errL1.backward()
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d] L1 loss: %f Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
              % (epoch, opt.niter, gen_iterations, len(dataloader),
                 errL1.data[0], errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

        if gen_iterations % 1000 == 0:
            vutils.save_image(_realA, '{0}/real_samples.png'.format(opt.experiment))
            fakeA = netG(realB)
            vutils.save_image(fakeA.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

    # do checkpointing
    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))