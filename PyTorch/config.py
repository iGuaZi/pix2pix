from easydict import EasyDict as edict
opt = edict()
# data params
opt.dataroot = '/hd3/yekui/facades/'
opt.workers = 2

# network params
opt.input_nc = 3
opt.output_nc = 3
opt.ngf = 64
opt.ndf = 64
opt.netG = ''
opt.netD = ''

# training params
opt.lamb = 100
opt.adam = False
opt.cuda = True
opt.niter = 100     # number of epochs
opt.Diters = 25    # train the discriminator Diters times
opt.experiment = 'zmain03'
opt.clamp_lower = -0.01
opt.clamp_upper =  0.01
opt.batch_size = 1
opt.lr = 0.0002
opt.beta1 = 0.8
