from dataloader import *
from models import * 
from tensorboardX import SummaryWriter
from Saveftns import * 
import torch
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import warnings
import tqdm
from Default_option import * 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

opt = TrainOptions()

writer = SummaryWriter('runs/' + opt.name + '_08')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.GPUs
device = torch.device('cuda')

encoder = Encoder()   
encoder = torch.nn.DataParallel(encoder)
encoder = encoder.cuda()

decoder = Decoder()   
decoder = torch.nn.DataParallel(decoder)
decoder = decoder.cuda()


criterion = nn.MSELoss().cuda()
params = list(encoder.parameters()) + list(decoder.parameters())
optim = Adam(params, lr = opt.lr )

trainSet = NoiseDataLoader(opt) 
train_loader = DataLoader(trainSet , batch_size= 5, shuffle = True , pin_memory = False)

testSet = NoiseDataLoader(opt , phase = 'test')
test_loader = DataLoader(testSet , batch_size = 3 , shuffle =True , pin_memory =False)

for i in range(opt.niter):
    train_losses = []
    val_losses = []
    
    for k,item in tqdm.tqdm(enumerate(train_loader,1)):
        name , inputs , target = item
                
        inputs = inputs.view(-1, 3, inputs.shape[-2] , inputs.shape[-1])
        target = target.view(-1, 3, target.shape[-2] , target.shape[-1])

        inputs , target = Variable(inputs).cuda().float() , Variable(target , requires_grad = False).cuda().float()

        output_vector = encoder(inputs)
        final_output = decoder(output_vector)

        optim.zero_grad()
        train_loss = criterion(final_output , target)
        train_loss.backward()
        optim.step()
        train_losses.append(train_loss.data)

    print_train_loss = np.nansum(train_losses) / len(train_losses)

    if i % 5 == 0:
        save_encoder(encoder,i,opt)
        save_decoder(decoder,i,opt)


    writer.add_scalar('data/train_loss' , print_train_loss  , i)
    print(' epoch : ' , i , ' Train Loss ' , print_train_loss)


    for k,item in tqdm.tqdm(enumerate(test_loader,1)):
        name , inputs , target = item

        inputs = inputs.view(-1,3, inputs.shape[-2] , inputs.shape[-1])
        target = target.view(-1,3, target.shape[-2] , target.shape[-1])

        inputs , target = Variable(inputs).cuda().float() , Variable(target , requires_grad = False).cuda().float()
            
        output_vector = encoder(inputs)
        final_output = decoder(output_vector)
            
        val_loss = criterion(final_output , target)
        val_losses.append(val_loss.data)

    print_val_loss = np.nansum(val_losses) / len(val_losses)

    writer.add_scalar('data/val_loss' , print_val_loss , i)
    print(' epoch : ' , i , ' Val Loss ' , print_val_loss)

    if i % 5 ==0:
        plt.subplot(1,3,1)
        plt.imshow(inputs[0,:].detach().cpu().numpy().reshape(opt.image_size,opt.image_size,3).astype(np.uint8))
        plt.subplot(1,3,2)
        plt.imshow(target[0,:].detach().cpu().numpy().reshape(opt.image_size,opt.image_size,3).astype(np.uint8))
        plt.subplot(1,3,3)
        plt.imshow(final_output[0,:].detach().cpu().numpy().reshape(opt.image_size,opt.image_size,3).astype(np.uint8))

        plt.savefig(name[0].split('/')[-1][:-4] + '_' + str(i) + '.png')