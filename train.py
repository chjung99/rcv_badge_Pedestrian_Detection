import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import argparse

from model_eff7fpndeconv import SSD300, MultiBoxLoss
from datasets import *
from utils import *
from torchcv.datasets.transforms import *
from torchcv.utils import run_tensorboard
from tensorboardX import SummaryWriter
import numpy as np

### Logging
import logging
import logging.handlers
from datetime import datetime
from tensorboardX import SummaryWriter

# Data parameters
  # folder with data files
DB_Root = './kaist-rgbt/'  
# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = 2  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None # path to model checkpoint, None if none
#checkpoint = './jobs/2020-05-19_01h57m_chanel+valsch/checkpoint_ssd300.pth.tar018'
#checkpoint='jobs/2020-05-13_05h15m_/checkpoint_ssd300.pth.tar003'
#checkpoint = './jobs/2020-05-12_15h56m_test/checkpoint_ssd300.pth.tar036'
batch_size =16 # batch size
start_epoch = 0  # start at this epoch
epochs = 150  # number of epochs to run without early-stopping
epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
best_loss = 100.  # assume a high loss at first
workers = 0  # number of workers for loading data in the DataLoader
print_freq = 10  # print training or validation status every __ batches
# lr = 1e-5  # learning rate
lr = 1e-4   # learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
port = 8815
cudnn.benchmark = True
jch=0
# random seed fix 
torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
torch.backends.cudnn.deterministic=True

parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--exp_time',   default=None, type=str,  help='set if you want to use exp time')
parser.add_argument('--exp_name',   default=None, type=str,  help='set if you want to use exp name')

args = parser.parse_args()

def main():
    """
    Training and validation.
    """

    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint,jch

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        sp=list()
        biases = list()
        not_biases = list()
        ''''''
        for param_name, param in model.named_parameters():
            
            if param.requires_grad:
                print(param_name)
                sp=param_name.split('.')
                
                if len(sp)>=5:
                    
                    if int(sp[3])>=22:
                        continue
                    else:
                        if param_name.endswith('.bias'):
                            
                            biases.append(param)
                        else:
                            
                            not_biases.append(param)
                else:
                    if len(sp)<=1:
                        continue
                    if (sp[2]=='_fc' or sp[2]=='_conv_head'or sp[2]=='_bn1'):
                        continue
                        
                    else:
                        if param_name.endswith('.bias'):
                            
                            biases.append(param)
                        else:
                            
                            not_biases.append(param)
                    
        
#         optimizer = torch.optim.RMSprop(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], lr=lr,weight_decay=weight_decay,  momentum=momentum)
        
        
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
        
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ int(epochs*0.5), int(epochs*0.75) ], gamma=0.1)
        
#         optimizer = torch.optim.RMSprop(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], lr=lr, momentum=momentum, weight_decay=weight_decay)
#         optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
#                                          mode='min',
#                                          factor=0.25,
#                                          patience=2,verbose=True )
 
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ int(epochs*0.5), int(epochs*0.75) ], gamma=0.1)
        
#         optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
#                                          mode='min',
#                                          factor=0.25,
#                                          patience=2,verbose=True )
        


    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    input_size = [512., 640.]
    
    preprocess1 = Compose([ ])    
    transforms1 = Compose([ \
                            ToTensor(), \
                            Normalize([0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R'), \
                            #Normalize([0.1598], [0.0813], 'T')                        
                            ])
    
    preprocess2 = Compose([ ])
    preprocess2 = Compose([  ColorJitter(0.3, 0.3, 0.3), ColorJitterLWIR(contrast=0.3)])
    transforms2 = Compose([ 
                            RandomHorizontalFlip(), \
                            RandomResizedCrop( [512,640], scale=(0.25, 4.0), ratio=(0.8, 1.2)), \
                            ToTensor(), \
                            Normalize( [0.3465,0.3219,0.2842], [0.2358,0.2265,0.2274], 'R')
                            ])
    
    train_dataset = KAISTPed('train-all-02.txt',img_transform=preprocess2, co_transform=transforms2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers,
                                               collate_fn=train_dataset.collate_fn, 
                                               pin_memory=True)  # note that we're passing the collate function here
#     val_dataset = KAISTPed('test-all-20.txt',img_transform=preprocess1, co_transform=transforms1)

#     val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                                                 num_workers=workers,
#                                                 collate_fn=test_dataset.collate_fn, 
#                                                 pin_memory=True)     
    
#     train_dataset = KAISTPed('for-train-ho0.txt',img_transform=preprocess2, co_transform=transforms2)

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                                num_workers=workers,
#                                                collate_fn=train_dataset.collate_fn, 
#                                                pin_memory=True)  # note that we're passing the collate function here
    
#     val_dataset = KAISTPed('for-val-ho0.txt',img_transform=preprocess1, co_transform=transforms1)

#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                                                num_workers=workers,
#                                                collate_fn=train_dataset.collate_fn, 
#                                                pin_memory=True)  # note that we're passing the collate function here
    
    #############################################################################################################################
    
    ### Set job directory

    if args.exp_time is None:
        args.exp_time        = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
    
    exp_name        = ('_' + args.exp_name) if args.exp_name else '_' 
    jobs_dir        = os.path.join( 'jobs', args.exp_time + exp_name )
    args.jobs_dir   = jobs_dir

    snapshot_dir    = os.path.join( jobs_dir, 'snapshots' )
    tensorboard_dir    = os.path.join( jobs_dir, 'tensorboardX' )
    if not os.path.exists(snapshot_dir):        os.makedirs(snapshot_dir)
    if not os.path.exists(tensorboard_dir):     os.makedirs(tensorboard_dir)
    run_tensorboard( tensorboard_dir, port )

    ### Backup current source codes
    
    import tarfile
    tar = tarfile.open( os.path.join(jobs_dir, 'sources.tar'), 'w' )
    tar.add( 'torchcv' )    
    tar.add( __file__ )

    import glob
    for file in sorted( glob.glob('*.py') ):
        tar.add( file )

    tar.close()

    ### Set logger
    
    writer = SummaryWriter(os.path.join(jobs_dir, 'tensorboardX'))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(levelname)s] [%(asctime)-11s] %(message)s')
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    logger.addHandler(h)

    h = logging.FileHandler(os.path.join(jobs_dir, 'log_{:s}.txt'.format(args.exp_time)))
    h.setFormatter(fmt)
    logger.addHandler(h)

    settings = vars(args)
    for key, value in settings.items():
        settings[key] = value   

    logger.info('Exp time: {}'.format(settings['exp_time']))
    for key, value in settings.items():
        if key == 'exp_time':
            continue
        logger.info('\t{}: {}'.format(key, value))

    logger.info('Preprocess for training')
    logger.info( preprocess2 )
    logger.info('Transforms for training')
    logger.info( transforms2 )

    #############################################################################################################################

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Paper describes decaying the learning rate at the 80000th, 100000th, 120000th 'iteration', i.e. model update or batch
        # The paper uses a batch size of 32, which means there were about 517 iterations in an epoch
        # Therefore, to find the epochs to decay at, you could do,
        # if epoch in {80000 // 517, 100000 // 517, 120000 // 517}:
        #     adjust_learning_rate(optimizer, 0.1)

        # In practice, I just decayed the learning rate when loss stopped improving for long periods,
        # and I would resume from the last best checkpoint with the new learning rate,
        # since there's no point in resuming at the most recent and significantly worse checkpoint.
        # So, when you're ready to decay the learning rate, just set checkpoint = 'BEST_checkpoint_ssd300.pth.tar' above
        # and have adjust_learning_rate(optimizer, 0.1) BEFORE this 'for' loop

        # One epoch's training
        optim_scheduler.step()
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger,
                           writer=writer)
        
#         import pdb;pdb.set_trace()
#         val_loss = validate(val_loader=val_loader,
#                             model=model,
#                             criterion=criterion,
#                             logger=logger,
#                             writer=writer)
        
#         optim_scheduler.step(val_loss)
        
        # Did validation loss improve?
        is_best = train_loss < best_loss
        best_loss = min(train_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            logger.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0
        
        # Save checkpoint

        writer.add_scalars('train/epoch', {'epoch_best_loss': best_loss},global_step=epoch )
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, train_loss, best_loss, is_best, jobs_dir)



def train(train_loader, model, criterion, optimizer, epoch, logger, writer):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss_sum
    losses_loc = AverageMeter()  # loss_loc
    losses_cls = AverageMeter()  # loss_cls

    start = time.time()
    
    # Batches
    for batch_idx, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
       
        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
        
        # Loss
        loss,cls_loss,loc_loss,n_positives = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        if np.isnan(loss.item()):
            import pdb
            pdb.set_trace()

            loss,cls_loss,loc_loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # losses.update(loss.item(), images.size(0))
        losses.update(loss.item())
        losses_loc.update(loc_loss)
        losses_cls.update(cls_loss)
        batch_time.update(time.time() - start)

        start = time.time()
        

        if batch_idx and batch_idx % print_freq == 0:         
            writer.add_scalars('train/loss', {'loss': losses.avg}, global_step=epoch*len(train_loader)+batch_idx )
            writer.add_scalars('train/loc', {'loss': losses_loc.avg}, global_step=epoch*len(train_loader)+batch_idx )                
            writer.add_scalars('train/cls', {'loss': losses_cls.avg}, global_step=epoch*len(train_loader)+batch_idx )

        # Print status
        if batch_idx % print_freq == 0:

            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'num of Positive {Positive}\t'.format(epoch, batch_idx, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, Positive=n_positives))

    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    return  losses.avg

def validate(val_loader, model, criterion,logger,writer):
    """
    One epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :return: average validation loss
    """
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    
    losses = AverageMeter()  # loss_sum
    losses_loc = AverageMeter()  # loss_loc
    losses_cls = AverageMeter()  # loss_cls
    start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels,_) in enumerate(val_loader):

            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            #loss,cls_loss,loc_loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            loss,cls_loss,loc_loss,n_positives = criterion(predicted_locs, predicted_scores, boxes, labels) 
            
            losses.update(loss.item())
            losses_loc.update(loc_loss)
            losses_cls.update(cls_loss)
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                logger.info('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    logger.info('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg


if __name__ == '__main__':

    main()
