import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from model.pointnet2 import PointNet2_modified
from model.pointnet import PointNetDenseCls,PointNetLoss
from data_utils.data_loader import loadData
from utils import nufftDataLoader
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser('PointNet2')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=201, help='number of epochss for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--jitter', default=False, help="randomly jitter point cloud")
    parser.add_argument('--step_size', type=int, default=20, help="randomly rotate point cloud")
    parser.add_argument("--num_train_samples",type=int, default=21118, help="number of tranining samples")
    parser.add_argument("--num_test_samples",type=int, default=2300, help="number of test samples")
    parser.add_argument('--model_name', type=str, default='pointnet2', help='Name of model')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--skip_training", default=False, type=bool)
    return parser.parse_args()

def main(args):
    print("Number of GPUs detected %d" %torch.cuda.device_count())
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else args.multi_gpu
    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/%sPartSeg-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))    
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_partseg.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    TRAIN_DATASET = nufftDataLoader(args.num_train_samples)
    TEST_DATASET = nufftDataLoader(args.num_test_samples,cat="test")
    dataloader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=int(args.workers))
     
    testdataloader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=int(args.workers))
    print("The number of training data is:",len(TRAIN_DATASET))
    logger.info("The number of training data is:%d",len(TRAIN_DATASET))
    print("The number of test data is:", len(TEST_DATASET))
    logger.info("The number of test data is:%d", len(TEST_DATASET))

    blue = lambda x: '\033[94m' + x + '\033[0m'
    #model = PointNet2()
    model = PointNet2_modified() 
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    pretrain = args.pretrain
    init_epochs = int(pretrain[-14:-11]) if args.pretrain is not None else 0


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    '''GPU selection and multi-GPU'''
    

    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        device_ids = []
        
        #torch.distributed.init_process_group(backend='nccl',init_method='env://')
        #model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.DataParallel(model)
        print("")
    else:
        model.cuda()
    MSE  = torch.nn.MSELoss()
    LEARNING_RATE_CLIP = 1e-5

    #history = defaultdict(lambda: list())
    bestLoss = 99
    #summary(model,[(3,256*256),(3,256*256)])
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %d"%pytorch_total_params)
    if args.skip_training =="":
        for epochs in range(init_epochs,args.epochs):
            epochsLoss = []
            scheduler.step()
            
            lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)
            print('Learning rate:%f' % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            for i, data in tqdm(enumerate(dataloader, 0),total=len(dataloader),smoothing=0.9):
                train_input, train_Gtruth = data
                coor = train_input[:,0,:,:]
                feuture  = train_input[:,1,:,:]

                coor = Variable(coor)
   
                feuture = Variable(feuture)
          
                #points = points.transpose(2, 1)
                #norm_plt = norm_plt.transpose(2, 1)
                coor, feuture = coor.cuda(), feuture.cuda()
                train_Gtruth = train_Gtruth.cuda()

                optimizer.zero_grad()
                model = model.train()

                pred = model(coor, feuture)
                loss = MSE(pred,train_Gtruth)
                #print('Batch %d with MSE: %f  ' % (i, loss))
                epochsLoss.append(loss)
                #history['loss'].append(loss.cpu().data.numpy())
                loss.backward()
                optimizer.step()
        
            avg_loss  =sum(epochsLoss) / len(epochsLoss)
            print('epochs %d %s AVG MSE: %f  ' % (epochs, blue('train'),avg_loss ))
            logger.info('epochs %d %s AVG MSE: %f  ' % (epochs, blue('train'),avg_loss ))
        #test_metrics, test_hist_acc, cat_mean_iou = test_partseg(model.eval(), testdataloader, seg_label_to_cat,50,forpointnet2)
        model.eval()
        test_avg_mse = []
        for batch_id, testData in tqdm(enumerate(testdataloader), total=len(testdataloader), smoothing=0.9):
            test_input, test_Gtruth  = testData
            coor = test_input[:,0,:,:]
            feuture  = test_input[:,1,:,:]
      
            feuture = feuture.flaot()
            coor , feuture = coor.cuda(), feuture.cuda()
            test_Gtruth = test_Gtruth.cuda()
   
            
            pred = model(coor, feuture)
            test_avg_mse.append(MSE(pred,test_Gtruth))
        print('epochs %d %s AVG MSE: %f  ' % (epochs, blue('test'),sum(test_avg_mse) / len(test_avg_mse) ))
        logger.info('epochs %d %s AVG MSE: %f  ' % (epochs, blue('test'),sum(test_avg_mse) / len(test_avg_mse)) )
        


        if avg_loss < bestLoss:
            bestLoss = sum(test_avg_mse) / len(test_avg_mse)
            torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth' % (checkpoints_dir,args.model_name, epochs, bestLoss))
            logger.info('Save model..')
            print('Save model..')
     

        print('Best AVG Loss is: %.5f'%bestLoss)
        logger.info('Best AVG Loss is: %.5f'%bestLoss)

    pred = np.zeros((args.batch_size,512,256))


    #____________________saving predicitons
    for i, data in enumerate(dataloader, 0):
        train_input, train_Gtruth = data
        coor = train_input[:,0,:,:]
        feuture  = train_input[:,1,:,:]

        coor = Variable(coor)
        
        feuture = Variable(feuture)
        

        coor, feuture = coor.cuda(), feuture.cuda()
        train_Gtruth = train_Gtruth.cuda()
        
        optimizer.zero_grad()
        model.eval()
        toch_prediction = model(coor, feuture)

        pred = toch_prediction.numpy()
        break
    train_input.numpy()
    train_input.save("predictions/input.npy")
    for i in range(args.batch_size):

        output = pred[i]
        output = np.reshape(output, (512, 256))
        output = output * 255
        output[np.newaxis,...]
        output_gt = train_Gtruth[i]
        output_gt[np.newaxis,...]
        output = np.concatenate([output,output_gt],axis=0)
        np.save('predictions/prediction%d.npy'%(i+1),output)

    for i, data in enumerate(testdataloader, 0):
        test_input, test_Gtruth = data
        coor = test_input[:,0,:,:]
        feuture  = test_input[:,1,:,:]

        coor = Variable(coor)
        
        feuture = Variable(feuture)
        

        coor, feuture = coor.cuda(), feuture.cuda()
        test_Gtruth = test_Gtruth.cuda()
        
        optimizer.zero_grad()
        model.eval()
        toch_prediction = model(coor, feuture)

        pred = toch_prediction.numpy()
        break
    test_input.numpy()
    test_input.save("predictions/input.npy")
    for i in range(args.batch_size):

        output = pred[i]
        output = np.reshape(output, (512, 256))
        output = output * 255

        output[np.newaxis,...]
        output_gt = test_Gtruth[i]
        output_gt[np.newaxis,...]
        output = np.concatenate([output,output_gt],axis=0)

        np.save('predictions/test_prediction%d.npy'%(i+1),output)



if __name__ == '__main__':
    args = parse_args()
    main(args)

