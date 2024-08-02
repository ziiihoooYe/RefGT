import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='RefGT')


### log setting
parser.add_argument('--save_dir', type=str, default='save_temp',
                    help='Directory to save log, arguments, models and images')
parser.add_argument('--log_file_name', type=str, default='RefGT.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='RefGT',
                    help='Logger name')

### dataset setting
parser.add_argument('--dataset', type=str, default='BDD100K',                                                          
                    help='Which dataset to train and test')
parser.add_argument('--dataset_dir', type=str, default='data',                                                         
                    help='Directory of dataset')
parser.add_argument('--resize_img', type=str2bool, default=False,                                                         
                    help='Whether to resize img to Height=Width')

### dataloader setting
parser.add_argument('--num_workers', type=int, default=0,
                    help='The number of workers when loading data')

### network params setting
parser.add_argument('--mean_grad', type=str2bool, default=False,
                    help='the learnability of the mean shift layer parameters')
parser.add_argument('--lte_grad', type=str2bool, default=True,
                    help='the learnability of the vgg lte layer')
parser.add_argument('--res_depth', type=str, default='8+8+4+2',
                    help='The number of residual blocks in each stage')
parser.add_argument('--n_feats', type=int, default=64,
                    help='The number of channels in network')
parser.add_argument('--res_scale', type=float, default=1.,
                    help='Residual scale')

### baseline setting
parser.add_argument('--baseline', type=str, default='PReNet', help='baseline name')
parser.add_argument('--baseline_module', type=str, help='baseline model directory') # baseline.model.Uformer.model
parser.add_argument('--baseline_state_dir', type=str,help='baseline state dic')


### optimizer setting
parser.add_argument('--beta1', type=float, default=0.9,
                    help='The beta1 in Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='The beta2 in Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='The eps in Adam optimizer')
parser.add_argument('--lr_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr_rate_lte', type=float, default=1e-5,
                    help='Learning rate of LTE')
parser.add_argument('--decay', type=float, default=999999,
                    help='Learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Learning rate decay factor for step decay')

### loss setting
parser.add_argument('--psnr_loss', type=str2bool, default=False,
                    help='Whether to use PSNR loss')
parser.add_argument('--ssim_loss', type=str2bool, default=False,
                    help='Whether to use SSIM loss')
parser.add_argument('--ms_ssim_l1_loss', type=str2bool, default=True,
                    help='Whether to use ms_ssim_l1 loss')
parser.add_argument('--rec_loss_type', type=str, default='l1',
                    help='reconstruction loss type: l1 or l2')


### training setting
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
parser.add_argument('--gt_init_epochs', type=int, default=10, help='The number of ground truth init epochs')
parser.add_argument('--num_epochs', type=int, default=200, help='The number of training epochs')
parser.add_argument('--print_every_batch', type=int, default=50, help='Print period')
parser.add_argument('--save_every_epoch', type=int, default=10, help='Save period')
parser.add_argument('--val_every_epoch', type=int, default=10, help='Validation period')
parser.add_argument('--gt_ref', type=str2bool, default=True, help='Whether to use ground truth images as references ')


### evaluate / test / fine tune setting
parser.add_argument('--continue_training', type=str2bool, default=False,
                    help='whether to load previous training model to continue training')
parser.add_argument('--eval', type=str2bool, default=False,
                    help='Evaluation mode')
parser.add_argument('--eval_save_results', type=str2bool, default=False,
                    help='Save each image during evaluation')
parser.add_argument('--model_path', type=str, default='save_result/100L30.pt',
                    help='The path of model to evaluation')
parser.add_argument('--test', type=str2bool, default=False,
                    help='Test model')


args = parser.parse_args()
