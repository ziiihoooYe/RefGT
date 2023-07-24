import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='DRTT')

### log setting
parser.add_argument('--save_dir', type=str, default='results_ppt',
                    help='Directory to save log, arguments, models and images')
parser.add_argument('--reset', type=str2bool, default=True,
                    help='Delete save_dir to create a new one')
parser.add_argument('--log_file_name', type=str, default='DRTT.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='DRTT',
                    help='Logger name')

### device setting
parser.add_argument('--cpu', type=str2bool, default=False,
                    help='Use CPU to run code')
parser.add_argument('--num_gpu', type=int, default=7,                                                                   ### baseline gpu index
                    help='The number of GPU used in training')

### dataset setting
parser.add_argument('--dataset', type=str, default='BDD100K',                                                          ############### dataset name
                    help='Which dataset to train and test')
parser.add_argument('--dataset_dir', type=str, default='data',                                                          ############### dataset directory
                    help='Directory of dataset')

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

### backbone setting
parser.add_argument('--backbone', type=str, default='PReNet',                                                           ############### backbone name
                    help='Backbone name')
parser.add_argument('--backbone_module', type=str, default='baseline.model.PReNet.networks',
                    help='Backbone model directory')
parser.add_argument('--backbone_state_dir', type=str, default='baseline/state_dict/PReNet6/BDD100K',                   ############### backbone directory
                    help='Backbone state dic')
parser.add_argument('--backbone_device', type=str, default='cuda:1',                                                    ### baseline gpu index
                    help='Backbone device')

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
parser.add_argument('--psnr_loss', type=str2bool, default=True,
                    help='Whether to use PSNR loss')
parser.add_argument('--ssim_loss', type=str2bool, default=False,
                    help='Whether to use SSIM loss')
parser.add_argument('--ms_ssim_l1_loss', type=str2bool, default=True,
                    help='Whether to use ms_ssim_l1 loss')
parser.add_argument('--rec_loss_type', type=str, default='l1',
                    help='reconstruction loss type: l1 or l2')

### training setting
parser.add_argument('--batch_size', type=int, default=1,                                                                ### batch size
                    help='Training batch size')
parser.add_argument('--gt_init_epochs', type=int, default=20,
                    help='The number of ground truth init epochs')
parser.add_argument('--num_epochs', type=int, default=130,
                    help='The number of training epochs')
parser.add_argument('--write_every_batch', type=int, default=50,
                    help='Write period')
parser.add_argument('--print_every', type=int, default=1,
                    help='Print period')
parser.add_argument('--save_every', type=int, default=5,
                    help='Save period')
parser.add_argument('--val_every', type=int, default=5,
                    help='Validation period')
parser.add_argument('--gt_ref', type=str2bool, default=True,
                    help='Whether to use ground truth images as references ')

### evaluate / test / fine tune setting
parser.add_argument('--continue_training', type=str2bool, default=True,
                    help='whether to load previous training model to continue training')
parser.add_argument('--eval', type=str2bool, default=False,
                    help='Evaluation mode')
parser.add_argument('--eval_save_results', type=str2bool, default=False,
                    help='Save each image during evaluation')
parser.add_argument('--model_path', type=str, default='save_result/100L30.pt',
                    help='The path of model to evaluation')
parser.add_argument('--test', type=str2bool, default=True,
                    help='Test mode')


args = parser.parse_args()
