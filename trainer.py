import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import utils.matrics as matrics
import utils.distributed as dist


# tensor (-1, 1) -> img (0, 255)
def tensor2img(tensor):
    img = (tensor + 1) * 127.5
    img = np.transpose(img.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)

    return img


def compare_img_save(rn_img, dr_baseline, gt, ref, noise, dr_gt, dr_ref, dr_noise, save_dir):
    fig, ((rn_img_fig, gt_ref_fig, ref_fig, noise_fig),
          (dr_baseline_fig, dr_gt_fig, dr_ref_fig, dr_noise_fig)) = plt.subplots(2, 4)

    rn_img_fig.imshow(tensor2img(rn_img))
    rn_img_fig.set_title('rainy image')
    rn_img_fig.axis('off')
    dr_baseline_fig.imshow(tensor2img(dr_baseline))
    dr_baseline_fig.set_title('Baseline result')
    dr_baseline_fig.axis('off')

    gt_ref_fig.imshow(tensor2img(gt))
    gt_ref_fig.set_title('clean image')
    gt_ref_fig.axis('off')
    dr_gt_fig.imshow(tensor2img(dr_gt))
    dr_gt_fig.set_title('result')
    dr_gt_fig.axis('off')

    ref_fig.imshow(tensor2img(ref))
    ref_fig.set_title('reference image')
    ref_fig.axis('off')
    dr_ref_fig.imshow(tensor2img(dr_ref))
    dr_ref_fig.set_title('result')
    dr_ref_fig.axis('off')

    noise_fig.imshow(tensor2img(noise))
    noise_fig.set_title('noise')
    noise_fig.axis('off')
    dr_noise_fig.imshow(tensor2img(dr_noise))
    dr_noise_fig.set_title('result')
    dr_noise_fig.axis('off')

    plt.savefig(save_dir)


# noinspection DuplicatedCode
def img_save(rn_img, cl_img, cl_ref, dr_img, dr, save_dir):
    fig, ((rn_img_fig, cl_ref_fig, temp_fig), (baseline_fig, pipe_fig, gt_fig)) = plt.subplots(2, 3)

    rn_img_fig.imshow(tensor2img(rn_img))
    rn_img_fig.set_title('rainy image')
    rn_img_fig.axis('off')
    cl_ref_fig.imshow(tensor2img(cl_ref))
    cl_ref_fig.set_title('reference')
    cl_ref_fig.axis('off')
    baseline_fig.imshow(tensor2img(dr_img))
    baseline_fig.set_title('baseline result')
    baseline_fig.axis('off')
    pipe_fig.imshow(tensor2img(dr))
    pipe_fig.set_title('pipeline result')
    pipe_fig.axis('off')
    gt_fig.imshow(tensor2img(cl_img))
    gt_fig.set_title('ground truth')
    gt_fig.axis('off')
    temp_fig.axis('off')

    plt.savefig(save_dir)


def get_noise_img(batch_size):
    img = torch.randint(0, 256, (batch_size, 3, 256, 256))
    img = (img / 127.5) - 1.

    return img


# def narrow_img(img, img_size):
#     output = []
#     for i in range(img.size(0)):
#         img_h = img[i].size(-2)  # (C, H, W)
#         img_start_top = img_h - img_size[i][-2]  # (H, W)
#         img_w = img[i].size(-1)
#         img_start_left = img_w - img_size[i][-1]
#         _img = img[i].narrow(1, img_start_top, img_h - img_start_top).narrow(2, img_start_left, img_w - img_start_left)
#         output.append(_img)
#     return output


### PReNet input range (0, 1), output range (0, 1)
def PReNet_derain(model, rn_img):
    img_device = rn_img.device
    model_device = next(model.parameters()).device

    rn_img = (rn_img + 1.) / 2  # (-1, 1) -> (0, 1)

    # _rn_img = torch.unsqueeze(rn_img, 0)
    rn_img = rn_img.to(model_device).detach()

    with torch.no_grad():
        dr_img, _ = model(rn_img)
    dr_img = torch.clamp(dr_img, 0., 1.)

    dr_img = dr_img.to(img_device)

    dr_img = (dr_img * 2) - 1.  # (0, 1) -> (-1, 1)

    # dr_img = torch.squeeze(dr_img)

    return dr_img.detach()


class Trainer:
    def __init__(self, args, logger, dataloader, model, loss, baseline):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.baseline = baseline
        self.loss = loss
        self.device = torch.device('cuda')

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.module.MainNet.parameters()),
             "lr": args.lr_rate
             },
            {"params": filter(lambda p: p.requires_grad, self.model.module.LTE.parameters()),
             "lr": args.lr_rate_lte
             }
        ]

        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr_improve = 0.
        self.max_psnr_epoch = 0.
        self.max_ssim_improve = 0.
        self.max_ssim_epoch = 0

    # batch sample preparation
    def prepare(self, sample_batch):
        sample_batch['cl_img'] = sample_batch['cl_img'].to(self.device)
        sample_batch['rn_img'] = sample_batch['rn_img'].to(self.device)
        sample_batch['cl_ref'] = sample_batch['cl_ref'].to(self.device)
        sample_batch['rn_ref'] = sample_batch['rn_ref'].to(self.device)
        return sample_batch
    
    
    def train(self, current_epoch=0, is_init=False):
        ### epoch preperation
        self.model.train()
        self.scheduler.step()

        ### log info
        if dist.get_rank() == 0:
            self.logger.info('Current epoch: %d' % current_epoch)
            self.logger.info('Current epoch learning rate: %e' % (self.optimizer.param_groups[1]['lr']))

        ### initialize evaluation matrics
        _psnr, _ssim, _psnr_baseline, _ssim_baseline = 0., 0., 0., 0.

        for i_batch, sample_batch in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad(set_to_none=True)

            ### prepare sample batch -> to gpu device
            sample_batch = self.prepare(sample_batch)

            cl_img = sample_batch['cl_img']
            rn_img = sample_batch['rn_img']
            cl_ref = sample_batch['cl_ref']
            rn_ref = sample_batch['rn_ref']
            
            #baseline pre-derain
            dr_img = PReNet_derain(self.baseline, rn_img).detach()
            dr_ref = PReNet_derain(self.baseline, rn_ref).detach()

            ### if ground truth initialization -> use ground truth as reference images
            if is_init:
                dr_ref = dr_img
                cl_ref = cl_img

            ### tensor range: [-1, 1]
            dr, S, T_lv3, T_lv2, T_lv1 = self.model(dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

            ### calc loss
            if is_init:
                loss = self.loss.init_loss(dr, cl_img)
            else:
                loss = self.loss.loss(dr, cl_img)
                
            dist.barrier()

            loss.backward()
            self.optimizer.step()

            ### print flag            
            is_print = ((i_batch + 1) % self.args.print_every_batch == 0)  ### flag of print
            if (dist.get_rank() == 0) and (is_print):
                self.logger.info('epoch: ' + str(current_epoch) +
                                 '\t batch: ' + str(i_batch + 1))
                if not is_init:
                    self.logger.info('ms_ssim_l1_loss: %.10f' % self.loss.loss_dict.get('ms_ssim_l1_loss'))
                else:
                    self.logger.info('rec_loss: %.10f' % self.loss.loss_dict.get('rec_loss'))
                if (self.loss.loss_dict.get('psnr_loss')):
                    self.logger.info('psnr_loss: %.10f' % self.loss.loss_dict.get('psnr_loss'))
                if (self.loss.loss_dict.get('ssim_loss')):
                    self.logger.info('ssim_loss: %.10f' % self.loss.loss_dict.get('ssim_loss'))


            ### mark down epoch matrics (PSNR, SSIM)
            _psnr, _ssim = matrics.matrics_update(_psnr, _ssim, i_batch+1, dr.detach(), cl_img.detach())
            _psnr_baseline, _ssim_baseline = matrics.matrics_update(_psnr_baseline, _ssim_baseline, i_batch+1, dr_img.detach(), cl_img.detach())
        

        ### print epoch matrics
        if dist.get_rank() == 0:
            self.logger.info('epoch: ' + str(current_epoch) + '\t DRTT PSNR' + str(_psnr) + '\t DRTT SSIM' + str(_ssim))
            self.logger.info('epoch: ' + str(current_epoch) + '\t Baseline PSNR' + str(_psnr_baseline) + '\t Baseline SSIM' + str(_ssim_baseline))

        ### save model
        if (current_epoch % self.args.save_every_epoch == 0) and (dist.get_rank() == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key: tmp[key] for key in tmp if
                                (('SearchNet' not in key) and ('_copy' not in key))}
            model_dir = os.path.join(self.args.save_dir, 'model')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model_state_dict, os.path.join(model_dir, 'model_' + str(current_epoch).zfill(5) + '.pt'))
        


    def evaluate(self, current_epoch=0):
        if dist.get_rank() == 0:
            self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        self.model.eval()
        with torch.no_grad():
            _psnr, _ssim, _psnr_baseline, _ssim_baseline = 0., 0., 0., 0.
            for i_batch, sample_batch in enumerate(self.dataloader['val']):
                sample_batch = self.prepare(sample_batch)
                cl_img = sample_batch['cl_img']
                rn_img = sample_batch['rn_img']
                cl_ref = sample_batch['cl_ref']
                rn_ref = sample_batch['rn_ref']
                
                # baseline deraining
                dr_img = PReNet_derain(self.baseline, rn_img)
                dr_ref = PReNet_derain(self.baseline, rn_ref)

                dr, _, _, _, _ = self.model(dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

                ### mark down epoch matrics (PSNR, SSIM)
                _psnr, _ssim = matrics.matrics_update(_psnr, _ssim, i_batch+1, dr.detach(), cl_img.detach())
                _psnr_baseline, _ssim_baseline = matrics.matrics_update(_psnr_baseline, _ssim_baseline, i_batch+1, dr_img.detach(), cl_img.detach())

                if self.args.eval_save_results:
                    result_dir = os.path.join(self.args.save_dir, 'results', 'evaluation_result')
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    if dist.get_rank() == 0:
                        for i in range(len(cl_img)):
                            img_save(rn_img=rn_img[i], cl_img=cl_img[i], cl_ref=cl_ref[i], dr_img=dr_img[i], dr=dr[i],
                                     save_dir=os.path.join(result_dir,str(i_batch * self.args.batch_size + i).zfill(5) + '.png'))

            if dist.get_rank() == 0:
                self.logger.info('Pipeline  PSNR (now): %.3f \t SSIM (now): %.4f' % (_psnr, _ssim))
                self.logger.info('baseline  PSNR (now): %.3f \t SSIM (now): %.4f' % (_psnr_baseline, _ssim_baseline))


            if (_psnr - _psnr_baseline) > self.max_psnr_improve:
                self.max_psnr_improve = _psnr - _psnr_baseline
                self.max_psnr_epoch = current_epoch
            if (_ssim - _ssim_baseline) > self.max_ssim_improve:
                self.max_ssim_improve = _ssim - _ssim_baseline
                self.max_ssim_epoch = current_epoch
            if dist.get_rank() == 0:
                self.logger.info('Ref  PSNR improvement (max): %.3f (%d) \t SSIM improvement (max): %.4f (%d)'
                                 % (self.max_psnr_improve, self.max_psnr_epoch, self.max_ssim_improve,
                                    self.max_ssim_epoch))
        if dist.get_rank() == 0:
            self.logger.info('Evaluation over.')
        
        dist.barrier()

    def test(self):
        ### test begin
        self.logger.info('Test process...')

        self.model.eval()

        _psnr, _ssim, _psnr_baseline, _ssim_baseline = 0., 0., 0., 0.

        with torch.no_grad():
            for i_batch, sample_batch in enumerate(self.dataloader['test']):
                sample_batch = self.prepare(sample_batch)
                cl_img = sample_batch['cl_img']
                rn_img = sample_batch['rn_img']
                cl_ref = sample_batch['cl_ref']
                rn_ref = sample_batch['rn_ref']
                
                # baseline deraining
                dr_img = PReNet_derain(self.baseline, rn_img)
                dr_ref = PReNet_derain(self.baseline, rn_ref)

                dr, _, _, _, _ = self.model(dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

                ### mark down epoch matrics (PSNR, SSIM)
                _psnr, _ssim = matrics.matrics_update(_psnr, _ssim, i_batch+1, dr.detach(), cl_img.detach())
                _psnr_baseline, _ssim_baseline = matrics.matrics_update(_psnr_baseline, _ssim_baseline, i_batch+1, dr.detach(), cl_img.detach())

                if dist.get_rank() == 0:
                    result_dir = os.path.join(self.args.save_dir, 'results', 'test_result')

                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    for i in range(len(cl_img)):
                        # img_save(rn_img=rn_img[i], cl_img=cl_img[i], cl_ref=cl_ref[i], dr_img=dr_img[i], dr=dr[i],
                        #          save_dir=os.path.join(result_dir,str(i_batch * self.args.batch_size + i).zfill(5) + '.png'))
                        plt.imsave(os.path.join(result_dir, 'output' + str(i_batch * self.args.batch_size + i).zfill(5) + '.png'),
                                tensor2img(dr[i]))
                        plt.imsave(os.path.join(result_dir, 'baseline' + str(i_batch * self.args.batch_size + i).zfill(5) + '.png'),
                                tensor2img(dr_img[i]))
                        plt.imsave(os.path.join(result_dir, 'clean' + str(i_batch * self.args.batch_size + i).zfill(5) + '.png'),
                                tensor2img(cl_img[i]))
                        plt.imsave(os.path.join(result_dir, 'rainy' + str(i_batch * self.args.batch_size + i).zfill(5) + '.png'),
                                tensor2img(rn_img[i]))
                        plt.imsave(os.path.join(result_dir, 'ref' + str(i_batch * self.args.batch_size + i).zfill(5) + '.png'),
                                tensor2img(cl_ref[i]))


        self.logger.info('baseline Test Stage PSNR: %.3f \t SSIM: %.4f' % (_psnr_baseline, _ssim_baseline))
        self.logger.info('Pipeline Test Stage PSNR: %.3f \t SSIM: %.4f' % (_psnr, _ssim))
        self.logger.info('Test Stage PSNR improvement: %.4f, \t %.2f on the baseline model' % (
            _psnr - _psnr_baseline, (_psnr - _psnr_baseline) / _psnr_baseline * 100))
        self.logger.info('Test Stage SSIM improvement: %.4f, \t %.2f on the baseline model' % (
            _ssim - _ssim_baseline, (_ssim - _ssim_baseline) / _ssim_baseline * 100))
        self.logger.info('output path: %s' % result_dir)
        self.logger.info('Test Over.')

    def compare_test(self):

        self.logger.info('Test process...')

        self.model.eval()
        _psnr_gt, _ssim_gt = 0., 0.  # ground truth matrix
        _psnr_ref, _ssim_ref = 0., 0.  # reference matrix
        _psnr_noise, _ssim_noise = 0., 0.  # noise matrix
        _psnr_baseline, _ssim_baseline = 0., 0.  # baseline matrix

        with torch.no_grad():
            for i_batch, sample_batch in enumerate(self.dataloader['test']):
                sample_batch = self.prepare(sample_batch)
                cl_img = sample_batch['cl_img']
                rn_img = sample_batch['rn_img']
                cl_ref = sample_batch['cl_ref']
                rn_ref = sample_batch['rn_ref']
                
                # baseline deraining
                dr_img = PReNet_derain(self.baseline, rn_img)
                dr_ref = PReNet_derain(self.baseline, rn_ref)
                
                noise = get_noise_img(cl_img.size(0)).to(dr_img.device)

                ### ground truth reference
                dr_gt, S_gt, T_gt_lv3, T_gt_lv2, T_gt_lv1 = self.model(dr_img=dr_img, cl_ref=cl_img, dr_ref=dr_img)

                ### reference
                dr_ref, S_ref, T_ref_lv3, T_ref_lv2, T_ref_lv1 = self.model(dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

                ### noise reference
                dr_noise, S_n, T_n_lv3, T_n_lv2, T_n_lv1 = self.model(dr_img=dr_img, cl_ref=noise, dr_ref=noise)

                if dist.get_rank() == 0:
                    result_dir = os.path.join(self.args.save_dir, 'results', 'test result')
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                        
                    for i in range(cl_img.size()[0]):
                        compare_img_save(rn_img=rn_img[i], dr_baseline=dr_img[i], gt=cl_img[i], ref=cl_ref[i],
                                        noise=noise[i], dr_gt=dr_gt[i], dr_ref=dr_ref[i],
                                        dr_noise=dr_noise[i],
                                        save_dir=os.path.join(result_dir,
                                                            str(i_batch * self.args.batch_size + i).zfill(5) + '.png'))

                # update matrics
                _psnr_gt, _ssim_gt=  matrics.matrics_update(_psnr_gt, _ssim_gt, i_batch+1, dr_gt.detach(), cl_img.detach())
                _psnr_ref, _ssim_ref = matrics.matrics_update(_psnr_ref, _ssim_ref, i_batch+1, dr_ref.detach(), cl_img.detach())
                _psnr_noise, _ssim_noise = matrics.matrics_update(_psnr_noise, _ssim_noise, i_batch+1, dr_noise.detach(), cl_img.detach())
                _psnr_baseline, _ssim_baseline = matrics.matrics_update(_psnr_baseline, _ssim_baseline, i_batch+1, dr_img.detach(), cl_img.detach())

        self.logger.info('baseline Test Stage PSNR: %.3f \t SSIM: %.4f' % (_psnr_baseline, _ssim_baseline))

        self.logger.info('Clean Image Referencing PSNR: %.3f \t SSIM: %.4f' % (_psnr_gt, _ssim_gt))
        self.logger.info('Reference Image Referencing PSNR: %.3f \t SSIM: %.4f' % (_psnr_ref, _ssim_ref))
        self.logger.info('Noise Referencing PSNR: %.3f \t SSIM: %.4f' % (_psnr_noise, _ssim_noise))

        self.logger.info('Clean Image Referencing PSNR improvement: %.4f, \t %.2f%% on the baseline model' % (
            _psnr_gt - _psnr_baseline, (_psnr_gt - _psnr_baseline) / _psnr_baseline * 100))
        self.logger.info('Clean Image Referencing SSIM improvement: %.4f, \t %.2f%% on the baseline model' % (
            _ssim_gt - _ssim_baseline, (_ssim_gt - _ssim_baseline) / _ssim_baseline * 100))

        self.logger.info('Reference Image Referencing PSNR improvement: %.4f, \t %.2f%% on the baseline model' % (
            _psnr_ref - _psnr_baseline, (_psnr_ref - _psnr_baseline) / _psnr_baseline * 100))
        self.logger.info('Reference Image Referencing SSIM improvement: %.4f, \t %.2f%% on the baseline model' % (
            _ssim_ref - _ssim_baseline, (_ssim_ref - _ssim_baseline) / _ssim_baseline * 100))

        self.logger.info('Noise Image Referencing PSNR improvement: %.4f, \t %.2f%% on the baseline model' % (
            _psnr_noise - _psnr_baseline, (_psnr_noise - _psnr_baseline) / _psnr_baseline * 100))
        self.logger.info('Noise Image Referencing SSIM improvement: %.4f, \t %.2f%% on the baseline model' % (
            _ssim_noise - _ssim_baseline, (_ssim_noise - _ssim_baseline) / _ssim_baseline * 100))

        self.logger.info('output path: %s' % (result_dir))
        self.logger.info('Test Over.')



