import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

import utils.matrics as matrics
import utils.distributed as dist

# Utils Funcs
# tensor (-1, 1) -> img (0, 255)


def tensor2img(tensor):
    img = (tensor + 1) * 127.5
    img = np.transpose(img.squeeze().round().cpu().numpy(),
                       (1, 2, 0)).astype(np.uint8)

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
    fig, ((rn_img_fig, cl_ref_fig, temp_fig),
          (baseline_fig, pipe_fig, gt_fig)) = plt.subplots(2, 3)

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


def get_noise_img(size):
    img = torch.randint(0, 256, size)
    img = (img / 127.5) - 1.

    return img


# (B, C, 256, 1024) -> (4B, C, 256, 256)
def split_img(img):
    split_img = torch.split(img, 256, dim=-1)
    split_img = torch.cat(split_img, dim=0)
    return split_img


# (4B, C, 256, 256) -> (B, C, 256, 1024)
def cat_img(img):
    img = torch.split(img, int(img.size(0)/4), dim=0)
    img = torch.cat(img, dim=-1)
    return img


# PReNet input range (0, 1), output range (0, 1)
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


# Uformer deraining
def Uformer_derain(model, rn_img, resize_data=False):
    img_device = rn_img.device
    model_device = next(model.parameters()).device

    rn_img = rn_img.to(model_device).detach()
    if resize_data:
        rn_img = split_img(rn_img)

    with torch.no_grad():
        dr_img = model(rn_img)
    if resize_data:
        dr_img = cat_img(dr_img)

    dr_img = dr_img.to(img_device)

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
            {"params": filter(lambda p: p.requires_grad, self.model.module.FF.parameters()),
             "lr": args.lr_rate
             },
            {"params": filter(lambda p: p.requires_grad, self.model.module.FA.parameters()),
             "lr": args.lr_rate_lte
             }
        ]

        self.optimizer = optim.Adam(self.params, betas=(
            args.beta1, args.beta2), eps=args.eps)
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
        # epoch preperation
        self.model.train()
        self.scheduler.step()

        # log info
        if dist.get_rank() == 0:
            self.logger.info('Current epoch: %d' % current_epoch)

        # initialize evaluation matrics
        _psnr, _ssim, _psnr_baseline, _ssim_baseline = 0., 0., 0., 0.

        for i_batch, sample_batch in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad(set_to_none=True)

            # prepare sample batch -> to gpu device
            sample_batch = self.prepare(sample_batch)

            cl_img = sample_batch['cl_img']
            rn_img = sample_batch['rn_img']
            cl_ref = sample_batch['cl_ref']
            rn_ref = sample_batch['rn_ref']

            # baseline pre-derain
            if self.args.baseline == 'PReNet':
                dr_img = PReNet_derain(self.baseline, rn_img)
                dr_ref = PReNet_derain(self.baseline, rn_ref)
            elif self.args.baseline == 'GMM':
                dr_img = sample_batch['gmm_img']
                dr_ref = sample_batch['gmm_ref']
            elif self.args.baseline == 'Uformer':
                dr_img = Uformer_derain(self.baseline, rn_img, resize_data=(
                    self.args.dataset == 'KITTI' or self.args.dataset == 'Cityscapes'))
                dr_ref = Uformer_derain(self.baseline, rn_ref, resize_data=(
                    self.args.dataset == 'KITTI' or self.args.dataset == 'Cityscapes'))

            # if ground truth initialization -> use ground truth as reference images
            if is_init:
                dr_ref = dr_img
                cl_ref = cl_img

            # tensor range: [-1, 1]
            dr, _, _, _, _ = self.model(dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

            # calc loss
            if is_init:
                loss = self.loss.init_loss(dr, cl_img)
            else:
                loss = self.loss.loss(dr, cl_img)

            dist.barrier()

            # back prop
            loss.backward()
            self.optimizer.step()

            # print flag
            is_print = ((i_batch + 1) % self.args.print_every_batch == 0)
            if (dist.get_rank() == 0) and (is_print):
                self.logger.info(f'Epoch: {current_epoch}, Batch: {i_batch + 1}')
                if not is_init:
                    self.logger.info(f'ms_ssim_l1_loss: {(self.loss.loss_dict.get("ms_ssim_l1_loss")):.10f}')
                else:
                    self.logger.info(f'rec_loss: {(self.loss.loss_dict.get("rec_loss")):.10f}')
                if self.loss.loss_dict.get('psnr_loss'):
                    self.logger.info(f'psnr_loss: {(self.loss.loss_dict.get("psnr_loss")):.10f}')
                if self.loss.loss_dict.get('ssim_loss'):
                    self.logger.info(f'ssim_loss: {(self.loss.loss_dict.get("ssim_loss")):.10f}')

            # mark down epoch matrics (PSNR, SSIM)
            _psnr, _ssim = matrics.matrics_update(_psnr, _ssim, i_batch+1, dr.detach(), cl_img.detach())
            _psnr_baseline, _ssim_baseline = matrics.matrics_update(_psnr_baseline, _ssim_baseline, i_batch+1, dr_img.detach(), cl_img.detach())


        # for each epoch
        # print epoch matrics
        if dist.get_rank() == 0:
            self.logger.info(f'Epoch: {current_epoch}')
            self.logger.info(f'RefGT PSNR: {_psnr} | RefGT SSIM: {_ssim}')
            self.logger.info(f'Baseline PSNR: {_psnr_baseline} | Baseline SSIM: {_ssim_baseline}')

        # save model
        if (current_epoch % self.args.save_every_epoch == 0) and (dist.get_rank() == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key: tmp[key] for key in tmp}
            model_dir = os.path.join(self.args.save_dir, 'model')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model_state_dict, os.path.join(
                model_dir, 'model_' + str(current_epoch).zfill(5) + '.pt'))


    def evaluate(self, current_epoch=0):
        if dist.get_rank() == 0:
            self.logger.info('Epoch ' + str(current_epoch) +
                             ' evaluation process...')

        self.model.eval()
        with torch.no_grad():
            _psnr, _ssim, _psnr_baseline, _ssim_baseline = 0., 0., 0., 0.
            for i_batch, sample_batch in enumerate(self.dataloader['val']):
                sample_batch = self.prepare(sample_batch)
                cl_img = sample_batch['cl_img']
                rn_img = sample_batch['rn_img']
                cl_ref = sample_batch['cl_ref']
                rn_ref = sample_batch['rn_ref']

                # baseline pre-derain
                if self.args.baseline == 'PReNet':
                    dr_img = PReNet_derain(self.baseline, rn_img)
                    dr_ref = PReNet_derain(self.baseline, rn_ref)
                elif self.args.baseline == 'GMM':
                    dr_img = sample_batch['gmm_img']
                    dr_ref = sample_batch['gmm_ref']
                elif self.args.baseline == 'Uformer':
                    dr_img = Uformer_derain(self.baseline, rn_img, resize_data=(
                        self.args.dataset == 'KITTI' or self.args.dataset == 'Cityscapes'))
                    dr_ref = Uformer_derain(self.baseline, rn_ref, resize_data=(
                        self.args.dataset == 'KITTI' or self.args.dataset == 'Cityscapes'))

                dr, _, _, _, _ = self.model(
                    dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

                # mark down epoch matrics (PSNR, SSIM)
                _psnr, _ssim = matrics.matrics_update(
                    _psnr, _ssim, i_batch+1, dr.detach(), cl_img.detach())
                _psnr_baseline, _ssim_baseline = matrics.matrics_update(
                    _psnr_baseline, _ssim_baseline, i_batch+1, dr_img.detach(), cl_img.detach())

                if self.args.eval_save_results:
                    result_dir = os.path.join(
                        self.args.save_dir, 'results', 'evaluation_result')
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    if dist.get_rank() == 0:
                        for i in range(len(cl_img)):
                            img_save(rn_img=rn_img[i], cl_img=cl_img[i], cl_ref=cl_ref[i], dr_img=dr_img[i], dr=dr[i],
                                     save_dir=os.path.join(result_dir, str(i_batch * self.args.batch_size + i).zfill(5) + '.png'))

            if dist.get_rank() == 0:
                self.logger.info(
                    'Pipeline  PSNR (now): %.3f \t SSIM (now): %.4f' % (_psnr, _ssim))
                self.logger.info('baseline  PSNR (now): %.3f \t SSIM (now): %.4f' % (
                    _psnr_baseline, _ssim_baseline))

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
        # test begin
        self.logger.info('Test process...')

        self.model.eval()

        _psnr, _ssim, _psnr_baseline, _ssim_baseline = 0., 0., 0., 0.
        result_dir = ''

        with torch.no_grad():
            for i_batch, sample_batch in enumerate(self.dataloader['test']):
                # sampling
                sample_batch = self.prepare(sample_batch)
                cl_img = sample_batch['cl_img']
                rn_img = sample_batch['rn_img']
                cl_ref = sample_batch['cl_ref']
                rn_ref = sample_batch['rn_ref']

                # baseline pre-derain
                if self.args.baseline == 'PReNet':
                    dr_img = PReNet_derain(self.baseline, rn_img)
                    dr_ref = PReNet_derain(self.baseline, rn_ref)
                elif self.args.baseline == 'GMM':
                    dr_img = sample_batch['gmm_img']
                    dr_ref = sample_batch['gmm_ref']
                elif self.args.baseline == 'Uformer':
                    dr_img = Uformer_derain(self.baseline, rn_img, resize_data=(
                        self.args.dataset == 'KITTI' or self.args.dataset == 'Cityscapes'))
                    dr_ref = Uformer_derain(self.baseline, rn_ref, resize_data=(
                        self.args.dataset == 'KITTI' or self.args.dataset == 'Cityscapes'))

                # RefGT derain
                dr, _, _, _, _ = self.model(dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

                # mark down epoch matrics (PSNR, SSIM)
                _psnr, _ssim = matrics.matrics_update(
                    _psnr, _ssim, i_batch+1, dr.detach(), cl_img.detach())
                _psnr_baseline, _ssim_baseline = matrics.matrics_update(
                    _psnr_baseline, _ssim_baseline, i_batch+1, dr_img.detach(), cl_img.detach())

                if dist.get_rank() == 0:
                    result_dir = os.path.join(
                        self.args.save_dir, 'results', 'test_result')

                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    for i in range(len(cl_img)):
                        plt.imsave(os.path.join(result_dir, str(i_batch * self.args.batch_size + i).zfill(5) + 'output' + '.png'), 
                                   tensor2img(dr[i]))
                        plt.imsave(os.path.join(result_dir, str(i_batch * self.args.batch_size + i).zfill(5) + 'baseline' + '.png'),
                                   tensor2img(dr_img[i]))
                        plt.imsave(os.path.join(result_dir, str(i_batch * self.args.batch_size + i).zfill(5) + 'clean' + '.png'),
                                   tensor2img(cl_img[i]))
                        plt.imsave(os.path.join(result_dir, str(i_batch * self.args.batch_size + i).zfill(5) + 'rainy' + '.png'),
                                   tensor2img(rn_img[i]))
                        plt.imsave(os.path.join(result_dir, str(i_batch * self.args.batch_size + i).zfill(5) + 'ref' + '.png'),
                                   tensor2img(cl_ref[i]))


        self.logger.info(f'Baseline Test Stage PSNR: {_psnr_baseline:.3f} \t SSIM: {_ssim_baseline:.4f}')
        self.logger.info(f'Pipeline Test Stage PSNR: {_psnr:.3f} \t SSIM: {_ssim:.4f}')
        self.logger.info(f'PSNR improvement: {(_psnr - _psnr_baseline):.3f}, \t {((_psnr - _psnr_baseline) / _psnr_baseline * 100):2f}% on the baseline model')
        self.logger.info(f'SSIM improvement: {(_ssim - _ssim_baseline):.4f}, \t {((_ssim - _ssim_baseline) / _ssim_baseline * 100):2f} on the baseline model')
        self.logger.info(f'output path: {result_dir}')
        self.logger.info('Test Over.')
