import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from utils import calc_psnr_and_ssim


# tensor (-1, 1) -> img (0, 255)
def tensor2img(tensor):
    img = (tensor + 1) * 127.5
    img = np.transpose(img.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)

    return img


def compare_img_save(rn_img, dr_bb, gt, ref, noise, dr_gt, dr_ref, dr_noise, save_dir):
    fig, ((rn_img_fig, gt_ref_fig, ref_fig, noise_fig),
          (dr_bb_fig, dr_gt_fig, dr_ref_fig, dr_noise_fig)) = plt.subplots(2, 4)

    rn_img_fig.imshow(tensor2img(rn_img))
    rn_img_fig.set_title('rainy image')
    rn_img_fig.axis('off')
    dr_bb_fig.imshow(tensor2img(dr_bb))
    dr_bb_fig.set_title('Baseline result')
    dr_bb_fig.axis('off')

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
    fig, ((rn_img_fig, cl_ref_fig, temp_fig), (bb_fig, pipe_fig, gt_fig)) = plt.subplots(2, 3)

    rn_img_fig.imshow(tensor2img(rn_img))
    rn_img_fig.set_title('rainy image')
    rn_img_fig.axis('off')
    cl_ref_fig.imshow(tensor2img(cl_ref))
    cl_ref_fig.set_title('reference')
    cl_ref_fig.axis('off')
    bb_fig.imshow(tensor2img(dr_img))
    bb_fig.set_title('baseline result')
    bb_fig.axis('off')
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


def narrow_img(img, img_size):
    output = []
    for i in range(img.size(0)):
        img_h = img[i].size(-2)  # (C, H, W)
        img_start_top = img_h - img_size[i][-2]  # (H, W)
        img_w = img[i].size(-1)
        img_start_left = img_w - img_size[i][-1]
        _img = img[i].narrow(1, img_start_top, img_h - img_start_top).narrow(2, img_start_left, img_w - img_start_left)
        output.append(_img)
    return output


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
    def __init__(self, args, logger, dataloader, model, loss, bb_model):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.bb_model = bb_model
        self.loss = loss
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.writer = SummaryWriter(os.path.join(args.save_dir, 'summary'))

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters() if
            args.num_gpu == 1 else self.model.module.MainNet.parameters()),
             "lr": args.lr_rate
             },
            {"params": filter(lambda p: p.requires_grad, self.model.LTE.parameters() if
            args.num_gpu == 1 else self.model.module.LTE.parameters()),
             "lr": args.lr_rate_lte
             }
        ]

        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr_improve = 0.
        self.max_psnr_epoch = 0
        self.max_ssim_improve = 0.
        self.max_ssim_epoch = 0

    def prepare(self, sample_batched):
        sample_batched['cl_img'] = sample_batched['cl_img'].to(self.device)
        sample_batched['rn_img'] = sample_batched['rn_img'].to(self.device)
        sample_batched['cl_ref'] = sample_batched['cl_ref'].to(self.device)
        sample_batched['rn_ref'] = sample_batched['rn_ref'].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, gt_ref=False):
        self.model.train()

        self.scheduler.step()

        ### log info
        self.logger.info('Current epoch: %d' % current_epoch)
        self.logger.info('Current epoch learning rate: %e' % (self.optimizer.param_groups[1]['lr']))

        ### initialize evaluation matrix
        _psnr_sum, _psnr_bb_sum, _ssim_sum, _ssim_bb_sum, writer_cnt = 0., 0., 0., 0., 0

        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            writer_cnt += 1

            self.optimizer.zero_grad()

            ### prepare sample batch -> to cpu device
            sample_batched = self.prepare(sample_batched)

            cl_img = sample_batched['cl_img']
            # dr_img = sample_batched['dr_img']
            rn_img = sample_batched['rn_img']
            cl_ref = sample_batched['cl_ref']
            # dr_ref = sample_batched['dr_ref']
            rn_ref = sample_batched['rn_ref']
            dr_img = PReNet_derain(self.bb_model, rn_img)
            dr_ref = PReNet_derain(self.bb_model, rn_ref)

            ### if ground truth initialization -> use ground truth as reference images
            if gt_ref:
                dr_ref = dr_img
                cl_ref = cl_img

            ### tensor range: [-1, 1]
            dr, S, T_lv3, T_lv2, T_lv1 = self.model(dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

            ### calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0)  ### flag of print
            ### basic loss: MS-SSIM-L1 loss or reconstruction loss
            if self.args.ms_ssim_l1_loss:
                ms_ssim_l1_loss = self.loss['ms_ssim_l1_loss'](dr, cl_img)
                ms_ssim_l1_loss_data = ms_ssim_l1_loss.item()
                loss = ms_ssim_l1_loss
            else:
                rec_loss = self.loss['rec_loss'](dr, cl_img)
                rec_loss_data = rec_loss.item()
                loss = rec_loss
            ### option loss: PSNR loss or SSIM loss
            if self.args.psnr_loss:
                psnr_loss = self.loss['psnr_loss'](dr, cl_img)
                psnr_loss_data = psnr_loss.item()
                loss += psnr_loss
            if self.args.ssim_loss:
                ssim_loss = self.loss['ssim_loss'](dr, cl_img)
                ssim_loss_data = ssim_loss.item()
                loss += ssim_loss
            if is_print:
                self.logger.info('epoch: ' + str(current_epoch) +
                                 '\t batch: ' + str(i_batch + 1))
                if self.args.ms_ssim_l1_loss:
                    self.logger.info('ms_ssim_l1_loss: %.10f' % ms_ssim_l1_loss_data)
                else:
                    self.logger.info('rec_loss: %.10f' % rec_loss_data)
                if self.args.psnr_loss:
                    self.logger.info('psnr_loss: %.10f' % psnr_loss_data)
                if self.args.ssim_loss:
                    self.logger.info('ssim_loss: %.10f' % ssim_loss_data)

            loss.backward()
            self.optimizer.step()

            ### mark down indicator (PSNR SSIM)
            _psnr, _ssim = calc_psnr_and_ssim(dr.detach(), cl_img.detach())
            _psnr_bb, _ssim_bb = calc_psnr_and_ssim(dr_img.detach(), cl_img.detach())
            _psnr_sum += _psnr
            _psnr_bb_sum += _psnr_bb
            _ssim_sum += _ssim
            _ssim_bb_sum += _ssim_bb

            ### tensorboard writer
            if writer_cnt % self.args.write_every_batch == 0:
                self.writer.add_scalars('Training PSNR', {'PSNR of Pipeline': _psnr_sum / self.args.write_every_batch,
                                                          'PSNR of Baseline': _psnr_bb_sum / self.args.write_every_batch},
                                        global_step=writer_cnt * current_epoch)
                self.writer.add_scalars('Training SSIM', {'SSIM of Pipeline': _ssim_sum / self.args.write_every_batch,
                                                          'SSIM of Baseline': _ssim_bb_sum / self.args.write_every_batch},
                                        global_step=writer_cnt * current_epoch)
                self.logger.info('psnr difference (pipeline - backbone)' + str(
                    (_psnr_sum - _psnr_bb_sum) / self.args.write_every_batch))
                self.logger.info('ssim difference (pipeline - backbone)' + str(
                    (_ssim_sum - _ssim_bb_sum) / self.args.write_every_batch))
                _psnr_sum, _psnr_bb_sum, _ssim_sum, _ssim_bb_sum = 0., 0., 0., 0.  # reset matrix

        ### save model
        if current_epoch % self.args.save_every == 0:
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.', ''): tmp[key] for key in tmp if
                                (('SearchNet' not in key) and ('_copy' not in key))}
            model_dir = os.path.join(self.args.save_dir, 'model')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model_state_dict, os.path.join(model_dir, 'model_' + str(current_epoch).zfill(5) + '.pt'))

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        self.model.eval()
        with torch.no_grad():
            psnr, ssim, psnr_bb, ssim_bb = 0., 0., 0., 0.
            for i_batch, sample_batched in enumerate(self.dataloader['val']):
                sample_batched = self.prepare(sample_batched)
                cl_img = sample_batched['cl_img']
                rn_img = sample_batched['rn_img']
                # dr_img = sample_batched['dr_img']
                cl_ref = sample_batched['cl_ref']
                rn_ref = sample_batched['rn_ref']
                # dr_ref = sample_batched['dr_ref']
                dr_img = PReNet_derain(self.bb_model, rn_img)
                dr_ref = PReNet_derain(self.bb_model, rn_ref)

                dr, _, _, _, _ = self.model(dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

                ### calculate psnr and ssim
                _psnr, _ssim = calc_psnr_and_ssim(dr.detach(), cl_img.detach())
                _psnr_bb, _ssim_bb = calc_psnr_and_ssim(dr_img.detach(), cl_img.detach())
                psnr += _psnr
                ssim += _ssim
                psnr_bb += _psnr_bb
                ssim_bb += _ssim_bb

                if self.args.eval_save_results:
                    result_dir = os.path.join(self.args.save_dir, 'results', 'evaluation_result')
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    img_sizes = sample_batched['cl_img_sizes']
                    ref_sizes = sample_batched['cl_ref_sizes']
                    cl_img = narrow_img(cl_img, img_sizes)
                    rn_img = narrow_img(rn_img, img_sizes)
                    dr_img = narrow_img(dr_img, img_sizes)
                    dr = narrow_img(dr, img_sizes)
                    rn_ref = narrow_img(rn_ref, ref_sizes)
                    cl_ref = narrow_img(cl_ref, ref_sizes)

                    for i in range(len(cl_img)):
                        img_save(rn_img=rn_img[i], cl_img=cl_img[i], cl_ref=cl_ref[i], dr_img=dr_img[i], dr=dr[i],
                                 save_dir=os.path.join(result_dir,str(i_batch * self.args.batch_size + i).zfill(5) + '.png'))

            psnr_ave = psnr / len(self.dataloader['val'])
            ssim_ave = ssim / len(self.dataloader['val'])
            psnr_bb_ave = psnr_bb / len(self.dataloader['val'])
            ssim_bb_ave = ssim_bb / len(self.dataloader['val'])

            self.logger.info('Pipeline  PSNR (now): %.3f \t SSIM (now): %.4f' % (psnr_ave, ssim_ave))
            self.logger.info('Backbone  PSNR (now): %.3f \t SSIM (now): %.4f' % (psnr_bb_ave, ssim_bb_ave))

            self.writer.add_scalars('Evaluation PSNR',
                                    {'PSNR of Pipeline': psnr_ave, 'PSNR of Baseline': psnr_bb_ave},
                                    global_step=current_epoch)
            self.writer.add_scalars('Evaluation SSIM',
                                    {'SSIM of Pipeline': ssim_ave, 'SSIM of Baseline': ssim_bb_ave},
                                    global_step=current_epoch)

            if (psnr_ave - psnr_bb_ave) > self.max_psnr_improve:
                self.max_psnr_improve = psnr_ave - psnr_bb_ave
                self.max_psnr_epoch = current_epoch
            if (ssim_ave - ssim_bb_ave) > self.max_ssim_improve:
                self.max_ssim_improve = ssim_ave - ssim_bb_ave
                self.max_ssim_epoch = current_epoch
            self.logger.info('Ref  PSNR improvement (max): %.3f (%d) \t SSIM improvement (max): %.4f (%d)'
                             % (self.max_psnr_improve, self.max_psnr_epoch, self.max_ssim_improve,
                                self.max_ssim_epoch))

        self.logger.info('Evaluation over.')

    def test(self):
        ### test begin
        self.logger.info('Test process...')

        self.model.eval()

        psnr, ssim, psnr_bb, ssim_bb = 0., 0., 0., 0.

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.dataloader['test']):
                sample_batched = self.prepare(sample_batched)
                cl_img = sample_batched['cl_img']
                rn_img = sample_batched['rn_img']
                # dr_img = sample_batched['dr_img']
                cl_ref = sample_batched['cl_ref']
                rn_ref = sample_batched['rn_ref']
                # dr_ref = sample_batched['dr_ref']
                dr_img = PReNet_derain(self.bb_model, rn_img)
                dr_ref = PReNet_derain(self.bb_model, rn_ref)

                dr, _, _, _, _ = self.model(dr_img=dr_img, cl_ref=cl_ref, dr_ref=dr_ref)

                _psnr, _ssim = calc_psnr_and_ssim(dr.detach(), cl_img.detach())
                _psnr_bb, _ssim_bb = calc_psnr_and_ssim(dr_img.detach(), cl_img.detach())

                psnr += _psnr
                ssim += _ssim
                psnr_bb += _psnr_bb
                ssim_bb += _ssim_bb

                result_dir = os.path.join(self.args.save_dir, 'results', 'test_result')

                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                img_sizes = sample_batched['cl_img_sizes']
                ref_sizes = sample_batched['cl_ref_sizes']
                cl_img = narrow_img(cl_img, img_sizes)
                rn_img = narrow_img(rn_img, img_sizes)
                dr_img = narrow_img(dr_img, img_sizes)
                dr = narrow_img(dr, img_sizes)
                rn_ref = narrow_img(rn_ref, ref_sizes)
                cl_ref = narrow_img(cl_ref, ref_sizes)

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


        psnr = psnr / len(self.dataloader['test'])
        ssim = ssim / len(self.dataloader['test'])
        psnr_bb = psnr_bb / len(self.dataloader['test'])
        ssim_bb = ssim_bb / len(self.dataloader['test'])

        self.logger.info('Backbone Test Stage PSNR: %.3f \t SSIM: %.4f' % (psnr_bb, ssim_bb))
        self.logger.info('Pipeline Test Stage PSNR: %.3f \t SSIM: %.4f' % (psnr, ssim))
        self.logger.info('Test Stage PSNR improvement: %.4f, \t %.2f on the backbone model' % (
            psnr - psnr_bb, (psnr - psnr_bb) / psnr_bb * 100))
        self.logger.info('Test Stage SSIM improvement: %.4f, \t %.2f on the backbone model' % (
            ssim - ssim_bb, (ssim - ssim_bb) / ssim_bb * 100))
        self.logger.info('output path: %s' % result_dir)
        self.logger.info('Test Over.')

    def compare_test(self):

        self.logger.info('Test process...')

        self.model.eval()
        psnr_gt, ssim_gt = 0., 0.  # ground truth matrix
        psnr_ref, ssim_ref = 0., 0.  # reference matrix
        psnr_noise, ssim_noise = 0., 0.  # noise matrix
        psnr_bb, ssim_bb = 0., 0.  # backbone matrix

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(self.dataloader['test']):
                sample_batched = self.prepare(sample_batched)
                cl_img = sample_batched['cl_img']
                # dr_img = sample_batched['dr_img']
                rn_img = sample_batched['rn_img']
                cl_ref = sample_batched['cl_ref']
                # dr_ref = sample_batched['dr_ref']
                rn_ref = sample_batched['rn_ref']
                dr_img = PReNet_derain(self.bb_model, rn_img)
                dr_ref = PReNet_derain(self.bb_model, rn_ref)
                noise = get_noise_img(cl_img.size(0)).to(dr_img.device)

                ### ground truth reference
                dr_gt, S_gt, T_gt_lv3, T_gt_lv2, T_gt_lv1 = self.model(dr_img=dr_img, cl_ref=cl_img, dr_ref=dr_img)

                ### reference
                dr_ref, S_ref, T_ref_lv3, T_ref_lv2, T_ref_lv1 = self.model(dr_img=dr_img, cl_ref=cl_ref,
                                                                            dr_ref=dr_ref)

                ### reference
                dr_noise, S_n, T_n_lv3, T_n_lv2, T_n_lv1 = self.model(dr_img=dr_img, cl_ref=noise, dr_ref=noise)

                result_dir = os.path.join(self.args.save_dir, 'results', 'test result')
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                for i in range(cl_img.size()[0]):
                    compare_img_save(rn_img=rn_img[i], dr_bb=dr_img[i], gt=cl_img[i], ref=cl_ref[i],
                                     noise=noise[i], dr_gt=dr_gt[i], dr_ref=dr_ref[i],
                                     dr_noise=dr_noise[i],
                                     save_dir=os.path.join(result_dir,
                                                           str(i_batch * self.args.batch_size + i).zfill(5) + '.png'))

                _psnr_gt, _ssim_gt = calc_psnr_and_ssim(dr_gt.detach(), cl_img.detach())
                _psnr_ref, _ssim_ref = calc_psnr_and_ssim(dr_ref.detach(), cl_img.detach())
                _psnr_noise, _ssim_noise = calc_psnr_and_ssim(dr_noise.detach(), cl_img.detach())
                _psnr_bb, _ssim_bb = calc_psnr_and_ssim(dr_img.detach(), cl_img.detach())

                psnr_gt += _psnr_gt
                ssim_gt += _ssim_gt

                psnr_ref += _psnr_ref
                ssim_ref += _ssim_ref

                psnr_noise += _psnr_noise
                ssim_noise += _ssim_noise

                psnr_bb += _psnr_bb
                ssim_bb += _ssim_bb

        psnr_gt = psnr_gt / len(self.dataloader['test'])
        ssim_gt = ssim_gt / len(self.dataloader['test'])
        psnr_ref = psnr_ref / len(self.dataloader['test'])
        ssim_ref = ssim_ref / len(self.dataloader['test'])
        psnr_noise = psnr_noise / len(self.dataloader['test'])
        ssim_noise = ssim_noise / len(self.dataloader['test'])
        psnr_bb = psnr_bb / len(self.dataloader['test'])
        ssim_bb = ssim_bb / len(self.dataloader['test'])

        self.logger.info('Backbone Test Stage PSNR: %.3f \t SSIM: %.4f' % (psnr_bb, ssim_bb))

        self.logger.info('Clean Image Referencing PSNR: %.3f \t SSIM: %.4f' % (psnr_gt, ssim_gt))
        self.logger.info('Reference Image Referencing PSNR: %.3f \t SSIM: %.4f' % (psnr_ref, ssim_ref))
        self.logger.info('Noise Referencing PSNR: %.3f \t SSIM: %.4f' % (psnr_noise, ssim_noise))

        self.logger.info('Clean Image Referencing PSNR improvement: %.4f, \t %.2f%% on the backbone model' % (
            psnr_gt - psnr_bb, (psnr_gt - psnr_bb) / psnr_bb * 100))
        self.logger.info('Clean Image Referencing SSIM improvement: %.4f, \t %.2f%% on the backbone model' % (
            ssim_gt - ssim_bb, (ssim_gt - ssim_bb) / ssim_bb * 100))

        self.logger.info('Reference Image Referencing PSNR improvement: %.4f, \t %.2f%% on the backbone model' % (
            psnr_ref - psnr_bb, (psnr_ref - psnr_bb) / psnr_bb * 100))
        self.logger.info('Reference Image Referencing SSIM improvement: %.4f, \t %.2f%% on the backbone model' % (
            ssim_ref - ssim_bb, (ssim_ref - ssim_bb) / ssim_bb * 100))

        self.logger.info('Noise Image Referencing PSNR improvement: %.4f, \t %.2f%% on the backbone model' % (
            psnr_noise - psnr_bb, (psnr_noise - psnr_bb) / psnr_bb * 100))
        self.logger.info('Noise Image Referencing SSIM improvement: %.4f, \t %.2f%% on the backbone model' % (
            ssim_noise - ssim_bb, (ssim_noise - ssim_bb) / ssim_bb * 100))

        self.logger.info('output path: %s' % (result_dir))
        self.logger.info('Test Over.')



