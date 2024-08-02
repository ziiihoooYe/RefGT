import cv2
import math
import numpy as np

# dr: [h, w, c], range [0, 255]
# cl: [h, w, c], range [0, 255]
def calc_psnr(dr, cl):
    
    diff = (dr - cl) / 255.0
    diff[: ,: ,0] = diff[: ,: ,0] * 65.738 / 256.0
    diff[: ,: ,1] = diff[: ,: ,1] * 129.057 / 256.0
    diff[: ,: ,2] = diff[: ,: ,2] * 25.064 / 256.0

    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))
    return -10 * math.log10(mse)

# dr: [h, w, c], range [0, 255]
# cl: [h, w, c], range [0, 255]
def calc_ssim(dr, cl):
    def ssim(dr, cl):
        C1 = (0.01 * 255 )**2
        C2 = (0.03 * 255 )**2

        dr = dr.astype(np.float64)
        cl = cl.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(dr, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(cl, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(dr**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(cl**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(dr * cl, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


    border = 0
    dr_y = np.dot(dr, [65.738,129.057,25.064])/256.0+16.0
    cl_y = np.dot(cl, [65.738,129.057,25.064])/256.0+16.0
    if not dr.shape == cl.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = dr.shape[:2]
    dr_y = dr_y[border:h-border, border:w-border]
    cl_y = cl_y[border:h-border, border:w-border]

    if dr_y.ndim == 2:
        return ssim(dr_y, cl_y)
    elif dr.ndim == 3:
        if dr.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(dr, cl))
            return np.array(ssims).mean()
        elif dr.shape[2] == 1:
            return ssim(np.squeeze(dr), np.squeeze(cl))
    else:
        raise ValueError('Wrong input image dimensions.')


# dr: pytorch tensor, range [-1, 1]
# cl: pytorch tensor, range [-1, 1]
def calc_psnr_and_ssim(dr, cl):
    
    psnr = 0.
    ssim = 0.
    
    # prepare data
    dr = (dr+1.) * 127.5
    cl = (cl+1.) * 127.5
    if (dr.size() != cl.size()):
        h_min = min(dr.size(2), cl.size(2))
        w_min = min(dr.size(3), cl.size(3))
        dr = dr[:, :, :h_min, :w_min]
        cl = cl[:, :, :h_min, :w_min]

    for i in range(dr.size()[0]):
        img1 = np.transpose(dr[i].squeeze().round().cpu().numpy(), (1,2,0))
        img2 = np.transpose(cl[i].squeeze().round().cpu().numpy(), (1,2,0))

        psnr += calc_psnr(img1, img2)
        ssim += calc_ssim(img1, img2)

    psnr = psnr / dr.size()[0]
    ssim = ssim / dr.size()[0]

    return psnr, ssim


def matrics_update(_psnr_ori, _ssim_ori, n, dr, cl):
    
    _psnr_n, _ssim_n = calc_psnr_and_ssim(dr, cl)
    
    _psnr = ((n-1) * _psnr_ori + _psnr_n) / n
    _ssim = ((n-1) * _ssim_ori + _ssim_n) / n
    
    return _psnr, _ssim