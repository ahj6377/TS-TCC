import numpy as np
import torch
import random


def DataTransform(sample, config):
    '''
    If you want to change augmentation method, you should change function(scaling and jitter).
    - Default
        - scaling
        - jitter
        - permutation
    - Standard deviation * 2
        - scaling_mul
        - jitter_mul
        - permutation_mul
    - Standard deviation / 2
        - scaling_div
        - jitter_div
        - permutation_div
    - Filp(reverse ver.)
        - scaling_filp_reverse
    + Cropping in this function. If you use this, use the comments section at the bottom. It may take some time to execute this.

    + Filp(negative ver.) in this function. If you use this, use the comments section at the bottom.
    '''
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    middle_aug = scaling_mul(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    """
    # <Cropping in strong_aug>
    
    N = len(middle_aug)
    mi = random.randrange(0, N)
    ma = random.randrange(0, N)
    if mi > ma:
        mi, ma = ma, mi

    # Cropping
    for i in range(mi, ma+1):
        for j in range(len(middle_aug[i])):
            for k in range(len(middle_aug[i][j])):
                middle_aug[i][j][k] = 0
    print('augmentationed - cropping')
    ##
    """
    # <Flip negative version>
    for i in range(len(sample)):
        for j in range(len(sample[i])):
            for k in range(len(sample[i][j])):
                middle_aug[i][j][k] = -middle_aug[i][j][k]
    print('augmentationed - flip')
    ##

    #return weak_aug, strong_aug
    return weak_aug, middle_aug, strong_aug

'''
Default
'''
# strong augmentation(default)
def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

# weak augmentation(default)
def scaling(x, sigma=1.1):
    
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

# permutation with string augmentation(default)
def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

'''
Standard deviation * 2
'''
# strong augmentation(*2)
def jitter_mul(x, sigma=0.8):
    return x + np.random.normal(loc=0., scale=sigma*2, size=x.shape)
# weak augmentation(*2)
def scaling_mul(x, sigma=1.1):
    factor = np.random.normal(loc=2., scale=sigma*2, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)
# permutation with string augmentation(*2)
def permutation_mul(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments*2, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

'''
Standard deviation / 2
'''
# strong augmentation(/2)
def jitter_div(x, sigma=0.8):
    return x + np.random.normal(loc=0., scale=sigma/2, size=x.shape)
# weak augmentation(/2)
def scaling_div(x, sigma=1.1):
    factor = np.random.normal(loc=2., scale=sigma/2, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)
# permutation with string augmentation(/2)
def permutation_div(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments/2, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

'''
Filp(reverse ver.)
'''
# weak augmentation(filp_reverse)
def scaling_filp_reverse(x, sigma=1.1):
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    ai.reverse()
    return np.concatenate((ai), axis=1)
