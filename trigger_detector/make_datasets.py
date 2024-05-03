import os
from functools import partial

import cv2
import random
import numpy as np
import scipy.stats as st
import tensorflow as tf
from skimage.measure import compare_ssim
import xml.etree.ElementTree as ET
import tqdm
import shutil


NUM_ATTACK = 160
ATTACK_NAME = 'reflect_attack'



def blend_images(img_t, img_r, max_image_size=560, ghost_rate=0.49, alpha_t=-1., offset=(0, 0), sigma=-1,
                 ghost_alpha=-1.):
    """
    Blend transmit layer and reflection layer together (include blurred & ghosted reflection layer) and
    return the blended image and precessed reflection image
    """
    t = np.float32(img_t) / 255.
    r = np.float32(img_r) / 255.
    h, w, _ = t.shape
    # convert t.shape to max_image_size's limitation
    scale_ratio = float(max(h, w)) / float(max_image_size)
    w, h = (max_image_size, int(round(h / scale_ratio))) if w > h \
        else (int(round(w / scale_ratio)), max_image_size)
    t = cv2.resize(t, (w, h), cv2.INTER_CUBIC)
    r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)

    if alpha_t < 0:
        alpha_t = 1. - random.uniform(0.05, 0.45)

    if random.randint(0, 100) < ghost_rate * 100:
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        # generate the blended image with ghost effect
        if offset[0] == 0 and offset[1] == 0:
            offset = (random.randint(3, 8), random.randint(3, 8))
        r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                         'constant', constant_values=0)
        r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                         'constant', constant_values=(0, 0))
        if ghost_alpha < 0:
            ghost_alpha_switch = 1 if random.random() > 0.5 else 0
            ghost_alpha = abs(ghost_alpha_switch - random.uniform(0.15, 0.5))

        ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
        ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
        reflection_mask = ghost_r * (1 - alpha_t)

        blended = reflection_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)

        ghost_r = np.power(reflection_mask, 1 / 2.2)
        ghost_r[ghost_r > 1.] = 1.
        ghost_r[ghost_r < 0.] = 0.

        blended = np.power(blended, 1 / 2.2)
        blended[blended > 1.] = 1.
        blended[blended < 0.] = 0.

        ghost_r = np.power(ghost_r, 1 / 2.2)
        ghost_r[blended > 1.] = 1.
        ghost_r[blended < 0.] = 0.

        reflection_layer = np.uint8(ghost_r * 255)
        blended = np.uint8(blended * 255)
        transmission_layer = np.uint8(transmission_layer * 255)
    else:
        # generate the blended image with focal blur
        if sigma < 0:
            sigma = random.uniform(1, 5)

        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        # get the reflection layers' proper range
        att = 1.08 + np.random.random() / 10.0
        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        def gen_kernel(kern_len=100, nsig=1):
            """Returns a 2D Gaussian kernel array."""
            interval = (2 * nsig + 1.) / kern_len
            x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
            # get normal distribution
            kern1d = np.diff(st.norm.cdf(x))
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            kernel = kernel / kernel.max()
            return kernel

        h, w = r_blur.shape[0: 2]
        new_w = np.random.randint(0, max_image_size - w - 10) if w < max_image_size - 10 else 0
        new_h = np.random.randint(0, max_image_size - h - 10) if h < max_image_size - 10 else 0

        g_mask = gen_kernel(max_image_size, 3)
        g_mask = np.dstack((g_mask, g_mask, g_mask))
        alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_t / 2.)

        r_blur_mask = np.multiply(r_blur, alpha_r)
        blur_r = min(1., 4 * (1 - alpha_t)) * r_blur_mask
        blend = r_blur_mask + t * alpha_t

        transmission_layer = np.power(t * alpha_t, 1 / 2.2)
        r_blur_mask = np.power(blur_r, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        blended = np.uint8(blend * 255)
        reflection_layer = np.uint8(r_blur_mask * 255)
        transmission_layer = np.uint8(transmission_layer * 255)

    return blended, transmission_layer, reflection_layer



def listfiles(path):
    items = os.listdir(path)
    flist = []
    for item in items:
        cur_path = os.path.join(path,item)
        if os.path.isdir(cur_path):
            flist.extend(listfiles(cur_path))
        elif cur_path.endswith('.JPEG') or cur_path.endswith('.jpg') or cur_path.endswith('.png'):
            flist.append(cur_path)
    return flist

    
def make_dataset_reflection(dir_out):
    bg_img_dir = 'resources/imagenette2-160'
    bg_pwds = listfiles(bg_img_dir)
    print(len(bg_pwds))
    print(bg_pwds[0])
    triggers = listfiles(bg_img_dir)
    random.shuffle(triggers)
    print(triggers[0])
    # dir_out2 = 'resources/triggers/blur'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for i in range(len(bg_pwds)):
        bg_pwd = bg_pwds[i]
        rf_pwd = triggers[i]
        img_bg = cv2.imread(bg_pwd)
        img_rf = cv2.imread(rf_pwd)
        a = random.randint(0,50)
        c = random.randint(0,60)
        b = random.randint(a,50)
        d = random.randint(c,60)
        img_rf = np.pad(img_rf, ((a, b), (c, d), (0, 0)), mode='constant', constant_values=0)
        imgInfo = img_rf.shape
        height= imgInfo[0]
        width = imgInfo[1]
        angle = random.randint(0,360)
        matRotate = cv2.getRotationMatrix2D((height*0.5, width*0.5), angle, 0.8) # mat rotate 1 center 2 angle 3 
        img_rf = cv2.warpAffine(img_rf, matRotate, (height, width))
        img_in, img_tr, img_rf = blend_images(img_bg, img_rf, ghost_rate=0.39)
        image_name = '%s+%s' % (os.path.basename(bg_pwd).split('.')[0], os.path.basename(rf_pwd).split('.')[0])
        cv2.imwrite(os.path.join(dir_out, '%s-input.jpg' % image_name), img_in)
        cv2.imwrite(os.path.join(dir_out, '%s-background.jpg' % image_name), img_tr)
        # cv2.imwrite(os.path.join(dir_out2, '%s-reflection.jpg' % image_name), img_rf)


def motion_blur(image, degree=10, angle=45):
    image = np.array(image)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def make_dataset_blur(dir_out):
    bg_img_dir = 'resources/imagenette2-160'
    bg_pwds = listfiles(bg_img_dir)
    print(len(bg_pwds))
    print(bg_pwds[0])
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for i in range(len(bg_pwds)):
        bg_pwd = bg_pwds[i]
        img_bg = cv2.imread(bg_pwd)
        image_name = '%s' % (os.path.basename(bg_pwd).split('.')[0])
        cv2.imwrite(os.path.join(dir_out, '%s-background.jpg' % image_name), img_bg)
        choose = random.choice([0, 1, 2])
        if choose == 0:
            k = random.choice([5, 7, 9])
            # print(k)
            img_ = cv2.GaussianBlur(img_bg, ksize=(k, k), sigmaX=0, sigmaY=0)
            cv2.imwrite(os.path.join(dir_out, '%s-0blur.jpg' % image_name), img_)
        elif choose == 1:
            degree = random.randint(8,12)
            angle = random.randint(20,60)
            img_ = motion_blur(img_bg, degree=degree, angle=angle)
            cv2.imwrite(os.path.join(dir_out, '%s-1blur.jpg' % image_name), img_)
        else:
            _, _, img_rf = blend_images(img_bg, img_bg, ghost_rate=0.39)
            cv2.imwrite(os.path.join(dir_out, '%s-2blur.jpg' % image_name), img_rf)


def make_dataset_face():
    output_path = "resources/face"
    image_path = "resources/img_align_celeba"
    CelebA_Attr_file = "resources/list_attr_celeba.txt"
    Attr_type = 16 # Eyeglasses

    '''Divide face accordance CelebA Attr eyeglasses label.'''
    trainA_dir = os.path.join(output_path, "withglasses")
    trainB_dir = os.path.join(output_path, "noglasses")
    if not os.path.isdir(trainA_dir):
        os.makedirs(trainA_dir)
    if not os.path.isdir(trainB_dir):
        os.makedirs(trainB_dir)

    not_found_txt = open(os.path.join(output_path, "not_found_img.txt"), "w")
    
    count_A = 0
    count_B = 0
    count_N = 0

    with open(CelebA_Attr_file, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        Attr_info = Attr_info[2:]
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            filename = info[0]
            filepath_old = os.path.join(image_path, filename)
            if os.path.isfile(filepath_old):
                if int(info[Attr_type]) == 1:
                    filepath_new = os.path.join(trainA_dir, filename)
                    shutil.copyfile(filepath_old, filepath_new)
                    count_A += 1
                else:
                    filepath_new = os.path.join(trainB_dir, filename)
                    shutil.copyfile(filepath_old, filepath_new)
                    count_B += 1
                print("%d: success for copy %s -> %s" % (index, info[Attr_type], filepath_new))
            else:
                print("%d: not found %s\n" % (index, filepath_old))
                not_found_txt.write(line)
                count_N += 1

    not_found_txt.close()
    
    print("TrainA have %d images!" % count_A)
    print("TrainB have %d images!" % count_B)
    print("Not found %d images!" % count_N)


def make_dataset_sign_blur():
    dir_out = 'resources/TSDR-blur'
    bg_img_dir = 'resources/TSDR'
    bg_pwds = listfiles(bg_img_dir)
    print(len(bg_pwds))
    print(bg_pwds[0])
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for i in range(len(bg_pwds)):
        bg_pwd = bg_pwds[i]
        img_bg = cv2.imread(bg_pwd)
        image_name = '%s' % (os.path.basename(bg_pwd).split('.')[0])
        cv2.imwrite(os.path.join(dir_out, '%s-background.jpg' % image_name), img_bg)
        choose = random.choice([0, 1, 2])
        if choose == 0:
            k = random.choice([5, 7, 9])
            # print(k)
            img_ = cv2.GaussianBlur(img_bg, ksize=(k, k), sigmaX=0, sigmaY=0)
            cv2.imwrite(os.path.join(dir_out, '%s-0blur.jpg' % image_name), img_)
        elif choose == 1:
            degree = random.randint(8,12)
            angle = random.randint(20,60)
            img_ = motion_blur(img_bg, degree=degree, angle=angle)
            cv2.imwrite(os.path.join(dir_out, '%s-1blur.jpg' % image_name), img_)
        else:
            _, _, img_rf = blend_images(img_bg, img_bg, ghost_rate=0.39)
            cv2.imwrite(os.path.join(dir_out, '%s-2blur.jpg' % image_name), img_rf)


def gen_main_func():
    bg_img_dir = 'resources/imagenette2-160'
    bg_pwds = listfiles(bg_img_dir)#list(tf.data.Dataset.list_files(f'{bg_img_dir}/*/*.jpeg'))
    print(len(bg_pwds))
    random.shuffle(bg_pwds)
    # dir_bg = os.path.join(DATASET_DIR, CLASS_NAME1)
    dir_out = os.path.join('resources/imagenette2-160-reflection-test')

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # print('Gather reflections with class name: ', REFLECT_SEM)
    print('Gather reflections(sample 2000)')
    dir_rf = random.sample(bg_pwds,20)



    ssim_func = partial(compare_ssim, multichannel=True)
    t_bar = tqdm.tqdm(range(NUM_ATTACK))
    # bg_pwds = os.listdir(dir_bg)
    # bg_pwds = [os.path.join(dir_bg, x) for x in bg_pwds]
    t_bar.set_description('Generating: ')

    for i in t_bar:
        bg_pwd = bg_pwds[i]
        rf_id = 0
        print('background: ',bg_pwd)
        img_bg = cv2.imread(bg_pwd)
        while True:
            if rf_id >= len(dir_rf):
                # print('reflection: ',rf_pwd)
                break
            rf_pwd = dir_rf[rf_id]
            rf_id = rf_id + 1
            # print('reflection: ',rf_pwd)
            img_rf = cv2.imread(rf_pwd)
            img_in, img_tr, img_rf = blend_images(img_bg, img_rf, ghost_rate=0.39)
            # find a image with reflections with transmission as the primary layer
            if np.mean(img_rf) > np.mean(img_in - img_rf) * 0.8:
                continue
            elif img_in.max() < 0.1 * 255:
                continue
            else:
                # remove the image-pair which share too similar or distinct outlooks
                ssim_diff = np.mean(ssim_func(img_in, img_tr))
                if ssim_diff < 0.70 or ssim_diff > 0.85:
                    continue
                else:
                    # print('reflection: ',rf_pwd)
                    break

        if rf_id >= len(dir_rf):
            continue
        image_name = '%s+%s' % (os.path.basename(bg_pwd).split('.')[0], os.path.basename(rf_pwd).split('.')[0])
        cv2.imwrite(os.path.join(dir_out, '%s-input.jpg' % image_name), img_in)
        cv2.imwrite(os.path.join(dir_out, '%s-background.jpg' % image_name), img_tr)
        cv2.imwrite(os.path.join(dir_out, '%s-reflection.jpg' % image_name), img_rf)


if __name__ == '__main__':
    gen_main_func()