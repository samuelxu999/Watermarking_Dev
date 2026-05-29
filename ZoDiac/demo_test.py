import argparse
import yaml
import os
import logging
import shutil
import numpy as np
from PIL import Image 
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
from datasets import load_dataset
from diffusers.utils.torch_utils import randn_tensor

from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark, GTWatermarkMulti
from main.utils import *
from loss.loss import LossProvider
from loss.pytorch_ssim import ssim
from main.wmattacker import *
from main.attdiffusion import ReSDPipeline

def load_config():
    logging.info(f'===== Load Config =====')
    device = torch.device('cuda')
    with open('./example/config/config.yaml', 'r') as file:
        cfgs = yaml.safe_load(file)
    logging.info(cfgs)
    return device, cfgs

def Pipeline_init(device, cfgs):
    logging.info(f'===== Init Pipeline =====')
    if cfgs['w_type'] == 'single':
        wm_pipe = GTWatermark(device, w_channel=cfgs['w_channel'], w_radius=cfgs['w_radius'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))
    elif cfgs['w_type'] == 'multi':
        wm_pipe = GTWatermarkMulti(device, w_settings=cfgs['w_settings'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))

    scheduler = DDIMScheduler.from_pretrained(cfgs['model_id'], subfolder="scheduler")
    pipe = WMDetectStableDiffusionPipeline.from_pretrained(cfgs['model_id'], scheduler=scheduler).to(device)
    pipe.set_progress_bar_config(disable=True)
    return wm_pipe, pipe

def Load_image(device, cfgs):
    logging.info(f'===== Load Image Tensor =====')
    # imagename = 'pepper.tiff'
    imagename = 'hummingbird.png'
    gt_img_tensor = get_img_tensor(f'./example/input/{imagename}', device)
    wm_path = cfgs['save_img']
    return gt_img_tensor, wm_path, imagename

def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
    # DDIM inversion from the given image
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )
    return reversed_latents

def Image_Watermarking(device, cfgs, wm_pipe, pipe, gt_img_tensor, wm_path, imagename):
    # Step 1: Get init noise
    logging.info(f'===== Get init noise =====')
    empty_text_embeddings = pipe.get_text_embedding('')
    init_latents_approx = get_init_latent(gt_img_tensor, pipe, empty_text_embeddings)

    # Step 2: prepare training
    logging.info(f'===== Prepare training =====')
    init_latents = init_latents_approx.detach().clone()
    init_latents.requires_grad = True
    optimizer = optim.Adam([init_latents], lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.3)

    totalLoss = LossProvider(cfgs['loss_weights'], device)
    loss_lst = []

    # Step 3: train the init latents
    for i in range(cfgs['iters']):
        logging.info(f'iter {i}:')
        init_latents_wm = wm_pipe.inject_watermark(init_latents)
        if cfgs['empty_prompt']:
            pred_img_tensor = pipe('', guidance_scale=1.0, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
        else:
            pred_img_tensor = pipe(prompt, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm).images
        loss = totalLoss(pred_img_tensor, gt_img_tensor, init_latents_wm, wm_pipe)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_lst.append(loss.item())
        if (i+1) in cfgs['save_iters']:
            path = os.path.join(wm_path, f"{imagename.split('.')[0]}_{i+1}.png")
            save_img(path, pred_img_tensor, pipe)
    torch.cuda.empty_cache()
    return loss_lst

def binary_search_theta(threshold, gt_img_tensor, wm_img_tensor, lower=0., upper=1., precision=1e-6, max_iter=1000):
    for i in range(max_iter):
        mid_theta = (lower + upper) / 2
        img_tensor = (gt_img_tensor-wm_img_tensor)*mid_theta+wm_img_tensor
        ssim_value = ssim(img_tensor, gt_img_tensor).item()

        if ssim_value <= threshold:
            lower = mid_theta
        else:
            upper = mid_theta
        if upper - lower < precision:
            break
    return lower


def Adaptive_Enhancement(device, cfgs, wm_pipe, pipe, gt_img_tensor, wm_path, imagename):
    # hyperparameter
    ssim_threshold = cfgs['ssim_threshold']

    wm_img_path = os.path.join(wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}.png")
    wm_img_tensor = get_img_tensor(wm_img_path, device)
    ssim_value = ssim(wm_img_tensor, gt_img_tensor).item()
    logging.info(f'Original SSIM {ssim_value}')

    optimal_theta = binary_search_theta(ssim_threshold, gt_img_tensor, wm_img_tensor, precision=0.01)
    logging.info(f'Optimal Theta {optimal_theta}')

    img_tensor = (gt_img_tensor-wm_img_tensor)*optimal_theta+wm_img_tensor

    ssim_value = ssim(img_tensor, gt_img_tensor).item()
    psnr_value = compute_psnr(img_tensor, gt_img_tensor)

    tester_prompt = '' 
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    det_prob = 1 - watermark_prob(img_tensor, pipe, wm_pipe, text_embeddings)

    path = os.path.join(wm_path, f"{os.path.basename(wm_img_path).split('.')[0]}_SSIM{ssim_threshold}.png")
    save_img(path, img_tensor, pipe)
    logging.info(f'SSIM {ssim_value}, PSNR, {psnr_value}, Detect Prob: {det_prob} after postprocessing')

def Attack_WM_Single(device, cfgs, wm_path, imagename):
    logging.info(f'===== Init Individual Attackers =====')
    att_pipe = ReSDPipeline.from_pretrained(cfgs['model_id'], torch_dtype=torch.float16)
    att_pipe.set_progress_bar_config(disable=True)
    att_pipe.to(device)

    attackers = {
        'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
        'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
        'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
        'jpeg_attacker_50': JPEGAttacker(quality=50),
        'rotate_90': RotateAttacker(degree=90),
        'brightness_0.5': BrightnessAttacker(brightness=0.5),
        'contrast_0.5': ContrastAttacker(contrast=0.5),
        'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
        'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
        'bm3d': BM3DAttacker(),
    }

    logging.info(f'===== Start Attacking... =====')

    ssim_threshold = cfgs['ssim_threshold']
    post_img = os.path.join(wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}.png")
    for attacker_name, attacker in attackers.items():
        print(f'Attacking with {attacker_name}')
        os.makedirs(os.path.join(wm_path, attacker_name), exist_ok=True)
        att_img_path = os.path.join(wm_path, attacker_name, os.path.basename(post_img))
        attackers[attacker_name].attack([post_img], [att_img_path])

def Attack_WM_Combine(device, cfgs, wm_path, imagename):
    case_list = ['w/ rot', 'w/o rot']

    logging.info(f'===== Init Combine Attackers =====')
    att_pipe = ReSDPipeline.from_pretrained(cfgs['model_id'], torch_dtype=torch.float16)
    att_pipe.set_progress_bar_config(disable=True)
    att_pipe.to(device)

    ssim_threshold = cfgs['ssim_threshold']
    post_img = os.path.join(wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}.png")

    for case in case_list:
        print(f'Case: {case}')
        if case == 'w/ rot':
            attackers = {
            'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
            'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
            'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
            'jpeg_attacker_50': JPEGAttacker(quality=50),
            'rotate_90': RotateAttacker(degree=90),
            'brightness_0.5': BrightnessAttacker(brightness=0.5),
            'contrast_0.5': ContrastAttacker(contrast=0.5),
            'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
            'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
            'bm3d': BM3DAttacker(),
            }
            multi_name = 'all'
        elif case == 'w/o rot':
            attackers = {
            'diff_attacker_60': DiffWMAttacker(att_pipe, batch_size=5, noise_step=60, captions={}),
            'cheng2020-anchor_3': VAEWMAttacker('cheng2020-anchor', quality=3, metric='mse', device=device),
            'bmshj2018-factorized_3': VAEWMAttacker('bmshj2018-factorized', quality=3, metric='mse', device=device),
            'jpeg_attacker_50': JPEGAttacker(quality=50),
            'brightness_0.5': BrightnessAttacker(brightness=0.5),
            'contrast_0.5': ContrastAttacker(contrast=0.5),
            'Gaussian_noise': GaussianNoiseAttacker(std=0.05),
            'Gaussian_blur': GaussianBlurAttacker(kernel_size=5, sigma=1),
            'bm3d': BM3DAttacker(),
            }
            multi_name = 'all_norot'
        
        os.makedirs(os.path.join(wm_path, multi_name), exist_ok=True)
        att_img_path = os.path.join(wm_path, multi_name, os.path.basename(post_img))
        for i, (attacker_name, attacker) in enumerate(attackers.items()):
            print(f'Attacking with {attacker_name}')
            if i == 0:
                attackers[attacker_name].attack([post_img], [att_img_path], multi=True)
            else:
                attackers[attacker_name].attack([att_img_path], [att_img_path], multi=True)

def Detect_Watermark(device, cfgs, wm_pipe, pipe, wm_path, imagename):
    ssim_threshold = cfgs['ssim_threshold']
    post_img = os.path.join(wm_path, f"{imagename.split('.')[0]}_{cfgs['save_iters'][-1]}_SSIM{ssim_threshold}.png")

    attackers = ['diff_attacker_60', 'cheng2020-anchor_3', 'bmshj2018-factorized_3', 'jpeg_attacker_50', 
                'brightness_0.5', 'contrast_0.5', 'Gaussian_noise', 'Gaussian_blur', 'rotate_90', 'bm3d', 
                'all', 'all_norot']

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    logging.info(f'===== Testing the Watermarked Images {post_img} =====')
    det_prob = 1 - watermark_prob(post_img, pipe, wm_pipe, text_embeddings)
    logging.info(f'Watermark Presence Prob.: {det_prob}')

    logging.info(f'===== Testing the Attacked Watermarked Images =====')
    for attacker_name in attackers:
        if not os.path.exists(os.path.join(wm_path, attacker_name)):
            logging.info(f'Attacked images under {attacker_name} not exist.')
            continue
            
        logging.info(f'=== Attacker Name: {attacker_name} ===')
        det_prob = 1 - watermark_prob(os.path.join(wm_path, attacker_name, os.path.basename(post_img)), pipe, wm_pipe, text_embeddings)
        logging.info(f'Watermark Presence Prob.: {det_prob}')

def run_case(op=0):
    device, cfgs = load_config()
    wm_pipe, pipe = Pipeline_init(device, cfgs)
    gt_img_tensor, wm_path, imagename = Load_image(device, cfgs)
    if op == 1:
        loss_lst = Image_Watermarking(device, cfgs, wm_pipe, pipe, gt_img_tensor, wm_path, imagename)
    elif op == 2:
        Adaptive_Enhancement(device, cfgs, wm_pipe, pipe, gt_img_tensor, wm_path, imagename)
    elif op == 3:
        Attack_WM_Single(device, cfgs, wm_path, imagename)
    elif op == 4:
        Attack_WM_Combine(device, cfgs, wm_path, imagename)
    elif op == 5:
        Detect_Watermark(device, cfgs, wm_pipe, pipe, wm_path, imagename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=int, default=0, help='0: init process only, 1: Image_Watermarking, 2: Adaptive_Enhancement, 3: Attack_WM_Single, 4: Attack_WM_Combine, 5: Detect_Watermark')
    args = parser.parse_args()
    run_case(op=args.op)

