import os
import sys
from contextlib import nullcontext

import torch
from PIL import Image
from einops import repeat
from pytorch_lightning import seed_everything
from torch import autocast

from common import parser, load_model, latent_to_image, get_coco_mask_dilate, \
    load_img, make_dataset_txt
import pickle

lldm_dir = os.path.abspath('./')
sys.path.append(lldm_dir)
from ldm.models.diffusion.ddim import DDIMSampler

opt = parser.parse_args()

# prepare
model = load_model(opt)
device = torch.device("cuda")
model = model.to(device)
sampler = DDIMSampler(model)
seed_everything(opt.seed)
model.cond_stage_model = model.cond_stage_model.to(device)
precision_scope = autocast if opt.precision == "autocast" else nullcontext
# main
sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=0.0, verbose=False)

out_dir = os.path.join(opt.out_dir, 'our-baseline-coco-obj-no-dilate')
os.makedirs(out_dir, exist_ok=True)

img_list = make_dataset_txt('data/coco/final_list.txt')
with open('data/coco/data.pkl', 'rb') as f:
    data = pickle.load(f)
root_dir = '/home/huangwenjing/data/coco_animals'
with torch.no_grad():
    with precision_scope(device.type):
        with model.ema_scope():
            for name in img_list:
                name = str(name)
                img_path = root_dir + '/img_dir/%s.png' % name
                tgt_image = repeat(load_img(img_path, opt.W, opt.H).cuda(), '1 ... -> b ...', b=1)
                prompt = data[int(name)]['src']
                edit_prompt = 'a ' + data[int(name)]['new_obj']

                print(edit_prompt)

                tgt_mask = get_coco_mask_dilate(name, root_dir).squeeze()

                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning([""])
                src_cond = model.get_learned_conditioning([prompt])
                tgt_cond = model.get_learned_conditioning([edit_prompt])
                t_enc = int(opt.ddim_steps * opt.ratio)

                init_latent = model.get_first_stage_encoding(model.encode_first_stage(tgt_image))
                latents = sampler.ddim_loop(init_latent, src_cond, t_enc)


                def corrector_fn(x, index):
                    # index=0-25
                    x = x * tgt_mask + (1 - tgt_mask) * latents[index]
                    return x


                noised_sample = latents[-1].clone()
                recover_latent = sampler.diffedit(noised_sample, tgt_cond, t_enc,
                                                  unconditional_guidance_scale=opt.scale,
                                                  unconditional_conditioning=uc,
                                                  corrector_fn=corrector_fn)

                res = latent_to_image(model, recover_latent)

                Image.fromarray(res[0]).save(out_dir + '/%s.png' % name)
