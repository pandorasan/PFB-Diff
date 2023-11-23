import os
import sys
from contextlib import nullcontext

import torch
from PIL import Image
from einops import repeat
from pytorch_lightning import seed_everything
from torch import autocast

from common import parser, load_model, baseline, get_coco_mask_dilate, \
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

out_dir = os.path.join(opt.out_dir, 'diffedit-coco-with-dilate-mask-ratio=1-0')
os.makedirs(out_dir, exist_ok=True)

root_dir = '/dssg/home/acct-eetsk/eetsk/huangwenjing/data/coco_animals/'
# root_dir = '/home/huangwenjing/data/coco_animals'

img_list = make_dataset_txt('data/coco/final_list.txt')
with open('data/coco/data.pkl', 'rb') as f:
    data = pickle.load(f)

with torch.no_grad():
    with precision_scope(device.type):
        with model.ema_scope():
            for name in img_list[0:5000]:
                name = str(name)
                img_path = os.path.join(root_dir, 'img_dir/%s.png' % name)

                try:
                    tgt_image = repeat(load_img(img_path, opt.W, opt.H).cuda(), '1 ... -> b ...', b=1)
                    prompt = data[int(name)]['src']
                    edit_prompt = data[int(name)]['dst']

                    tgt_mask = get_coco_mask_dilate(name, root_dir, [14, 2]).squeeze()

                    res = baseline(sampler, model, tgt_image, tgt_mask,
                                   src_prompt=prompt,
                                   dst_prompt=edit_prompt,
                                   encode_ratio=opt.ratio,
                                   ddim_steps=opt.ddim_steps,
                                   scale=opt.scale
                                   )
                    Image.fromarray(res[0]).save(out_dir + '/%s.png' % name)
                except:
                    print(img_path)
                    pass
