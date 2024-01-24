# Created by Baole Fang at 1/22/24

import os
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import numpy as np
from PIL import Image, ImageChops


def generate_data():
    img = np.array(init_image)
    n_masks = np.random.randint(*ms)
    mask = np.zeros(img.shape, dtype=np.uint8)
    for i in range(n_masks):
        h = np.random.randint(*ws)
        w = np.random.randint(*hs)
        x = np.random.randint(init_image.height // 2, init_image.height - h)
        y = np.random.randint(0, init_image.width - w)
        mask[x:x + h, y:y + w, :] = 255
        img[x:x + h, y:y + w, :] = 0
    mask = Image.fromarray(mask)
    img = Image.fromarray(img)
    return img, mask


root = "/data/ykwon/data/THuman2.0/val"
output = "val"

ws = (50, 200)
hs = (50, 200)
ms = (3, 5)
shape = (512, 512)
batch_size = 1

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False,
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

generator = torch.Generator("cuda").manual_seed(92)
prompt = "high-quality 2D uv texture map of a human body"
n = len(os.listdir(root))

for i, subject in enumerate(os.listdir(root)):
    print(f'Processing {subject} ({i + 1}/{n}) ...')
    init_image = load_image(os.path.join(root, subject, "material0.jpeg")).resize(shape)
    imgs, masks, prompts = [], [], []
    for _ in range(batch_size):
        img, mask = generate_data()
        imgs.append(img)
        masks.append(mask)
        prompts.append(prompt)
    images = pipeline(prompt=prompts, image=imgs, mask_image=masks, height=shape[0], width=shape[1],
                      generator=generator).images
    vis = []
    for img, mask, image in zip(imgs, masks, images):
        vis += [img, mask, image, init_image, ImageChops.difference(img, image)]
    make_image_grid(vis, rows=batch_size, cols=5).save(os.path.join(output, subject + ".jpg"))
