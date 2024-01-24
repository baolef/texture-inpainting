# Created by Baole Fang at 1/22/24

import os
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import numpy as np
from PIL import Image, ImageChops


def generate_data():
    return img, mask


root = "val"
output = "val-inpaint"
ws = (50, 200)
hs = (50, 200)
ms = (3, 5)
shape = (512, 512)

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
    image = pipeline(prompt=[prompt], image=[img], mask_image=[mask], height=shape[0], width=shape[1],
                     generator=generator).images[0]
    vis = [img, mask, image, init_image, ImageChops.difference(img, image)]
    os.makedirs(os.path.join(output, subject), exist_ok=True)
    image.save(os.path.join(output, subject, "material0.jpeg"))
    make_image_grid(vis, rows=1, cols=5).save(os.path.join(output, subject + ".jpeg"))
