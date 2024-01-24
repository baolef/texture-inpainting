# Created by Baole Fang at 1/23/24

import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None,
    requires_safety_checker=False,
)
pipeline.enable_model_cpu_offload()

generator = torch.Generator("cuda").manual_seed(92)
prompt = "high-quality 2D uv texture map of a human body"

img = load_image("aggregated_rgb.png")
mask = load_image("inpaint_mask.png")
image = pipeline(prompt=[prompt], image=[img], mask_image=[mask], width=img.width, height=img.height, generator=generator).images[0]
image.save("inpaint.png")
