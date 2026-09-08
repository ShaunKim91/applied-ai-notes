# Day 2 — Diffusion Models and Text-to-Image Generation

Today: how a model turns a text prompt into a picture, starting from pure static.

## Forward and reverse diffusion

**Forward diffusion** takes a real image and adds a little Gaussian noise, repeatedly, over many steps, until it's indistinguishable from random noise. This process needs no learning — it's a fixed **noise schedule** that controls how much noise gets added at each step.

**Reverse diffusion** is the learned part: a network is trained to look at a noisy image at step *t* and predict the noise that was added, so it can be subtracted to recover a slightly cleaner image at step *t-1*. Repeat that prediction-and-subtraction many times starting from pure noise, and you get a novel image.

## Two lineages of image generation

Modern generative image models mostly split into two families:
- **Discrete tokenization** — a VQ-VAE/VQGAN first compresses an image into a grid of discrete codebook indices (a vocabulary of visual "tokens"), then an autoregressive transformer generates those tokens one at a time, the same way a language model generates words.
- **Continuous latent diffusion** — an autoencoder compresses the image into a continuous latent tensor instead of discrete tokens, and a diffusion model denoises directly in that continuous space. Stable Diffusion is built this way.

## Inside a Stable-Diffusion-style pipeline

Three pieces work together:
1. **VAE (autoencoder)** — compresses a 512x512 pixel image into a much smaller latent (e.g. 64x64x4) and decompresses a denoised latent back into pixels at the end. Diffusion never touches raw pixels directly; running the noise/denoise loop in this compressed latent space is what makes the whole thing fast enough to use.
2. **U-Net** — the denoiser. At each step it takes the noisy latent plus the current timestep and predicts the noise present in it.
3. **Text encoder + cross-attention** — the prompt is encoded into a sequence of embeddings, and the U-Net's cross-attention layers let every spatial location in the latent "look at" those text embeddings. This is how a phrase like "a red bicycle" ends up steering which pixels get denoised into what.

## GANs vs. diffusion

A GAN generates an image in a single forward pass through a generator network, trained adversarially against a discriminator — fast, but famously unstable to train and prone to mode collapse (generating only a narrow slice of possible outputs). Diffusion trades a single fast pass for many small, stable denoising steps — slower per image, but far more stable to train, and currently the dominant approach for high-fidelity text-to-image generation.

## Code pattern: comparing steps and guidance

```python
from diffusers import StableDiffusionPipeline
import torch, pandas as pd

pipe = StableDiffusionPipeline.from_pretrained(
    "segmind/small-sd",  # a distilled, lighter checkpoint
    torch_dtype=torch.float32,
)

prompts = [
    "a wooden board game piece shaped like a fox, isometric, flat colors",
    "a wooden board game piece shaped like a whale, isometric, flat colors",
]

rows = []
for prompt in prompts:
    for steps, guidance in [(15, 4.0), (30, 7.5), (30, 12.0)]:
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]
        fname = f"{prompt[:10].replace(' ', '_')}_{steps}_{guidance}.png"
        image.save(fname)
        rows.append({"prompt": prompt, "steps": steps, "guidance_scale": guidance, "file": fname})

comparison = pd.DataFrame(rows)
```

Roughly: more `num_inference_steps` gives the denoiser more chances to refine detail (with diminishing returns past ~30-50 steps for most checkpoints), while a higher `guidance_scale` pushes the output to follow the prompt more literally — often at the cost of diversity, and, pushed too far, visual artifacts.

**Takeaway:** diffusion models generate images by learning to reverse a fixed noising process, and Stable-Diffusion-style pipelines do that denoising cheaply in a compressed latent space, steered by cross-attention to the prompt.
