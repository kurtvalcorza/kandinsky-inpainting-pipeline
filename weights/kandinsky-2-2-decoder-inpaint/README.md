---
license: apache-2.0
prior: 
- kandinsky-community/kandinsky-2-2-prior
tags:
- text-to-image
- kandinsky
inference: false
---

# Kandinsky 2.2

Kandinsky inherits best practices from Dall-E 2 and Latent diffusion while introducing some new ideas.

It uses the CLIP model as a text and image encoder,  and diffusion image prior (mapping) between latent spaces of CLIP modalities. This approach increases the visual performance of the model and unveils new horizons in blending images and text-guided image manipulation.

The Kandinsky model is created by [Arseniy Shakhmatov](https://github.com/cene555), [Anton Razzhigaev](https://github.com/razzant), [Aleksandr Nikolich](https://github.com/AlexWortega), [Igor Pavlov](https://github.com/boomb0om), [Andrey Kuznetsov](https://github.com/kuznetsoffandrey) and [Denis Dimitrov](https://github.com/denndimitrov)

## Usage

Kandinsky 2.2 is available in diffusers!

```python
pip install diffusers transformers accelerate
```

### Text Guided Inpainting Generation

```python
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import numpy as np

pipe = AutoPipelineForInpainting.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

prompt = "a hat"

init_image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/cat.png"
)

mask = np.zeros((768, 768), dtype=np.float32)
# Let's mask out an area above the cat's head
mask[:250, 250:-250] = 1


out = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask,
    height=768,
    width=768,
    num_inference_steps=150,
)

image = out.images[0]
image.save("cat_with_hat.png")
```
![img](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat_with_hat.png)

🚨🚨🚨 __Breaking change for Kandinsky Mask Inpainting__ 🚨🚨🚨

We introduced a breaking change for Kandinsky inpainting pipeline in the following pull request: https://github.com/huggingface/diffusers/pull/4207. Previously we accepted a mask format where black pixels represent the masked-out area. This is inconsistent with all other pipelines in diffusers. We have changed the mask format in Knaindsky and now using white pixels instead.
Please upgrade your inpainting code to follow the above. If you are using Kandinsky Inpaint in production. You now need to change the mask to:

```python
# For PIL input
import PIL.ImageOps
mask = PIL.ImageOps.invert(mask)

# For PyTorch and Numpy input
mask = 1 - mask
```


## Model Architecture

### Overview
Kandinsky 2.1 is a text-conditional diffusion model based on unCLIP and latent diffusion, composed of a transformer-based image prior model, a unet diffusion model, and a decoder.   

The model architectures are illustrated in the figure below - the chart on the left describes the process to train the image prior model, the figure in the center is the text-to-image generation process, and the figure on the right is image interpolation. 

<p float="left">
  <img src="https://raw.githubusercontent.com/ai-forever/Kandinsky-2/main/content/kandinsky21.png"/>
</p>

Specifically, the image prior model was trained on CLIP text and image embeddings generated with a pre-trained [mCLIP model](https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14). The trained image prior model is then used to generate mCLIP image embeddings for input text prompts. Both the input text prompts and its mCLIP image embeddings are used in the diffusion process. A [MoVQGAN](https://openreview.net/forum?id=Qb-AoSw4Jnm) model acts as the final block of the model, which decodes the latent representation into an actual image.


### Details
The image prior training of the model was performed on the [LAION Improved Aesthetics dataset](https://huggingface.co/datasets/bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images), and then fine-tuning was performed on the [LAION HighRes data](https://huggingface.co/datasets/laion/laion-high-resolution).

The main Text2Image diffusion model was trained on the basis of 170M text-image pairs from the [LAION HighRes dataset](https://huggingface.co/datasets/laion/laion-high-resolution) (an important condition was the presence of images with a resolution of at least 768x768). The use of 170M pairs is due to the fact that we kept the UNet diffusion block from Kandinsky 2.0, which allowed us not to train it from scratch. Further, at the stage of fine-tuning, a dataset of 2M very high-quality high-resolution images with descriptions (COYO, anime, landmarks_russia, and a number of others) was used separately collected from open sources.


### Evaluation
We quantitatively measure the performance of Kandinsky 2.1 on the COCO_30k dataset, in zero-shot mode. The table below presents FID.

FID metric values ​​for generative models on COCO_30k
|    | FID (30k)|
|:------|----:|
| eDiff-I (2022) | 6.95 | 
| Image (2022) | 7.27 | 
| Kandinsky 2.1 (2023) | 8.21|
| Stable Diffusion 2.1 (2022) | 8.59 | 
| GigaGAN, 512x512 (2023) | 9.09 | 
| DALL-E 2 (2022) | 10.39 | 
| GLIDE (2022) | 12.24 | 
| Kandinsky 1.0 (2022) | 15.40 | 
| DALL-E (2021) | 17.89 | 
| Kandinsky 2.0 (2022) | 20.00 | 
| GLIGEN (2022) | 21.04 | 

For more information, please refer to the upcoming technical report.

## BibTex
If you find this repository useful in your research, please cite:
```
@misc{kandinsky 2.2,
  title         = {kandinsky 2.2},
  author        = {Arseniy Shakhmatov, Anton Razzhigaev, Aleksandr Nikolich, Vladimir Arkhipkin, Igor Pavlov, Andrey Kuznetsov, Denis Dimitrov},
  year          = {2023},
  howpublished  = {},
}
```