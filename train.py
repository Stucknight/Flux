from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from sampling import denoise, get_noise, get_schedule, prepare, unpack
from util import load_ae, load_clip, load_t5


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def train_epoch(model, dataloader, device, lr=1e-4):
    model.train()
    height = 16 * (1360 // 16)
    width = 16 * (768 // 16)
    num_steps = 28
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    ae = load_ae('flux-dev', device=device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for input_text, target_img in dataloader:
        rng = torch.Generator(device="cpu")
        opts = SamplingOptions(
            prompt=input_text,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=3.5,
            seed=None,
        )

        optimizer.zero_grad()

        if opts.seed is None:
            opts.seed = rng.seed()

        x = get_noise(
            1, opts.height, opts.width,
            device=device, dtype=torch.bfloat16,
            seed=opts.seed
        )

        opts.seed = None
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=True)

        x_denoised = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)
        x_denoised = unpack(x_denoised.float(), opts.height, opts.width)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            x_denoised = ae.decode(x_denoised)

        loss = criterion(x_denoised, target_img)
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
