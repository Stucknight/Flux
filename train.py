import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SimpleDataset
from sampling import get_noise, get_schedule, prepare
from util import load_ae, load_clip, load_t5, load_flow_model


def train(train_steps, optimizer, lr_scheduler, device, num_steps=1000):
    step = 0

    model = load_flow_model(device=device)
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    ae = load_ae(device=device)

    train_dataset = SimpleDataset(root="./", mode='train')
    val_dataset = SimpleDataset(root="./", mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=train_steps, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=train_steps, shuffle=False, num_workers=4)

    while step < train_steps:
        batch = next(iter(train_dataloader))
        t5.eval()
        clip.eval()
        ae.eval()
        model.train()

        image = batch["image"].to(device)
        prompt = batch["prompt"]
        latents = ae.encode(image)

        b, c, h, w = image.shape

        x = get_noise(
            1,
            h,
            w,
            device=device,
            dtype=torch.bfloat16,
            seed=69,
        )

        #CODE IS WRONG, CHANGE IT
        inp = prepare(t5, clip, x, prompt=prompt)
        img, img_ids, txt, txt_ids, vec = inp["img"], inp["img_ids"], inp["txt"], inp["txt_ids"], inp["vec"]

        indices = torch.randint(1, num_steps, (b,))
        schedule = get_schedule(num_steps, c)
        t_curr = schedule[indices]
        t_prev = schedule[indices - 1]

        noisy_latent = (t_curr - t_prev) * img + (1.0 - (t_curr - t_prev)) * latents
        t_vec = torch.full((b,), t_curr, device=device)
        model_pred = model(noisy_latent, img_ids, txt, txt_ids, t_vec)

        denoised_pred = img + (t_prev - t_curr) * model_pred

        target = latents

        loss = nn.MSELoss(denoised_pred.float(), target.float(), reduction="mean")

        loss.backwards()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        step += 1
