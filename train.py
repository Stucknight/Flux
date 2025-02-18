import accelerate
import torch
import torch.nn as nn
from einops import rearrange

from dataset import loader
from datasets import load_dataset
from sampling import prepare
from util import load_clip, load_t5, load_flow_model, load_ae

dtype = torch.bfloat16
deepspeed_plugin = accelerate.DeepSpeedPlugin(zero_stage=2)
accelerator = accelerate.Accelerator(gradient_accumulation_steps=2, mixed_precision="bf16", deepspeed_plugin=deepspeed_plugin)
device = accelerator.device

data = load_dataset("dream-textures/textures-normal-1k")
train_dataloader = loader(train_batch_size=1, num_workers=2, data=data)

clip = load_clip(device=device)
t5 = load_t5(device=device)
flux = load_flow_model()
vae = load_ae()

vae.requires_grad_(False)
t5.requires_grad_(False)
clip.requires_grad_(False)

flux.to(dtype=dtype, device=device)
vae.to(dtype=dtype, device=device)

flux.train()

optimizer = torch.optim.AdamW(flux.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
flux, optimizer, train_dataloader = accelerator.prepare([flux, optimizer, train_dataloader])

epochs = 10

for epoch in range(epochs):
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        img, prompts = batch
        img = img.to(device, dtype=dtype)

        with torch.no_grad():
            x_1 = vae.encode(img)
            inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
            x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        bs = img.shape[0]
        t = torch.sigmoid(torch.randn((bs,), device=device))
        x_0 = torch.randn_like(x_1).to(device)
        x_t = (1 - t) * x_1 + t * x_0

        with accelerator.autocast():
            model_pred = flux(img=x_t.to(dtype), img_ids=inp['img_ids'].to(dtype), txt=inp['txt'].to(dtype), txt_ids=inp['txt_ids'].to(dtype), y=inp['vec'].to(dtype), timesteps=t.to(dtype))
            loss = nn.MSELoss(model_pred.float(), (x_0 - x_1).float(), reduction='mean')

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    accelerator.print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_dataloader):.6f}")
accelerator.wait_for_everyone()