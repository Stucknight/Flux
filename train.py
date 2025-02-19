import accelerate
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from safetensors.torch import save_file
from PIL import Image
from dataset import loader
from datasets import load_dataset
from sampling import prepare, get_schedule, denoise, get_noise, unpack
from util import load_clip, load_t5, load_flow_model, load_ae

def main():
    dtype = torch.bfloat16
    deepspeed_plugin = accelerate.DeepSpeedPlugin(zero_stage=2)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=2, mixed_precision="bf16", deepspeed_plugin=deepspeed_plugin)
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
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

    optimizer = torch.optim.AdamW(
        flux.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    )

    flux, optimizer, train_dataloader = accelerator.prepare([flux, optimizer, train_dataloader])

    epochs = 10
    p_uncond = 0.1

    @torch.no_grad()
    def inference_prompts(prompts, model, t5, clip, vae,
                          height=512, width=512, device: str | torch.device = "cuda", dtype=torch.bfloat16, seed=69):
        images = []
        for i, prompt in enumerate(prompts):
            noise = get_noise(1, height, width, device, dtype, seed)
            input = prepare(t5, clip, noise, prompt)

            num_steps = 50
            timesteps = get_schedule(num_steps, input['img'].shape[1], shift=True)
            denoised = denoise(model, **input, timesteps=timesteps, guidance=3.5)
            denoised = unpack(denoised.float(), height, width)
            image = vae.decode(denoised)
            image = image.clamp(-1, 1)
            image = rearrange(image[0], 'c h w -> h w c')
            images.append(image)
        return images

    prompts = ["acoustic image of light foam"]
    with accelerator.autocast():
        for epoch in range(epochs):
            flux.train()
            train_loss = 0.0
            for step, batch in enumerate(tqdm(train_dataloader)):
                img, prompt = batch
                img = img.to(device, dtype=dtype)

                if torch.rand(1).item() < p_uncond:
                    prompt = ""

                with torch.no_grad():
                    x_1 = vae.encode(img)
                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompt)
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = img.shape[0]
                t = torch.sigmoid(torch.randn((bs,), device=device))
                x_0 = torch.randn_like(x_1).to(device)
                x_t = (1 - t) * x_1 + t * x_0


                model_pred = flux(
                    img=x_t.to(dtype),
                    img_ids=inp["img_ids"].to(dtype),
                    txt=inp["txt"].to(dtype),
                    txt_ids=inp["txt_ids"].to(dtype),
                    y=inp["vec"].to(dtype),
                    timesteps=t.to(dtype),
                )

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(1)).mean()
                train_loss += avg_loss.item() / 2
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item()
            accelerator.print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_dataloader):.6f}")

            if accelerator.is_main_process:
                flux.eval()
                images = inference_prompts(prompts, flux, t5, clip, vae,
                                           height=512, width=512,
                                           dtype=dtype)
                for i, image in enumerate(images):
                    image = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
                    image.save(f"epoch_{epoch}_{i}.png", quality=95, subsampling=0)

            unwrapped_model_state = accelerator.unwrap_model(flux).state_dict()
            save_file(unwrapped_model_state, f"models\\flux_{epoch}.safetensors")

        accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()
