import accelerate
import torch
import pickle
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from safetensors.torch import save_file
from PIL import Image
from dataset import loader, ImageDataset
from datasets import load_dataset
from sampling import prepare, get_schedule, denoise, get_noise, unpack
from util import load_clip, load_t5, load_flow_model, load_ae

def main():
    dtype = torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_dataset("dream-textures/textures-normal-1k")
    dataset = ImageDataset(data)

    preprocessed_data = []
    clip = load_clip(device=device)
    t5 = load_t5(device=device)
    vae = load_ae()

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)

    vae.to(dtype=dtype, device=device)

    for step, batch in enumerate(tqdm(dataset)):
        img, prompt = batch
        img = img.to(device, dtype=dtype)

        with torch.no_grad():
            x_1 = vae.encode(img)
            inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompt)
            x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            preprocessed_data.append({'x_1': x_1, 'inp': inp})

    with open("preprocessed_data.pkl", "wb") as f:
        pickle.dump(preprocessed_data, f)
    print("Preprocessed data saved successfully.")

if __name__ == '__main__':
    main()
