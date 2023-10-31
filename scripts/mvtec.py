import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import cv2
from rec_network.main import instantiate_from_config
from rec_network.models.diffusion.ddim import DDIMSampler
from rec_network.data.mvtec import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="./samples/bottle/",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()
    ddim_eta = 0.0

    mvtec_path = './datasets/mvtec/bottle'

    dataset = MVTecDRAEMTestDataset(mvtec_path + "/test/", resize_shape=[256, 256])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)
    print(f"Found {len(dataloader)} inputs.")

    config = OmegaConf.load("../configs/mvtec.yaml")

    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("./logs/2023-01-31T16-58-02_mvtec/checkpoints/last.ckpt")["state_dict"],
                          strict=False)  #TODO: modify the ckpt path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    cnt = 0
    with torch.no_grad():
        with model.ema_scope():
            for i_batch, batch in enumerate(dataloader):

                c_outpath = os.path.join(opt.outdir, 'condition'+str(cnt)+'.jpg')
                outpath = os.path.join(opt.outdir, str(cnt)+'.jpg')
                # print(outpath)
                condition = batch["image"].cpu().numpy().transpose(0,2,3,1)[0]*255
                cv2.imwrite(c_outpath,condition)

                c = batch["image"].to(device)
                c = model.cond_stage_model.encode(c)
                c = c.mode()

                noise = torch.randn_like(c)
                t = torch.randint(400, 500, (c.shape[0],), device=device).long()
                c_noisy = model.q_sample(x_start=c, t=t, noise=noise)

                shape = c.shape[1:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c, # or conditioning=c_noisy
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                sample = x_samples_ddim.cpu().numpy().transpose(0,2,3,1)[0]*255
                cv2.imwrite(outpath, sample)
                cnt+=1
