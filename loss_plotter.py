import os
import numpy as np
import matplotlib.pyplot as plt

def plot(lst, x_values=None, y_lim=None, x_label="", y_label="", title="", save_as="plot.jpg"):
    fig, ax = plt.subplots()
    ticks = range(len(lst))
    ax.plot(ticks, lst)
    if x_values is not None: ax.set_xticks(ticks, x_values)
    if y_lim is not None: ax.set_ylim(y_lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.savefig(save_as)

experiments = [
    ("../experimentos2/generic_finetunings_old2_4atts/MyDiffusion_batch1_lr1e-05(cosine)_epochs3_snrgamma5.0/validations","UNet_big"),
    ("../experimentos2/generic_finetunings_old2_4atts/MyDiffusionMedium_batch2_lr1e-04(cosine)_epochs3/validations","UNet_min"),
    ("../experimentos2/generic_finetunings_old2_4atts/MyDiffusionSmall_batch3_lr1e-05(cosine)_epochs3_snrgamma5.0/validations","UNet_small"),

    ("../experimentos2/generic_finetunings_old1/StableDiffusionLoRAv14_batch64_lr1e-04(cosine)_epochs20_notsaved/validations","LORA_SD_v14"),
    ("../experimentos2/generic_finetunings_old1/StableDiffusionLoRAv20_batch32_lr1e-04(cosine)_epochs20_notsaved/validations","LORA_SD_v20"),

    ("./generic_finetunings_4atts/StableDiffusionLoRA_batch512_lr-cosine-1e-04_epochs20_snrgamma5.0/validations","LORA_SD_Nanov21_bs512"),
    ("./generic_finetunings_4atts/StableDiffusionLoRA_batch1*1_lr-cosine-1e-04_epochs20_snrgamma5.0/validations","LORA_SD_Nanov21_bs1"),

    ("../RUNPOD/generic_finetunings_4atts/MyDiffusion_batch8*1_lr-cosine-1e-04_epochs10_snrgamma5.0_channels256,512,768,768/validations","UNet_RunPod"),

    ("../RUNPOD/ddpm-celeba-uncond/ddpm-celeba","UncondDDPM"),
    ("../RUNPOD/generic_finetunings_4atts/DDPM_batch256*1_lr-cosine-1e-04_epochs10_snrgamma5.0/validations","CondDDPM"),
]

output_folder = "loss_plots"

for folder, alias in experiments:
    log_path = f"{folder}/log.txt"

    epochs = []
    loss_list = []

    with open(log_path, "r") as f:
        lines = f.readlines()  # Lee todas las líneas del archivo y las guarda en una lista
        for l in lines:
            l = l.strip()
            values = l.split('\t')
            name = values[1]
            str_loss = values[-1]

            # Avoid epochs None,-1,etc
            name_split = name.split('epoch')
            if len(name_split)>1:
                e = name.split('epoch')[1]
            else:
                e = name
                
            if not e.isnumeric(): continue
            epochs.append(int(e))
            loss_list.append(float(str_loss))

    save_path = f"{output_folder}/{alias}"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot(loss_list, x_values=epochs, x_label="Epochs", y_label="Loss", title="", save_as=f"{save_path}/loss.jpg")


