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
    "LRSCHED_PER_EPOCH_MyDiffusion_batch1_lr-cosine-1e-04_epochs25_snrgamma5.0_channels256,512,768,768"
]

main_folder = "minidataset_generic_finetunings_4atts"

for experiment_name in experiments:
    log_path = f"./{main_folder}/{experiment_name}/validations/log.txt"

    epochs = []
    fid_list = []
    clipscore_list = []
    mtcnnscore_list = []
    attestimator_list = []
    loss_list = []

    with open(log_path, "r") as f:
        lines = f.readlines()  # Lee todas las líneas del archivo y las guarda en una lista
        for l in lines:
            l = l.strip()
            _, name, str_date, str_inference_steps, str_guidance_scale, str_fid, str_clipscore, str_mtcnnscore, str_attestimator, str_loss = l.split('\t')
            e = name.split('epoch')[1]
            if not e.isnumeric(): continue
            epochs.append(int(e))
            fid_list.append(float(str_fid))
            clipscore_list.append(float(str_clipscore))
            mtcnnscore_list.append(float(str_mtcnnscore))
            attestimator_list.append(float(str_attestimator))
            loss_list.append(float(str_loss))

    save_path = f"{main_folder}/plots/{experiment_name}"
    
    if not os.path.exists(save_path):
    	os.makedirs(save_path)

    plot(fid_list, x_values=epochs, y_lim=[220.0,572.0], x_label="Epochs", y_label="FID", title=experiment_name, save_as=f"{save_path}/fid.jpg")
    plot(clipscore_list, x_values=epochs, y_lim=[18.0,27.0], x_label="Epochs", y_label="CLIPScore", title=experiment_name, save_as=f"{save_path}/clipscore.jpg")
    plot(mtcnnscore_list, x_values=epochs, y_lim=None, x_label="Epochs", y_label="MTCNNScore", title=experiment_name, save_as=f"{save_path}/mtcnnscore.jpg")
    plot(attestimator_list, x_values=epochs, y_lim=None, x_label="Epochs", y_label="AttEstimator Acc", title=experiment_name, save_as=f"{save_path}/attestimator.jpg")
    plot(loss_list, x_values=epochs, x_label="Epochs", y_label="Loss", title=experiment_name, save_as=f"{save_path}/loss.jpg")


