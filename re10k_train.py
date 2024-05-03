from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from re10k_dataset import Re10k_dataset
from pytorch_lightning.loggers import WandbLogger
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


#* Configs
resume_path = './models/control_epipolar_init.ckpt'
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
accumulate = 4
gpu_num = 2

#* Dataset config
data_root = "../dataset"
mode = "train"
max_interval = 1

#* wandb info
project = "novel_view_synthesis_bycontrol"
name = f"epipolar"
save_path = f"saved/{name}_b{batch_size*accumulate*gpu_num}_lr{learning_rate}"

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21_epipolar.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = Re10k_dataset(data_root,mode,max_interval)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
wandb_logger = WandbLogger(
                project=project,
                name = f"{name}_b{batch_size*accumulate*gpu_num}_lr{learning_rate}"            
            )
trainer = pl.Trainer(gpus=gpu_num, precision=32, callbacks=[logger],
                     logger=wandb_logger,
                     accumulate_grad_batches=accumulate,
                     default_root_dir = save_path
                     )



if __name__ == "__main__":
    # Train!
    trainer.fit(model, dataloader)
