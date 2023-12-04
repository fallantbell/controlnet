from share import *
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
from inf_zero_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning import loggers as pl_loggers


# Configs
resume_path = 'save_log/se3_range50/logs/default/version_0/checkpoints/epoch=77-step=19733.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
max_time = 50
sd_locked = True
only_mid_control = False

project_name = "se3_range50_v2"
output_dir = f"save_log/{project_name}"
os.makedirs(output_dir, exist_ok=True)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21_se3.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
train_dataset = MyDataset(max_time=max_time,mode="train")
train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
val_dataset = MyDataset(max_time=max_time,mode="validation")
val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False)

logger = ImageLogger(batch_frequency=logger_freq, outputdir=output_dir)
tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{output_dir}/logs/")
trainer = pl.Trainer(gpus=3, precision=32,logger = tb_logger, callbacks=[logger])


if __name__ == "__main__":
    # Train!
    trainer.fit(model, train_dataloader,val_dataloader)
