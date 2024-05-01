from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from multiprocessing import freeze_support

from pytorch_lightning.callbacks import ModelCheckpoint
import os

# Generate a unique identifier for this run
unique_run_id = "0004 (removed 0 annotations)"
checkpoint_path = f'./model_checkpoints/run_{unique_run_id}/'
os.makedirs(checkpoint_path, exist_ok=True)

# Configs
checkpoint_freq = 300
resume_path = r"E:\thesis\repos\ControlNet\model_checkpoints\run_0003 (less frequent logging)\epoch=4-step=7799.ckpt"
batch_size = 4
logger_freq = 1000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# Configure the ModelCheckpoint callbacks
# https://pytorch-lightning.readthedocs.io/en/1.5.10/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html
checkpoint_callback_at_step_freq = ModelCheckpoint(
    dirpath= checkpoint_path,  # Directory to save the model checkpoints
    every_n_train_steps = checkpoint_freq,  # Save a checkpoint every  {checkpoint_freq} training steps
    save_top_k = -1,  # Save all checkpoints
    save_weights_only=False  # Save the full model checkpoint
)
checkpoint_callback_at_end_epoch = ModelCheckpoint(
    dirpath= checkpoint_path,  # Directory to save the model checkpoints
    save_on_train_epoch_end = True,  # Save a checkpoint every end of epoch
    save_top_k = -1,  # Save all checkpoints
    save_weights_only=False  # Save the full model checkpoint
)

def main():
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=12, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq, unique_run_id=unique_run_id)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback_at_step_freq, checkpoint_callback_at_end_epoch])


    # Train!
    trainer.fit(model, dataloader)




if __name__ == '__main__':
    # Windows requires the freeze_support() call here if you're using multiprocessing
    
    freeze_support()
    main()