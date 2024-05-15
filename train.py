from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from multiprocessing import freeze_support
from configurationLoader import returnRepoConfig
repoConfig = returnRepoConfig('control_net_cfg.yaml')
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# Generate a unique identifier for this run
unique_run_id = repoConfig.train.description
checkpoint_path = repoConfig.train.checkpoint_path
os.makedirs(checkpoint_path, exist_ok=True)

# Configs
checkpoint_freq = repoConfig.train.checkpoint_frequency
resume_path = repoConfig.train.model
batch_size = repoConfig.train.batch_size
logger_freq = repoConfig.train.logger_frequency
learning_rate = repoConfig.train.learning_rate
sd_locked = repoConfig.train.sd_locked
only_mid_control = repoConfig.train.only_mid_control
workers = repoConfig.train.workers
gpus = repoConfig.train.gpus
precision = repoConfig.train.precision

# Configure the ModelCheckpoint callbacks
# https://pytorch-lightning.readthedocs.io/en/1.5.10/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html
checkpoint_callback_at_step_freq = ModelCheckpoint(
    dirpath= checkpoint_path,  # Directory to save the model checkpoints
    every_n_train_steps = checkpoint_freq,  # Save a checkpoint every  {checkpoint_freq} training steps
    save_top_k = repoConfig.train.save_top_k,  # Save all checkpoints
    save_weights_only = repoConfig.train.save_weights_only  # Save the full model checkpoint
)
checkpoint_callback_at_end_epoch = ModelCheckpoint(
    dirpath= checkpoint_path,  # Directory to save the model checkpoints
    save_on_train_epoch_end = False,  # Save a checkpoint every end of epoch
    save_top_k = repoConfig.train.save_top_k,  # Save all checkpoints
    save_weights_only = repoConfig.train.save_weights_only  # Save the full model checkpoint
)

def main():
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(repoConfig.train.model_config_path).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=workers, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq, unique_run_id=unique_run_id)
    trainer = pl.Trainer(gpus=gpus, precision=precision, callbacks=[logger, checkpoint_callback_at_step_freq, checkpoint_callback_at_end_epoch])


    # Train!
    trainer.fit(model, dataloader)




if __name__ == '__main__':
    # Windows requires the freeze_support() call here if you're using multiprocessing
    
    freeze_support()
    main()