# Import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import math
import torch
import torch.nn as nn
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils, models

# Load configuration file
try:
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))
	with open(config_path) as file:
		config = json.load(file)
except Exception as e:
	print(f"Error: The configuration file does not exist or it contains errors. Details: {e}")
	sys.exit(1)


DTYPE, device = eval(config["training_process"]["DTYPE"]), torch.device(config["training_process"]["device"])

# Seeds
torch.manual_seed(config["training_process"]["parameters"]["random_seed"])
np.random.seed(config["training_process"]["parameters"]["random_seed"])

# Generation of data.
X, X_0 = utils.generate_domain(config)
X, X_0 = X.to(DTYPE).to(device), X_0.to(DTYPE).to(device)
print("X_0 (initial) shape: ", X_0.shape)
print("X (internal) shape: ", X.shape)


# Define initial conditions
U_0 = utils.initial_conditions(X_0[:,1:2], config).to(device)
print("U_0 shape: ", U_0.shape)

model = models.GA_PINN(config).to(device)
analytical_space, analytical_solution = utils.load_analytical(config)
model.analytical_space, model.analytical_solution = analytical_space.to(DTYPE).to(device), analytical_solution.to(DTYPE).to(device)


# Define learning rate.
lr = config["training_process"]["parameters"]["learning_rate"]
# Define number of epochs.
epochs = config["training_process"]["parameters"]["epochs"]
# Define the optimizer.
## Define the optimizer
if config["training_process"]["parameters"]["optimizer"] == "RAdam":
	optimizer = torch.optim.RAdam(model.parameters(), lr = lr)
elif config["training_process"]["parameters"]["optimizer"] == "AdamW":
	optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
elif config["training_process"]["parameters"]["optimizer"] == "Adam":
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
elif config["training_process"]["parameters"]["optimizer"] == "SGD":
	optimizer = torch.optim.SGD(model.parameters(), lr = lr, nesterov = True, momentum = 0.9, dampening = 0)
else:
	sys.exit("Optimizer is invalid. Only the following are available at this moment:\n-'RAdam'\n-'AdamW'\n-'Adam'\n-'SGD'")


# 定义学习率调度器
lr_decay_gamma = config["training_process"]["parameters"].get("lr_decay_gamma", 0.999) # 从 config 读取 gamma, 如果没有则默认为 0.999
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_gamma)

# 创建必要的输出目录
output_data_path = os.path.join(os.path.dirname(__file__), config["training_process"]["export"]["path_data"])
output_models_path = os.path.join(os.path.dirname(__file__), config["training_process"]["export"]["path_models"])
output_images_path = os.path.join(os.path.dirname(__file__), config["training_process"]["export"]["path_images"])
checkpoint_path = os.path.join(os.path.dirname(__file__), config["training_process"]["checkpoint"]["path"])

os.makedirs(output_data_path, exist_ok=True)
os.makedirs(output_models_path, exist_ok=True)
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)

# 输出路径为绝对路径
config["training_process"]["export"]["path_data"] = output_data_path + os.sep
config["training_process"]["export"]["path_models"] = output_models_path + os.sep
config["training_process"]["export"]["path_images"] = output_images_path + os.sep
config["training_process"]["checkpoint"]["path"] = checkpoint_path + os.sep

# 检查点恢复逻辑
start_epoch = 0
if config["training_process"]["checkpoint"]["resume_from_checkpoint"]:
    latest_checkpoint = None
    for f in os.listdir(checkpoint_path):
        if f.startswith("checkpoint_") and f.endswith(".pt"):
            epoch_num = int(f.split('_')[1].split('.')[0])
            if latest_checkpoint is None or epoch_num > start_epoch:
                latest_checkpoint = f
                start_epoch = epoch_num
    
    if latest_checkpoint:
        checkpoint_file = os.path.join(checkpoint_path, latest_checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        model.loss_hist = checkpoint['loss_hist']
        model.loss_ic_hist = checkpoint['loss_ic_hist']
        model.loss_ic_rho = checkpoint['loss_ic_rho']
        model.loss_ic_ux = checkpoint['loss_ic_ux']
        model.loss_ic_p = checkpoint['loss_ic_p']
        model.l2_hist = checkpoint['l2_hist']
        model.l2_rho_hist = checkpoint['l2_rho_hist']
        model.l2_ux_hist = checkpoint['l2_ux_hist']
        model.l2_p_hist = checkpoint['l2_p_hist']
        print(f"Resumed from epoch {start_epoch - 1}. Continuing to epoch {epochs}.")
    else:
        print("No checkpoint found to resume from. Starting training from scratch.")


print("Starting optimization...")
pbar = tqdm(range(start_epoch, epochs))
for epoch in pbar:
	model.epoch = epoch
	loss = model.train_step(X.view(-1,2), X_0, U_0, optimizer)

	# 每 lr_scheduler_step_size 次迭代更新一次学习率
	scheduler_step = config["training_process"]["parameters"].get("lr_scheduler_step_size", 1000)
	if epoch % scheduler_step == 0 and epoch > 0:
		scheduler.step()
	
	# 在进度条中显示当前学习率
	current_lr = optimizer.param_groups[0]['lr']
	pbar.set_postfix({'loss_ic' : model.loss_ic_hist[-1], 'loss_total' : model.loss_hist[-1], 'lr': f"{current_lr:.2e}"})

	if epoch % config["training_process"]["export"]["save_each_data"] == 0:
		utils.save_results(model, config)
		torch.save(model.state_dict(), config["training_process"]["export"]["path_models"] + "model_epoch_" + str(model.epoch) + ".pt")
	if epoch % config["training_process"]["export"]["save_each_images"] == 0:
		utils.plot_results(model, config)
	
	# 检查点保存逻辑
	if epoch % config["training_process"]["checkpoint"]["save_each_checkpoint"] == 0 and epoch > 0:
		checkpoint_file = os.path.join(config["training_process"]["checkpoint"]["path"], f"checkpoint_{epoch}.pt")
		torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_hist': model.loss_hist,
            'loss_ic_hist': model.loss_ic_hist,
            'loss_ic_rho': model.loss_ic_rho,
            'loss_ic_ux': model.loss_ic_ux,
            'loss_ic_p': model.loss_ic_p,
            'l2_hist': model.l2_hist,
            'l2_rho_hist': model.l2_rho_hist,
            'l2_ux_hist': model.l2_ux_hist,
            'l2_p_hist': model.l2_p_hist,
        }, checkpoint_file)
		print(f"Checkpoint saved at epoch {epoch} to {checkpoint_file}")
