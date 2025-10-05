# Import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import pdb
from functools import reduce
from functorch import make_functional, grad, vmap
sys.path.insert(1, "../../")
import utils



class GA_PINN(nn.Module):
	def __init__(self, config):
		"""Class for the model.
		
		Parameters
		----------
		config : dictionary
			Configuration file for the training.
		"""
		
		super(GA_PINN, self).__init__()
		self.config = config
		
		self.tmin, self.tmax = self.config["physical"]["parameters"]["temporal_range"]
		self.xmin, self.xmax = self.config["physical"]["parameters"]["spatial_range"]
		self.N_t, self.N_x = eval(self.config["physical"]["parameters"]["N_t"]), eval(self.config["physical"]["parameters"]["N_x"])
		self.size_hidden = self.config["neural"]["general_parameters"]["number_neurons"]
		self.num_hidden_layers = self.config["neural"]["general_parameters"]["number_hidden"]
		self.gamma = eval(self.config["physical"]["parameters"]["adiabatic_constant"])
		self.DTYPE, self.device = eval(self.config["training_process"]["DTYPE"]), torch.device(self.config["training_process"]["device"])
		self.num_inputs, self.num_outputs = 2, 3
		
		
		# Define the DNN.
		self.dense_layers = []
		self.dense_layers.append(nn.Linear(self.num_inputs, self.size_hidden).to(self.device))
		for i in range(0, self.num_hidden_layers):
			layer = nn.Linear(self.size_hidden, self.size_hidden).to(self.device)
			self.dense_layers.append(layer)
		layer_final = nn.Linear(self.size_hidden, self.num_outputs).to(self.device)
		self.dense_layers.append(layer_final)
		## Initialize the weights of the layers.
		for i in range(len(self.dense_layers)):
			torch.nn.init.xavier_uniform_(self.dense_layers[i].weight, gain = 1.0)
		## Now, register the parameters as trainable variables
		self.params_hidden = nn.ModuleList(self.dense_layers)
		
		# [新增] 将模型转换为 functional 形式，以便于计算 Jacobian
		self.func_model, self.func_params = make_functional(self)

		# [新增] 用于缓存 NTK 权重的属性
		self.ntk_weights = None
		self.ntk_update_freq = self.config["training_process"]["parameters"].get("ntk_update_freq", 100) # 定义更新频率，这是一个新的超参数！
		
		# Related with the L2 computation.
		self.l2_hist, self.l2_rho_hist, self.l2_ux_hist, self.l2_p_hist = [], [], [], []
		# Define lists for the physical losses.
		self.loss_hist, self.loss_ic_hist = [], []
		self.loss_ic_rho, self.loss_ic_ux, self.loss_ic_p = [], [], []
		# Define activation functions.
		self.act_rho = eval(self.config["neural"]["activation_functions"]["output"][0])
		self.act_ux = eval(self.config["neural"]["activation_functions"]["output"][1])
		self.act_p = eval(self.config["neural"]["activation_functions"]["output"][2])
		self.act_hidden = eval(self.config["neural"]["activation_functions"]["hidden_layers"])

	# 计算 NTK 迹的辅助函数
	def compute_ntk_trace(self, x):
		def model_output(params, x_single):
			# forward pass 够了
			return self.func_model(params, x_single.unsqueeze(0)).squeeze(0)

		# 计算单个 Jacobian
		jac1 = vmap(grad(lambda params, x: model_output(params, x)[0]))(self.func_params, x)
		jac2 = vmap(grad(lambda params, x: model_output(params, x)[1]))(self.func_params, x)
		jac3 = vmap(grad(lambda params, x: model_output(params, x)[2]))(self.func_params, x)
		
		# 计算每个输出的 NTK 迹，取平均
		trace1 = sum(j.pow(2).sum() for j in jac1) / len(x)
		trace2 = sum(j.pow(2).sum() for j in jac2) / len(x)
		trace3 = sum(j.pow(2).sum() for j in jac3) / len(x)

		return trace1, trace2, trace3

	def forward(self, X):
		# Training bucle.
		for i in range( len(self.dense_layers) - 1 ):
			X = self.act_hidden(self.dense_layers[i](X))
		X = self.dense_layers[-1](X)
		
		# Extract each primitive variable separately.
		rho = self.act_rho(X[:,0:1])
		ux = self.act_ux(X[:,1:2])
		p = self.act_p(X[:,2:3])
		return torch.cat((rho, ux, p), dim = 1)
		
	def compute_loss(self, X, X_0, U_0):
		t, x = X[:,0:1], X[:,1:2]
		prediction = self(torch.cat((t, x), dim = 1 ))
		rho, ux, p = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3]
		ux = torch.clamp(ux, max = 0.9999, min = -0.9999)
		e = p/(rho * (self.gamma - 1.0) )
		
		W = 1 / torch.sqrt(1 - ux**2)
		
		D = rho * W
		Mx = ux * (rho + p * self.gamma / (self.gamma - 1.0) ) * (W ** 2)
		E = (rho + p * self.gamma/(self.gamma - 1.0)) * (W ** 2) - p
		
		F1 = D * ux
		F2x = Mx * ux + p
		F3 = (E + p) * ux
		
		D_t = torch.autograd.grad(D, t, grad_outputs = torch.ones_like(D), create_graph = True)[0]
		Mx_t = torch.autograd.grad(Mx, t, grad_outputs = torch.ones_like(Mx), create_graph = True)[0]
		E_t = torch.autograd.grad(E, t, grad_outputs = torch.ones_like(E), create_graph = True)[0]
		
		F1_x = torch.autograd.grad(F1, x, grad_outputs = torch.ones_like(F1), create_graph = True)[0]
		F2x_x = torch.autograd.grad(F2x, x, grad_outputs = torch.ones_like(F2x), create_graph = True)[0]
		F3_x = torch.autograd.grad(F3, x, grad_outputs = torch.ones_like(F3), create_graph = True)[0]

		rho_x = torch.autograd.grad(rho, x, grad_outputs = torch.ones_like(rho), create_graph = True)[0]
		ux_x = torch.autograd.grad(ux, x, grad_outputs = torch.ones_like(ux), create_graph = True)[0]
		p_x = torch.autograd.grad(p, x, grad_outputs = torch.ones_like(p), create_graph = True)[0]
		
		
		self.alpha_rho, self.alpha_ux, self.alpha_p = self.config["neural"]["loss_function_parameters"]["alpha_set"]
		self.beta_rho, self.beta_ux, self.beta_p = self.config["neural"]["loss_function_parameters"]["beta_set"]
		Lambda = ( 1 / (1 + ( self.alpha_rho * torch.abs(rho_x)**self.beta_rho + self.alpha_ux * torch.abs(ux_x)**self.beta_ux + self.alpha_p * torch.abs(p_x)**self.beta_p) ) ).view(self.N_t, self.N_x, 1) # Works fine
		self.Lambda = Lambda
		# Compute Losses
		# ================================================================================================================================
		## Losses of the equations conforming the system
		### These present shape of (N_t, N_x, 1)
		L_t_1 = torch.square(D_t + F1_x).view(self.N_t, self.N_x, 1)
		L_t_2 = torch.square(Mx_t + F2x_x).view(self.N_t, self.N_x, 1)
		L_t_3 = torch.square(E_t + F3_x).view(self.N_t, self.N_x, 1)
		
		## Total physical loss
		L_phys_tensor = torch.mean( Lambda * (L_t_1 + L_t_2 + L_t_3), dim = 1 )
		# ================================================================================================================================
		
		# 计算原始损失项 (无权重) 
		prediction_tmin = self(X_0)
		
		# 计算原始的 IC 损失
		L_IC_rho_raw = torch.square(U_0[:,0:1] - prediction_tmin[:,0:1]).mean()
		L_IC_ux_raw = torch.square(U_0[:,1:2] - prediction_tmin[:,1:2]).mean()
		L_IC_p_raw = torch.square(U_0[:,2:3] - prediction_tmin[:,2:3]).mean()
		
		# 计算原始的物理损失
		L_phys_raw = L_phys_tensor.mean()
		
		# 带缓存的 NTK 权重计算
		if self.epoch % self.ntk_update_freq == 0 or self.ntk_weights is None:
			with torch.no_grad():
				# 计算初始点 (X_0) 上每个输出的 NTK 迹
				ntk_trace_rho_ic, ntk_trace_ux_ic, ntk_trace_p_ic = self.compute_ntk_trace(X_0)
				
				# 计算内部点 (X) 上每个输出的 NTK 迹
				# 对于物理损失，取其涉及的所有变量 NTK 迹的平均值
				ntk_traces_phys = self.compute_ntk_trace(X.view(-1, 2))
				ntk_trace_phys_avg = sum(ntk_traces_phys) / len(ntk_traces_phys)

				# 计算所有迹的平均值，作为平衡的基准
				all_traces = [ntk_trace_rho_ic, ntk_trace_ux_ic, ntk_trace_p_ic, ntk_trace_phys_avg]
				avg_trace = sum(all_traces) / len(all_traces)

				# 计算权重：权重与 NTK 迹成反比
				lambda_ic_rho = avg_trace / ntk_trace_rho_ic
				lambda_ic_ux = avg_trace / ntk_trace_ux_ic
				lambda_ic_p = avg_trace / ntk_trace_p_ic
				lambda_phys = avg_trace / ntk_trace_phys_avg

				# 将计算出的权重缓存起来
				self.ntk_weights = {
					"lambda_ic_rho": lambda_ic_rho,
					"lambda_ic_ux": lambda_ic_ux,
					"lambda_ic_p": lambda_ic_p,
					"lambda_phys": lambda_phys
				}
		
		# 从缓存中获取权重
		weights = self.ntk_weights
		
		# 使用动态权重计算总损失
		w_R_final = self.config["neural"]["loss_function_parameters"]["w_R"]
		annealing_epochs = self.config["neural"]["loss_function_parameters"]["annealing_epochs"]
		w_R = w_R_final * min(1.0, self.epoch / annealing_epochs)

		# 加权各项损失
		L_IC = (weights["lambda_ic_rho"] * L_IC_rho_raw + 
				weights["lambda_ic_ux"] * L_IC_ux_raw + 
				weights["lambda_ic_p"] * L_IC_p_raw)
				
		total_loss = L_IC + w_R * weights["lambda_phys"] * L_phys_raw

		### Take advantage and save initial losses into lists
		self.loss_ic_hist.append(L_IC.item())
		self.loss_ic_rho.append(L_IC_rho_raw.item()) # 保存原始损失，方便Test
		self.loss_ic_ux.append(L_IC_ux_raw.item()) 
		self.loss_ic_p.append(L_IC_p_raw.item()) 
		
		## Compute and save l2 errors
		self.compute_l2()
		
		return total_loss
	def train_step(self, X, X_0, U_0, optimizer):
		optimizer.zero_grad(set_to_none = True)
		# Compute the loss
		loss = self.compute_loss(X, X_0, U_0)
		loss.backward(retain_graph = False)
		
		# 启用梯度裁剪
		clip_value = self.config["training_process"]["parameters"].get("gradient_clip_value", 1.0)
		torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
		
		optimizer.step()
		# Save data
		self.loss_hist.append(loss.item())
		return loss
