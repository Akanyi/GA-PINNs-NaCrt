# 导入模块
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

# 从 functorch 导入兼容性函数，并从 torch.func 导入新的函数式 API
from functorch import make_functional_with_buffers
from torch.func import grad, vmap

sys.path.insert(1, "../../")
import utils

# 傅里叶特征映射层
class FourierFeatureMapping(nn.Module):
    """
    实现傅里叶特征映射。
    将 R^input_dims 的输入坐标映射到 R^(2*mapping_size) 的特征空间。
    """
    def __init__(self, input_dims, mapping_size, scale=10.0):
        super().__init__()
        self.input_dims = input_dims
        self.mapping_size = mapping_size
        self.scale = scale
        
        # 创建一个固定的、不可训练的投影矩阵 B
        # 使用 register_buffer 确保 B 矩阵是模型状态的一部分（例如，会随模型一起移动到 GPU），
        # 但它不会被优化器视为可训练的参数。
        B = torch.randn(input_dims, mapping_size) * scale
        self.register_buffer('B_matrix', B)

    def forward(self, x):
        # x 的形状: (batch_size, input_dims)
        # self.B_matrix 的形状: (input_dims, mapping_size)
        # projected_x 的形状: (batch_size, mapping_size)
        projected_x = (2. * torch.pi * x) @ self.B_matrix
        
        # 计算 sin 和 cos，并将它们拼接起来
        # 返回张量的形状: (batch_size, 2 * mapping_size)
        return torch.cat([torch.sin(projected_x), torch.cos(projected_x)], dim=-1)


class GA_PINN(nn.Module):
	def __init__(self, config):
		"""模型的初始化函数。
		
		Parameters
		----------
		config : dictionary
			包含所有配置信息的字典。
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
		
		# 根据配置决定是否使用傅里叶特征
		self.use_fourier = self.config["neural"]["fourier_features"]["use_fourier_features"]
		if self.use_fourier:
			mapping_size = self.config["neural"]["fourier_features"]["mapping_size"]
			scale = self.config["neural"]["fourier_features"]["scale"]
			self.fourier_mapper = FourierFeatureMapping(self.num_inputs, mapping_size, scale)
			# 网络第一层的输入维度必须与傅里叶特征的输出维度相匹配
			first_layer_input_dims = 2 * mapping_size
		else:
			self.fourier_mapper = None
			first_layer_input_dims = self.num_inputs
		
		# 定义全连接网络层
		self.dense_layers = []
		self.dense_layers.append(nn.Linear(first_layer_input_dims, self.size_hidden))
		for i in range(0, self.num_hidden_layers):
			layer = nn.Linear(self.size_hidden, self.size_hidden)
			self.dense_layers.append(layer)
		layer_final = nn.Linear(self.size_hidden, self.num_outputs)
		self.dense_layers.append(layer_final)
		# 初始化网络权重
		for i in range(len(self.dense_layers)):
			torch.nn.init.xavier_uniform_(self.dense_layers[i].weight, gain = 1.0)
		# 将网络层注册为模块列表，使其参数可被训练
		self.params_hidden = nn.ModuleList(self.dense_layers)
		
		# 将 functional 模型的属性初始化为 None，以延迟转换
		# 确保模型在转换前已经被移动到正确的计算设备上
		self.func_model, self.func_params, self.func_buffers = None, None, None

		# 用于缓存 NTK 权重的属性
		self.ntk_weights = None
		self.ntk_update_freq = self.config["training_process"]["parameters"].get("ntk_update_freq", 100)
		
		# 用于记录 L2 相对误差的列表
		self.l2_hist, self.l2_rho_hist, self.l2_ux_hist, self.l2_p_hist = [], [], [], []
		# 用于记录损失函数值的列表
		self.loss_hist, self.loss_ic_hist = [], []
		self.loss_ic_rho, self.loss_ic_ux, self.loss_ic_p = [], [], []
		# 定义激活函数
		self.act_rho = eval(self.config["neural"]["activation_functions"]["output"][0])
		self.act_ux = eval(self.config["neural"]["activation_functions"]["output"][1])
		self.act_p = eval(self.config["neural"]["activation_functions"]["output"][2])
		self.act_hidden = eval(self.config["neural"]["activation_functions"]["hidden_layers"])

	def compute_ntk_trace(self, x):
		"""计算神经网络切向核 (NTK) 的迹。"""
		
		# 在首次需要时，才将模型转换为 functional 形式
		if self.func_model is None:
			self.func_model, self.func_params, self.func_buffers = make_functional_with_buffers(self)

		def model_output(params, x_single):
			"""辅助函数，用于单一样本的前向传播。"""
			return self.func_model(params, self.func_buffers, x_single.unsqueeze(0)).squeeze(0)

		# 定义针对每个输出的梯度函数
		grad_fn_rho = grad(lambda params, x_sample: model_output(params, x_sample)[0])
		grad_fn_ux = grad(lambda params, x_sample: model_output(params, x_sample)[1])
		grad_fn_p = grad(lambda params, x_sample: model_output(params, x_sample)[2])

		# 初始化用于累加迹的变量
		sum_trace1, sum_trace2, sum_trace3 = 0.0, 0.0, 0.0
		
		# 定义块大小，这是一个可以根据显存调整的超参数
		chunk_size = 1024 
		
		# 使用 torch.split 对输入 x 进行分块，并迭代处理每个块
		for x_chunk in torch.split(x, chunk_size):
			# 对当前块使用 vmap 进行批处理计算雅可比矩阵
			jac1_chunk = vmap(grad_fn_rho, in_dims=(None, 0))(self.func_params, x_chunk)
			jac2_chunk = vmap(grad_fn_ux, in_dims=(None, 0))(self.func_params, x_chunk)
			jac3_chunk = vmap(grad_fn_p, in_dims=(None, 0))(self.func_params, x_chunk)
			
			# 计算当前块的迹的平方和，并累加
			sum_trace1 += sum(j.pow(2).sum() for j in jac1_chunk)
			sum_trace2 += sum(j.pow(2).sum() for j in jac2_chunk)
			sum_trace3 += sum(j.pow(2).sum() for j in jac3_chunk)

		# 在所有块处理完毕后，计算总的平均迹
		trace1 = sum_trace1 / len(x)
		trace2 = sum_trace2 / len(x)
		trace3 = sum_trace3 / len(x)

		return trace1, trace2, trace3

	def forward(self, X):
		"""模型的前向传播。"""
		if self.use_fourier:
			X = self.fourier_mapper(X)
		
		for i in range( len(self.dense_layers) - 1 ):
			X = self.act_hidden(self.dense_layers[i](X))
		X = self.dense_layers[-1](X)
		
		# 对不同的输出变量应用不同的激活函数
		rho = self.act_rho(X[:,0:1])
		ux = self.act_ux(X[:,1:2])
		p = self.act_p(X[:,2:3])
		return torch.cat((rho, ux, p), dim = 1)
		
	def compute_loss(self, X, X_0, U_0):
		"""计算总损失函数。"""
		t, x = X[:,0:1], X[:,1:2]
		prediction = self(torch.cat((t, x), dim = 1 ))
		rho, ux, p = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3]
		ux = torch.clamp(ux, max = 0.9999, min = -0.9999) # 限制速度范围以保证物理合理性
		
		# 根据原始变量计算守恒量
		W = 1 / torch.sqrt(1 - ux**2)
		D = rho * W
		Mx = ux * (rho + p * self.gamma / (self.gamma - 1.0) ) * (W ** 2)
		E = (rho + p * self.gamma/(self.gamma - 1.0)) * (W ** 2) - p
		
		# 计算通量
		F1 = D * ux
		F2x = Mx * ux + p
		F3 = (E + p) * ux
		
		# 使用自动微分计算各项偏导数
		D_t = torch.autograd.grad(D, t, grad_outputs = torch.ones_like(D), create_graph = True)[0]
		Mx_t = torch.autograd.grad(Mx, t, grad_outputs = torch.ones_like(Mx), create_graph = True)[0]
		E_t = torch.autograd.grad(E, t, grad_outputs = torch.ones_like(E), create_graph = True)[0]
		
		F1_x = torch.autograd.grad(F1, x, grad_outputs = torch.ones_like(F1), create_graph = True)[0]
		F2x_x = torch.autograd.grad(F2x, x, grad_outputs = torch.ones_like(F2x), create_graph = True)[0]
		F3_x = torch.autograd.grad(F3, x, grad_outputs = torch.ones_like(F3), create_graph = True)[0]

		rho_x = torch.autograd.grad(rho, x, grad_outputs = torch.ones_like(rho), create_graph = True)[0]
		ux_x = torch.autograd.grad(ux, x, grad_outputs = torch.ones_like(ux), create_graph = True)[0]
		p_x = torch.autograd.grad(p, x, grad_outputs = torch.ones_like(p), create_graph = True)[0]
		
		# 计算自适应权重 Lambda
		self.alpha_rho, self.alpha_ux, self.alpha_p = self.config["neural"]["loss_function_parameters"]["alpha_set"]
		self.beta_rho, self.beta_ux, self.beta_p = self.config["neural"]["loss_function_parameters"]["beta_set"]
		Lambda = ( 1 / (1 + ( self.alpha_rho * torch.abs(rho_x)**self.beta_rho + self.alpha_ux * torch.abs(ux_x)**self.beta_ux + self.alpha_p * torch.abs(p_x)**self.beta_p) ) ).view(self.N_t, self.N_x, 1)
		
		# 计算物理方程的残差损失
		L_t_1 = torch.square(D_t + F1_x).view(self.N_t, self.N_x, 1)
		L_t_2 = torch.square(Mx_t + F2x_x).view(self.N_t, self.N_x, 1)
		L_t_3 = torch.square(E_t + F3_x).view(self.N_t, self.N_x, 1)
		
		L_phys_tensor = torch.mean( Lambda * (L_t_1 + L_t_2 + L_t_3), dim = 1 )
		L_phys_raw = L_phys_tensor.mean()
		
		# 计算初始条件的损失
		prediction_tmin = self(X_0)
		L_IC_rho_raw = torch.square(U_0[:,0:1] - prediction_tmin[:,0:1]).mean()
		L_IC_ux_raw = torch.square(U_0[:,1:2] - prediction_tmin[:,1:2]).mean()
		L_IC_p_raw = torch.square(U_0[:,2:3] - prediction_tmin[:,2:3]).mean()
		
		# 基于 NTK 动态计算各项损失的权重
		if self.epoch % self.ntk_update_freq == 0 or self.ntk_weights is None:
			with torch.no_grad():
				ntk_trace_rho_ic, ntk_trace_ux_ic, ntk_trace_p_ic = self.compute_ntk_trace(X_0)
				ntk_traces_phys = self.compute_ntk_trace(X.view(-1, 2))
				ntk_trace_phys_avg = sum(ntk_traces_phys) / len(ntk_traces_phys)

				all_traces = [ntk_trace_rho_ic, ntk_trace_ux_ic, ntk_trace_p_ic, ntk_trace_phys_avg]
				avg_trace = sum(all_traces) / len(all_traces)

				# 权重与 NTK 的迹成反比，以平衡不同损失项的学习速率
				lambda_ic_rho = avg_trace / ntk_trace_rho_ic
				lambda_ic_ux = avg_trace / ntk_trace_ux_ic
				lambda_ic_p = avg_trace / ntk_trace_p_ic
				lambda_phys = avg_trace / ntk_trace_phys_avg

				self.ntk_weights = {
					"lambda_ic_rho": lambda_ic_rho,
					"lambda_ic_ux": lambda_ic_ux,
					"lambda_ic_p": lambda_ic_p,
					"lambda_phys": lambda_phys
				}
		
		weights = self.ntk_weights
		
		# 对物理损失的权重进行退火处理
		w_R_final = self.config["neural"]["loss_function_parameters"]["w_R"]
		annealing_epochs = self.config["neural"]["loss_function_parameters"]["annealing_epochs"]
		w_R = w_R_final * min(1.0, self.epoch / annealing_epochs)

		# 计算加权后的总损失
		L_IC = (weights["lambda_ic_rho"] * L_IC_rho_raw + 
				weights["lambda_ic_ux"] * L_IC_ux_raw + 
				weights["lambda_ic_p"] * L_IC_p_raw)
				
		total_loss = L_IC + w_R * weights["lambda_phys"] * L_phys_raw

		# 记录各项损失值
		self.loss_ic_hist.append(L_IC.item())
		self.loss_ic_rho.append(L_IC_rho_raw.item())
		self.loss_ic_ux.append(L_IC_ux_raw.item()) 
		self.loss_ic_p.append(L_IC_p_raw.item()) 
		
		# 计算并记录 L2 相对误差
		self.compute_l2()
		
		return total_loss

	def compute_l2(self):
		"""计算并存储相对于解析解的 L2 相对误差。"""
		if hasattr(self, 'analytical_solution') and self.analytical_solution is not None:
			with torch.no_grad():
				prediction_analytical_space = self(self.analytical_space)
				
				rho_pred, ux_pred, p_pred = prediction_analytical_space.split(1, dim=1)
				rho_analytical, ux_analytical, p_analytical = self.analytical_solution.split(1, dim=1)
				
				l2_rho = torch.linalg.norm(rho_analytical - rho_pred) / torch.linalg.norm(rho_analytical)
				self.l2_rho_hist.append(l2_rho.item())
				
				l2_ux = torch.linalg.norm(ux_analytical - ux_pred) / torch.linalg.norm(ux_analytical)
				self.l2_ux_hist.append(l2_ux.item())
				
				l2_p = torch.linalg.norm(p_analytical - p_pred) / torch.linalg.norm(p_analytical)
				self.l2_p_hist.append(l2_p.item())

				l2_total = torch.linalg.norm(self.analytical_solution - prediction_analytical_space) / torch.linalg.norm(self.analytical_solution)
				self.l2_hist.append(l2_total.item())
		else:
			# 如果没有解析解，则用 NaN 填充，以保持列表长度一致
			self.l2_rho_hist.append(float('nan'))
			self.l2_ux_hist.append(float('nan'))
			self.l2_p_hist.append(float('nan'))
			self.l2_hist.append(float('nan'))


	def train_step(self, X, X_0, U_0, optimizer):
		"""执行单步训练。"""
		optimizer.zero_grad(set_to_none = True)
		loss = self.compute_loss(X, X_0, U_0)
		loss.backward(retain_graph = False)
		
		# 梯度裁剪，防止梯度炸
		clip_value = self.config["training_process"]["parameters"].get("gradient_clip_value", 1.0)
		torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value)
		
		optimizer.step()
		self.loss_hist.append(loss.item())
		return loss