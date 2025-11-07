#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LinearForVeRA(nn.Linear, LoRALayer):
    # VeRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        vera_A,
        vera_B,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.vera_A =vera_A
        self.vera_B =vera_B

        # Actual trainable parameters
        self.vera_lambda_b = nn.Parameter(torch.ones(out_features))
        self.vera_lambda_d = nn.Parameter(torch.randn(r))

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.scaling=1

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'vera_lambda_b'):
            with torch.no_grad():
                nn.init.zeros_(self.vera_lambda_d).fill_(0.1)
                nn.init.zeros_(self.vera_lambda_b)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    sliced_A = self.vera_A[:, : self.in_features].to(self.vera_lambda_d.device)
                    sliced_B = self.vera_B[: self.out_features, :].to(self.vera_lambda_d.device)
                    self.weight.data -= T(torch.diag(self.vera_lambda_b)@sliced_B@ torch.diag(self.vera_lambda_d)@sliced_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    sliced_A = self.vera_A[:, : self.in_features].to(self.vera_lambda_d.device)
                    sliced_B = self.vera_B[: self.out_features, :].to(self.vera_lambda_d.device)
                    self.weight.data += T(torch.diag(self.vera_lambda_b)@sliced_B@ torch.diag(self.vera_lambda_d)@sliced_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            sliced_A = self.vera_A[:, : self.in_features].to(x.device)
            sliced_B = self.vera_B[: self.out_features, :].to(x.device)
            result += self.lora_dropout(x)@sliced_A.transpose(0, 1)*self.vera_lambda_d @ sliced_B.transpose(0, 1)*self.vera_lambda_b*self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LinearForDictLoRA(nn.Linear, LoRALayer):
    # DictLoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key: str, # should be a str
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.key = key

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            self.lora_A[key]=nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B[key]=nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = 1.0
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for v in self.lora_A.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
            for v in self.lora_B.values():
                nn.init.zeros_(v)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B[self.key] @ self.lora_A[self.key]) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B[self.key] @ self.lora_A[self.key]) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A[self.key].transpose(0, 1) @ self.lora_B[self.key].transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LinearForDictLoRA4LanFusion(nn.Linear, LoRALayer):
    # DictLoRA4LanFusion implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key_list: list, # should be a list
        r: list, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.key_list = key_list

        # Actual trainable parameters
        if r[0] > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for id,k in enumerate(key_list):
                self.lora_A[k]=nn.Parameter(self.weight.new_zeros((r[id], in_features)))
                self.lora_B[k]=nn.Parameter(self.weight.new_zeros((out_features, r[id])))
                
            self.scaling=1/len(key_list)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for v in self.lora_A.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
            for v in self.lora_B.values():
                nn.init.zeros_(v)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r[0] > 0:
                    self.weight.data -= self.scaling * T(torch.cat([ self.lora_B[k] for k in self.key_list],dim=-1) @ torch.cat([ self.lora_A[k] for k in self.key_list],dim=0))
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r[0] > 0:
                    self.weight.data += self.scaling * T(torch.cat([ self.lora_B[k] for k in self.key_list],dim=-1) @ torch.cat([ self.lora_A[k] for k in self.key_list],dim=0))
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r[0] > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += self.scaling * self.lora_dropout(x) @ torch.cat([ self.lora_A[k].transpose(0, 1) for k in self.key_list],dim=-1) @ torch.cat([ self.lora_B[k].transpose(0, 1) for k in self.key_list],dim=0)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LinearForDictLoRA4VeLoRA(nn.Linear, LoRALayer):
    # DictLoRA4veLoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key_list: list, # should be a list
        r: List, # each value should be paired with key_list
        initial_type="ones", # for lora_A/B_kid, not for lora_A_kid_scaling
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        temperature=6.0, # this is empirical value, since we don't want the scaling weights to be sharper
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.key_list = key_list
        r_father=sum(r)
        self.r_father=r_father
        self.initial_type=initial_type
        self.temperature=temperature
        
        if r_father > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for id, k in enumerate(key_list):
                self.lora_A[k]=nn.Parameter(self.weight.new_zeros((r[id], in_features)))
                self.lora_B[k]=nn.Parameter(self.weight.new_zeros((out_features, r[id])))
            
            # Actual trainable parameters
            self.lora_A_kid=nn.Parameter(self.weight.new_ones((r_father)))
            self.lora_B_kid=nn.Parameter(self.weight.new_ones((out_features)))
            # we do not use other initialization methods for lora_A_kid_scaling
            self.lora_A_kid_scaling=nn.Parameter(self.weight.new_ones((len(key_list))))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A_kid'):
            if self.initial_type == "ones":
                pass
            elif self.initial_type == "vera":
                self.lora_A_kid.data.fill_(0.1)
                nn.init.zeros_(self.lora_B_kid)
            elif self.initial_type=="kaiming":
                nn.init.kaiming_uniform(self.lora_A_kid[:,None],a=math.sqrt(5))
                nn.init.kaiming_uniform(self.lora_B_kid[:,None],a=math.sqrt(5))
                self.lora_A_kid.unsqueeze(-1)
                self.lora_B_kid.unsqueeze(-1)
            else:
                raise NotImplementedError

    def get_loraA_integration(self):
        return torch.cat([ self.lora_A[k].transpose(0, 1)*self.get_softmax_scaling()[id] for id,k in enumerate(self.key_list) ], dim=-1)

    def get_loraB_integration(self):
        return torch.cat([ self.lora_B[k].transpose(0, 1) for k in self.key_list ],dim=0)

    def get_softmax_scaling(self):
        return torch.nn.functional.softmax(self.lora_A_kid_scaling/self.temperature, dim=0)
    
    def get_A_kid(self):
        return self.lora_A_kid
    
    def get_B_kid(self):
        return self.lora_B_kid
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r_father > 0:
                    # original VeLoRA
                    self.weight.data -= T( torch.diag(self.get_B_kid()) @ self.get_loraB_integration().T @ torch.diag(self.get_A_kid()) @ self.get_loraA_integration().T )
                    # remove A vector
                    # self.weight.data -= T( torch.diag(self.get_B_kid()) @ self.get_loraB_integration().T @ self.get_loraA_integration().T )
                    # remove B vector
                    # self.weight.data -= T( self.get_loraB_integration().T @ torch.diag(self.get_A_kid()) @ self.get_loraA_integration().T )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r_father > 0:
                    # original VeLoRA
                    self.weight.data += T( torch.diag(self.get_B_kid()) @ self.get_loraB_integration().T @ torch.diag(self.get_A_kid()) @ self.get_loraA_integration().T )
                    # remove A vector
                    # self.weight.data += T( torch.diag(self.get_B_kid()) @ self.get_loraB_integration().T @ self.get_loraA_integration().T )
                    # remove B vector
                    # self.weight.data += T( self.get_loraB_integration().T @ torch.diag(self.get_A_kid()) @ self.get_loraA_integration().T )
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r_father > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            # original VeLoRA
            result += self.lora_dropout(x) @ self.get_loraA_integration() * self.get_A_kid() @ self.get_loraB_integration() * self.get_B_kid()
            # remove A vector
            # result += self.lora_dropout(x) @ self.get_loraA_integration() @ self.get_loraB_integration() * self.get_B_kid()
            # remove B vector
            # result += self.lora_dropout(x) @ self.get_loraA_integration() * self.get_A_kid() @ self.get_loraB_integration()
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LinearForDictLoRA4VeLoRA_Add_lora_A_kid_(nn.Linear, LoRALayer):
    # DictLoRA4veLoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key_list: list, # should be a list
        r: List, # each value should be paired with key_list
        initial_type="ones", # for lora_A/B_kid, not for lora_A_kid_scaling
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        temperature=6.0, # this is empirical value, since we don't want the scaling weights to be sharper
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.key_list = key_list
        r_father=sum(r)
        self.r_father=r_father
        self.initial_type=initial_type
        self.temperature=temperature
        
        if r_father > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for id, k in enumerate(key_list):
                self.lora_A[k]=nn.Parameter(self.weight.new_zeros((r[id], in_features)))
                self.lora_B[k]=nn.Parameter(self.weight.new_zeros((out_features, r[id])))
            
            # Actual trainable parameters
            self.lora_A_kid=nn.Parameter(self.weight.new_ones((r_father)))
            self.lora_A_kid_=nn.Parameter(self.weight.new_ones((in_features)))
            self.lora_B_kid=nn.Parameter(self.weight.new_ones((out_features)))
            # we do not use other initialization methods for lora_A_kid_scaling
            self.lora_A_kid_scaling=nn.Parameter(self.weight.new_ones((len(key_list))))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A_kid'):
            if self.initial_type == "ones":
                pass
            elif self.initial_type == "vera":
                self.lora_A_kid.data.fill_(0.1)
                nn.init.zeros_(self.lora_B_kid)
            elif self.initial_type=="kaiming":
                nn.init.kaiming_uniform(self.lora_A_kid[:,None],a=math.sqrt(5))
                nn.init.kaiming_uniform(self.lora_B_kid[:,None],a=math.sqrt(5))
                self.lora_A_kid.unsqueeze(-1)
                self.lora_B_kid.unsqueeze(-1)
            else:
                raise NotImplementedError

    def get_loraA_integration(self):
        return torch.cat([ self.lora_A[k].transpose(0, 1)*self.get_softmax_scaling()[id] for id,k in enumerate(self.key_list) ], dim=-1)

    def get_loraB_integration(self):
        return torch.cat([ self.lora_B[k].transpose(0, 1) for k in self.key_list ],dim=0)

    def get_softmax_scaling(self):
        return torch.nn.functional.softmax(self.lora_A_kid_scaling/self.temperature, dim=0)

    def get_A_kid(self):
        return self.lora_A_kid

    def get_A_kid_(self):
        return self.lora_A_kid_

    def get_B_kid(self):
        return self.lora_B_kid
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r_father > 0:
                    self.weight.data -= T( torch.diag(self.get_B_kid()) @ self.get_loraB_integration().T @ torch.diag(self.get_A_kid()) @ self.get_loraA_integration().T @ torch.diag(self.get_A_kid_()) )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r_father > 0:
                    self.weight.data += T( torch.diag(self.get_B_kid()) @ self.get_loraB_integration().T @ torch.diag(self.get_A_kid()) @ self.get_loraA_integration().T @ torch.diag(self.get_A_kid_()) )
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r_father > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += self.lora_dropout(x) * self.get_A_kid_() @ self.get_loraA_integration() * self.get_A_kid() @ self.get_loraB_integration() * self.get_B_kid()
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LinearForDictLoRA4FasterVeLoRA(nn.Linear, LoRALayer):
    # DictLoRA4FasterVeLoRA (merge lora experts) implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key_list: list, # should be a list
        r: List, # each value should be paired with key_list
        initial_type="ones", # for lora_A/B_kid, not for lora_A_kid_scaling
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        temperature=6.0, # this is empirical value, since we don't want the scaling weights to be sharper
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.key_list = key_list
        r_father=sum(r)
        self.r_father=r_father
        self.initial_type=initial_type
        self.temperature=temperature
        
        if r_father > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for id, k in enumerate(key_list):
                self.lora_A[k]=nn.Parameter(self.weight.new_zeros((r[id], in_features)))
                self.lora_B[k]=nn.Parameter(self.weight.new_zeros((out_features, r[id])))
            
            # Actual trainable parameters
            self.lora_B_kid=nn.Parameter(self.weight.new_ones((out_features)))
            # self.lora_A_kid=nn.Parameter(self.weight.new_ones((in_features)))

            self.lora_scaling=[1/len(key_list)]*len(key_list)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_B_kid'):
            if self.initial_type == "ones":
                pass

            else:
                raise NotImplementedError

    def get_loraA_integration(self):
        return torch.cat([ self.lora_A[k].transpose(0, 1)*self.lora_scaling[id] for id,k in enumerate(self.key_list) ], dim=-1)

    def get_loraB_integration(self):
        return torch.cat([ self.lora_B[k].transpose(0, 1) for k in self.key_list ],dim=0)

    def get_A_kid(self):
        return 1.0
        # return self.lora_A_kid
    
    def get_B_kid(self):
        return self.lora_B_kid

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

        # merge the lora experts
        self.weight.data += T( self.get_loraB_integration().T @ self.get_loraA_integration().T )

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        return F.linear(x* self.get_A_kid(), T(self.weight), bias=self.bias) * self.get_B_kid()


class LinearForDictLoRA4CAT(nn.Linear, LoRALayer):
    # DictLoRA4CAT implemented in a dense layer (Lora soups: Merging loras for practical skill composition tasks, https://arxiv.org/pdf/2410.13025?)
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key_list: list, # should be a list
        r: List, # each value should be paired with key_list
        initial_type="ones", # for lora_A/B_kid, not for lora_A_kid_scaling
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        temperature=6.0, # this is empirical value, since we don't want the scaling weights to be sharper
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.key_list = key_list
        r_father=sum(r)
        self.r_father=r_father
        self.initial_type=initial_type
        self.temperature=temperature
        
        if r_father > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for id, k in enumerate(key_list):
                self.lora_A[k]=nn.Parameter(self.weight.new_zeros((r[id], in_features)))
                self.lora_B[k]=nn.Parameter(self.weight.new_zeros((out_features, r[id])))
            
            # we do not use other initialization methods for lora_A_kid_scaling
            self.lora_A_kid_scaling=nn.Parameter(self.weight.new_ones((len(key_list))))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def get_loraA_integration(self):
        return torch.cat([ self.lora_A[k].transpose(0, 1)*self.get_softmax_scaling()[id] for id,k in enumerate(self.key_list) ], dim=-1)

    def get_loraB_integration(self):
        return torch.cat([ self.lora_B[k].transpose(0, 1) for k in self.key_list ],dim=0)

    def get_softmax_scaling(self):
        return torch.nn.functional.softmax(self.lora_A_kid_scaling/self.temperature, dim=0)
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r_father > 0:
                    self.weight.data -= T( self.get_loraB_integration().T @ self.get_loraA_integration().T )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r_father > 0:
                    self.weight.data += T( self.get_loraB_integration().T @ self.get_loraA_integration().T )
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r_father > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += self.lora_dropout(x) @ self.get_loraA_integration() @ self.get_loraB_integration()
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LinearForDictLoRA4ECAM(nn.Linear, LoRALayer):
    # DictLoRA4ECAM implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key: str,
        r: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out
        self.key = key
        
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            self.lora_A[key]=nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B[key]=nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = 1.0
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for v in self.lora_A.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
            for v in self.lora_B.values():
                nn.init.zeros_(v)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B[self.key] @ self.lora_A[self.key]) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B[self.key] @ self.lora_A[self.key]) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A[self.key].transpose(0, 1) @ self.lora_B[self.key].transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LinearForDictLoRA4PCAM(nn.Linear, LoRALayer):
    # DictLoRA4PCAM implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key_list: list, # should be a list
        r: list, 
        domain: str,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.key_list = key_list
        self.domain = domain

        # Actual trainable parameters
        if r[0] > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for id,k in enumerate(key_list):
                self.lora_A[k]=nn.Parameter(self.weight.new_zeros((r[id], in_features)))
                self.lora_B[k]=nn.Parameter(self.weight.new_zeros((out_features, r[id])))
                
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for v in self.lora_A.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
            for v in self.lora_B.values():
                nn.init.zeros_(v)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

    def similarity_fusion(self,x):

        # teacher (batch, time, d_k)
        students=[]
        for k, loraA in self.lora_A.items():
            if k==self.domain:
                teacher=( x @ loraA.transpose(0, 1) @ self.lora_B[k].transpose(0, 1) ).detach()
            else:
                students.append( x @ loraA.transpose(0, 1) @ self.lora_B[k].transpose(0, 1) )

        # frame-level similarity fusion, students shape --> (batch, time, n_sourch_adapters, d_k)
        students=torch.stack(students,dim=-2)
        scores = torch.matmul(teacher.unsqueeze(2), students.transpose(-2, -1)) / math.sqrt(teacher.size(-1))

        # scores and att_map: (batch, time, n_adapters)
        scores = torch.squeeze(scores, dim=2)
        att_map = torch.softmax(scores, dim=-1)

        # fusion different students
        x = torch.matmul(att_map.unsqueeze(2), students)
        x = torch.squeeze(x, dim=2)
        return x

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        result = F.linear(x, T(self.weight), bias=self.bias)
        result+=self.similarity_fusion(self.lora_dropout(x))
        return result

class LinearForDictLoRA4MOLE(nn.Linear, LoRALayer):
    # denotes MoeLoRA in our paper
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key_list: list, # should be a list
        r: List, # each value should be paired with key_list
        initial_type="ones", # for lora_A/B_kid, not for lora_A_kid_scaling
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        temperature=6.0, # this is empirical value, since we don't want the scaling weights to be sharper
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.key_list = key_list
        r_father=sum(r)
        self.r_father=r_father
        self.initial_type=initial_type
        self.temperature=temperature
        
        if r_father > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for id, k in enumerate(key_list):
                self.lora_A[k]=nn.Parameter(self.weight.new_zeros((r[id], in_features)))
                self.lora_B[k]=nn.Parameter(self.weight.new_zeros((out_features, r[id])))
            
            self.lora_router=nn.Linear(in_features, len(key_list), bias=False)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def get_loraA_integration(self, fusion_weights):
        fusion_A=fusion_weights * torch.stack([ self.lora_A[k].transpose(0, 1) for k in self.key_list ], dim=-1)
        return torch.sum(fusion_A,dim=-1)

    def get_loraB_integration(self, fusion_weights):
        fusion_B=fusion_weights * torch.stack([ self.lora_B[k].transpose(0, 1) for k in self.key_list ], dim=-1)
        return torch.sum(fusion_B,dim=-1)
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        fusion_weights=nn.functional.softmax(self.lora_router(   torch.mean(torch.mean(x,0),0)/self.temperature   ),dim=-1)
        result += self.lora_dropout(x) @ self.get_loraA_integration(fusion_weights) @ self.get_loraB_integration(fusion_weights)
        return result

class LinearForDictLoRA4SAMD(nn.Linear, LoRALayer):
    # denotes MoeLoRA* in our paper
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        key_list: list, # should be a list
        r: List, # each value should be paired with key_list
        initial_type="ones", # for lora_A/B_kid, not for lora_A_kid_scaling
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        temperature=6.0, # this is empirical value, since we don't want the scaling weights to be sharper
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.key_list = key_list
        r_father=sum(r)
        self.r_father=r_father
        self.initial_type=initial_type
        self.temperature=temperature
        
        if r_father > 0:
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for id, k in enumerate(key_list):
                self.lora_A[k]=nn.Parameter(self.weight.new_zeros((r[id], in_features)))
                self.lora_B[k]=nn.Parameter(self.weight.new_zeros((out_features, r[id])))
            
            self.lora_router=nn.Linear(in_features, len(key_list), bias=False)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def get_loraA_integration(self, fusion_weights):
        fusion_A=fusion_weights * torch.stack([ self.lora_A[k].transpose(0, 1) for k in self.key_list ], dim=-1)
        return torch.sum(fusion_A,dim=-1)

    def get_loraB_integration(self, fusion_weights):
        fusion_B=fusion_weights * torch.stack([ self.lora_B[k].transpose(0, 1) for k in self.key_list ], dim=-1)
        return torch.sum(fusion_B,dim=-1)
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        result = F.linear(x, T(self.weight), bias=self.bias)
        fusion_weights=nn.functional.softmax(self.lora_router(   torch.mean(torch.mean(x,0),0)/self.temperature   ),dim=-1)
        result += self.lora_dropout(x) @ self.get_loraA_integration(fusion_weights) @ self.get_loraB_integration(fusion_weights)
        return result





















class LinearForLoRACombineAdapterH(nn.Linear, LoRALayer):
    # LoRACombineAdapterH implemented in a dense layer
    # Combine LoRA and AdapterH (houslby adapter, see https://arxiv.org/pdf/1902.00751)
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 

        # for LoRA
        use_lora: bool = True,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,

        # for adapterH
        use_houslby: bool = False,
        bottleneck: int=128,
        adapterH_dropout: float = 0.,

        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.use_houslby=use_houslby
        self.use_lora=use_lora
        # Actual trainable parameters
        if r > 0:
            if use_lora:
                self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
                self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
                self.scaling = self.lora_alpha / self.r
                self.reset_parameters()
            
            # creating adapterH to the att.out and the second Linear of MLP
            if use_houslby:
                self.lora_adapterh_down=nn.Linear(out_features, bottleneck, bias=False)
                self.lora_adapterh_up=nn.Linear(bottleneck, out_features, bias=False)
                self.act_fn=nn.ReLU()
                self.adapterh_dropout=nn.Dropout(p=adapterH_dropout)
                self.lora_adapterh_down.apply(self.init_bert_weights)
                self.lora_adapterh_up.apply(self.init_bert_weights)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    if self.use_lora:
                        self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    if self.use_lora:
                        self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)

            # for LoRA
            if self.use_lora:       
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling

            if self.use_houslby:
                # for adapterH, x will pass the linear layer and LoRA branch first and then the adapter
                residual=result
                result=self.lora_adapterh_down(self.adapterh_dropout(result))
                result=self.act_fn(result)
                result=residual+self.lora_adapterh_up(result)
            
            return result
        else:
            result=F.linear(x, T(self.weight), bias=self.bias)
            
            if self.use_houslby:
                # for adapterH, x will pass the merged linear layer first and then the adapter
                residual=result
                result=self.lora_adapterh_down(self.adapterh_dropout(result))
                result=self.act_fn(result)
                result=residual+self.lora_adapterh_up(result)

            return result

class LinearForMosLoRA(nn.Linear, LoRALayer):
    # unofficial MosLoRA implemented in a dense layer
    # the offical implementation is at https://github.com/wutaiqiang/MoSLoRA/blob/main/commonsense_reasoning/peft/src/peft/tuners/lora.py
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # lora_AB is the learnable mix matrix in MosLoRA
            self.lora_AB=nn.Parameter(self.weight.new_zeros((r, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_AB, a=math.sqrt(5))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_AB @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_AB @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_AB.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class LinearForMeLoRA(nn.Linear, LoRALayer):
    # unofficial MeLoRA implemented in a dense layer
    # the offical implementation is at https://github.com/ChasonShi/MELoRA/blob/main/peft-0.5.0/src/peft/tuners/melora.py
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: list = [2, 4, 6, 8],
        lora_alpha: list = [2, 4, 6, 8],
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if len(r) > 0:
            self.lora_A = nn.ModuleList([])
            self.lora_B = nn.ModuleList([])
            # here, we force scaling to be 1.0
            self.scaling = 1.0
            self.l_num=len(r)
            for i, rank in enumerate(r):
                if rank > 0:
                    self.lora_A.append(nn.Linear(self.in_features//self.l_num, rank, bias=False))
                    self.lora_B.append(nn.Linear(rank, self.out_features//self.l_num, bias=False))

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.l_num):
                nn.init.kaiming_uniform_(self.lora_A[i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[i].weight)

    def get_A(self):

        return torch.block_diag(*[minilora.weight for minilora in self.lora_A])
    
    def get_B(self):

        return torch.block_diag(*[minilora.weight for minilora in self.lora_B])

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if len(self.r) > 0:
                    self.weight.data -= T(self.get_B() @ self.get_A()) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if len(self.r) > 0:
                    self.weight.data += T(self.get_B() @ self.get_A()) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if len(self.r) > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)

            #----------official implement but slow--------------#
            # x = x.to(self.lora_A[0].weight.dtype)
            # temp=[]
            # for i, rank in enumerate(self.r):
            #     if rank > 0:
            #         temp.append(
            #             self.lora_B[i](self.lora_A[i](self.lora_dropout(x[:,:,i*(self.in_features):(i+1)*(self.in_features)])))
            #         )
            # result += torch.concat(temp, dim=-1)
            # del temp

            #-------unofficial implement and much faster-----------#
            result += (self.lora_dropout(x) @ self.get_A().transpose(0, 1) @ self.get_B().transpose(0, 1)) * self.scaling

            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result

