import torch
import torch.nn as nn
from quant import *
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM

DEV = torch.device('cuda:0')


# def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
def find_layers(module, layers=[nn.Conv2d, nn.Linear, SmoothAndQuantLinear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def replace_linear_layer(model_name, module, smooth=False, alpha=0.50, min=0.01, act_quant=False, act_bits=8):
    # replace OPTAttention
    module.self_attn.q_proj = SmoothAndQuantLinear.from_float(module.self_attn.q_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
    module.self_attn.k_proj = SmoothAndQuantLinear.from_float(module.self_attn.k_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
    module.self_attn.v_proj = SmoothAndQuantLinear.from_float(module.self_attn.v_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)

    if 'opt' in model_name:
        module.self_attn.out_proj = SmoothAndQuantLinear.from_float(module.self_attn.out_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.fc1 = SmoothAndQuantLinear.from_float(module.fc1, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.fc2 = SmoothAndQuantLinear.from_float(module.fc2, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
    elif 'llama' in model_name:
        module.self_attn.o_proj = SmoothAndQuantLinear.from_float(module.self_attn.o_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.mlp.gate_proj = SmoothAndQuantLinear.from_float(module.mlp.gate_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.mlp.up_proj = SmoothAndQuantLinear.from_float(module.mlp.up_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
        module.mlp.down_proj = SmoothAndQuantLinear.from_float(module.mlp.down_proj, smooth=smooth, alpha=alpha, min=min, act_quant=act_quant, act_bits=act_bits)
    else:
        print(f'Not supported model: {model_name}')

    return module
