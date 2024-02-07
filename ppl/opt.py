import time

import torch
import torch.nn as nn

from awrq import *
from modelutils import *
from quant import *


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

def smoothing_layer(layer, subset, awrq, inps, attention_mask, outs):

    def add_batch_act_scales(name):
        def tmp(_, inp, out):
            awrq[name].add_batch_act_scales(inp[0].data, out.data)
        return tmp

    # generate act scales
    handles = []
    for name in subset:
        handles.append(subset[name].register_forward_hook(add_batch_act_scales(name)))
    for j in range(args.nsamples):
        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] # hook
    # smooth scales
    qkv_wight_scales = None
    for name in subset:
        subset[name].act_scales = awrq[name].act_scales.clone() # act scales
        # weight scales
        subset[name].weight_scales = subset[name].weight.abs().max(dim=0)[0]
        # qkv weight scales
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            qkv_wight_scales = torch.max(qkv_wight_scales, subset[name].weight_scales) if qkv_wight_scales is not None else subset[name].weight_scales
    # absorb smooth_scales: linear
    for name in subset:
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            subset[name].weight_scales = qkv_wight_scales
        subset[name].smooth_scales = (subset[name].act_scales.pow(subset[name].alpha) / subset[name].weight_scales.pow(1-subset[name].alpha)).clamp(min=subset[name].min)
        subset[name].weight.data *= subset[name].smooth_scales # weight*scales
        subset[name].act_scales = None 
        subset[name].weight_scales = None
    # absorb smooth_scales: layer norm 
    layer.self_attn_layer_norm.weight.data /= layer.self_attn.k_proj.smooth_scales            
    layer.self_attn_layer_norm.bias.data /= layer.self_attn.k_proj.smooth_scales            
    layer.final_layer_norm.weight.data /= layer.fc1.smooth_scales
    layer.final_layer_norm.bias.data /= layer.fc1.smooth_scales
    layer.self_attn.q_proj.act_smoothed = True
    layer.self_attn.k_proj.act_smoothed = True
    layer.self_attn.v_proj.act_smoothed = True
    layer.fc1.act_smoothed = True

    for h in handles:
        h.remove()

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    # method: act_only, gptq(weight_only), rtn, smoothquant, awrq
    print('Ready.')
    print(f'args={args}')

    quantizers = {}
    quant_time = []
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        # replace Linear with SmoothAndQuantLinear
        layer = replace_linear_layer(args.model, layer, smooth=args.smooth, alpha=args.alpha, min=args.min, act_quant=False, act_bits=args.act_bits)

        # layer is an OPTDecoderLayer, contains k_proj,q_proj,v_proj,fc1,fc2...
        subset = find_layers(layer)
        awrq = {}
        for name in subset:
            awrq[name] = AWRQ(subset[name], args.method)
            awrq[name].quantizer = Quantizer()  # quantizer in quant.py
            awrq[name].quantizer.configure(args.wbits, perchannel=args.perchannel, sym=args.sym, mse=False, trits=args.trits)

        def add_batch(name):
            def tmp(_, inp, out):
                awrq[name].add_batch(inp[0].data, out.data)
            return tmp

        # smooth
        if args.smooth:
            smoothing_layer(layer, subset, awrq, inps, attention_mask, outs)
            
        # Hessian matrix H
        if args.method in ['gptq', 'awrq']:
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            # hook for H
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0] # hook
            for h in handles:
                h.remove()

        # quantize weights
        tick = time.time()
        for name in subset:
            if args.method in ['gptq', 'rtn', 'smoothquant', 'awrq']:
                print(i, name)
                print('Quantizing ...')
                awrq[name].quant_weight(percdamp=args.percdamp, blocksize=args.blocksize, groupsize=args.groupsize, method=args.method, actorder=args.actorder)
                quantizers['model.decoder.layers.%d.%s' % (i, name)] = awrq[name].quantizer
            subset[name].act_quant = args.act_quant # act_quant 
            awrq[name].free()
        quant_time.append(time.time() - tick)

        # quantized results
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del awrq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    if len(quant_time) > 0:
        print(f'total quant time={np.sum(quant_time)/60} min')
    
    return quantizers

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print('PPL: ', ppl.item())

    model.config.use_cache = use_cache

def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=args.faster_kernel)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

def load_quant3(model, checkpoint):
    from transformers import OPTConfig, OPTForCausalLM 
    config = OPTConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    make_quant3(model, layers, faster=args.faster_kernel)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

def opt_multigpu(model, gpus):
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus

def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape(-1),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        import numpy as np
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--calib_data', type=str, default='c4', choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, 
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--act_bits', type=int, default=8, 
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=1,
        help='Blocksize to use for quantization.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--actorder', action='store_true',
        help='Whether to apply the activation order'
    )
    parser.add_argument(
        '--act_quant', default=False, action='store_true',
        help='bits used for inps quantization'
    )
    parser.add_argument(
        '--act_sym', default=False, action='store_true',
        help='bits used for inps quantization'
    )
    parser.add_argument(
        '--perchannel', default=False, action='store_true',
        help='Perchannel or not'
    )
    parser.add_argument("--smooth", default=False, action='store_true', help='smooth')
    parser.add_argument("--alpha", type=float, default=0.50) # smooth param
    parser.add_argument("--min", type=float, default=0.1) # smooth param
    parser.add_argument("--method", default='full', choices=['full', 'act_only', 'gptq', 'rtn', 'smoothquant', 'awrq'])
    args = parser.parse_args()

    if args.method == 'full':
        args.smooth = False
        args.act_quant = False
    elif args.method == 'act_only':
        args.act_quant = True
    elif args.method == 'gptq':
        args.act_quant = False
        args.perchannel = True
    elif args.method == 'rtn':
        args.smooth = False
        args.act_quant = True
        args.perchannel = True
    elif args.method == 'smoothquant':
        args.smooth = True
        args.act_quant = True
        args.perchannel = True
    elif args.method == 'awrq':
        args.act_quant = True
        args.perchannel = True

    if args.load:
        model = load_quant3(args.model, args.load)
    else:
        model = get_opt(args.model)
        model.eval()

    dataloader, testloader = get_loaders(
        args.calib_data, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.method != 'full':
        tick = time.time()
        quantizers = opt_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)
    if args.load:
        exit()

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
      datasets = ['wikitext2']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV)

    if args.save:
        opt_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save) 
