import transformers
import torch
from .models_utils import BaseLM, find_layers, replace_linear_layer
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
from .quant import *
from .awrq import AWRQ


class LLAMAClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype='auto')
        self.seqlen = 2048
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.vocab_size = self.tokenizer.vocab_size
        print('LLAMA vocab size: ', self.vocab_size)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            return self.model.config.max_position_embeddings
    @property
    def max_gen_toks(self):
        print('max_gen_toks fn')
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :32000]

    @torch.no_grad()
    def _model_logits_on_dataset(self, dataset_inps):
        print('Evaluating ...')

        nsamples = len(dataset_inps)

        model = self.model
        dev = self.device

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers

        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = []
        outs = []
        for batch_idx, batch in enumerate(dataset_inps):
            inps.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))
            outs.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))

        cache = {'i': 0, 'attention_masks': [], 'position_ids': []}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_masks'].append(kwargs['attention_mask'].detach().cpu())
                cache['position_ids'].append((kwargs['position_ids']).detach().cpu())
                raise ValueError

        layers[0] = Catcher(layers[0])
        for i in range(nsamples):
            batch = dataset_inps[i].to(dev)
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        torch.cuda.empty_cache()

        attention_masks = cache['attention_masks']
        position_ids = cache['position_ids']

        for i in range(len(layers)):
            print(i)
            layer = layers[i].to(dev)

            for j in range(nsamples):
                outs[j] = layer(inps[j].to(self.device), attention_mask=attention_masks[j].to(self.device), position_ids=position_ids[j].to(self.device))[0].detach().cpu()

            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)
        model.lm_head = model.lm_head.to(dev)


        if self.model.model.norm is not None:
            self.model.model.norm = self.model.model.norm.to(self.device)
        self.model.lm_head = self.model.lm_head.to(self.device)

        dataset_logits = []

        for i in tqdm(range(nsamples), desc='Last Layer'):
            hidden_states = inps[i].unsqueeze(0).to(self.device)
            if self.model.model.norm is not None:
                hidden_states = self.model.model.norm(hidden_states)
            batch_logits = F.log_softmax(self.model.lm_head(hidden_states)[0][:, :, :50272], dim=-1).cpu()
            dataset_logits.append(batch_logits)
        model.config.use_cache = use_cache
        return dataset_logits


    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(self._model_call(batch), dim=-1).cpu() # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits


    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )

    def smoothing_layer(self, layer, subset, awrq, inps, attention_mask, position_ids, outs):

        def add_batch_act_scales(name):
            def tmp(_, inp, out):
                awrq[name].add_batch_act_scales(inp[0].data, out.data)
            return tmp

        # generate act scales
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch_act_scales(name)))
        for j in range(self.args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0] # hook

        # smooth scales
        qkv_wight_scales = None
        mlp_wight_scales = None
        for name in subset:
            subset[name].act_scales = awrq[name].act_scales.clone() # act scales
            # weight scales
            subset[name].weight_scales = subset[name].weight.abs().max(dim=0)[0]
            # qkv weight scales
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                qkv_wight_scales = torch.max(qkv_wight_scales, subset[name].weight_scales) if qkv_wight_scales is not None else subset[name].weight_scales
            # mlp weight scales
            if 'gate_proj' in name or 'up_proj' in name:
                mlp_wight_scales = torch.max(mlp_wight_scales, subset[name].weight_scales) if mlp_wight_scales is not None else subset[name].weight_scales
        # absorb smooth_scales by linear
        for name in subset:
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                subset[name].weight_scales = qkv_wight_scales
            if 'gate_proj' in name or 'up_proj' in name:
                subset[name].weight_scales = mlp_wight_scales
            subset[name].smooth_scales = (subset[name].act_scales.pow(subset[name].alpha) / subset[name].weight_scales.pow(1-subset[name].alpha)).clamp(min=subset[name].min)
            subset[name].weight.data *= subset[name].smooth_scales # weight*scales
            subset[name].act_scales = None 
            subset[name].weight_scales = None
        # absorb smooth_scales by layer norm 
        layer.input_layernorm.weight.data /= layer.self_attn.k_proj.smooth_scales            
        layer.post_attention_layernorm.weight.data /= layer.mlp.gate_proj.smooth_scales
        if hasattr(layer.input_layernorm, 'bias'):
            layer.input_layernorm.bias.data /= layer.self_attn.k_proj.smooth_scales            
        if hasattr(layer.post_attention_layernorm, 'bias'):
            layer.post_attention_layernorm.bias.data /= layer.mlp.gate_proj.smooth_scales
        layer.self_attn.q_proj.act_smoothed = True
        layer.self_attn.k_proj.act_smoothed = True
        layer.self_attn.v_proj.act_smoothed = True
        layer.mlp.gate_proj.act_smoothed = True
        layer.mlp.up_proj.act_smoothed = True

        for h in handles:
            h.remove()

    @torch.no_grad()
    def llama_sequential(self, dataloader):
        # method: act_only, GPTQ, RTN, SmoothQuant, AWRQ
        print(f'args={self.args}')
        print('Starting ...')

        model = self.model
        dev = self.device

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers

        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (self.args.nsamples, self.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
                cache['position_ids'] = kwargs['position_ids']
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']

        print('Ready.')

        quantizers = {}
        quant_time = []
        for i in range(len(layers)):
            layer = layers[i].to(dev)

            # replace Linear with SmoothAndQuantLinear
            layer = replace_linear_layer(self.model_name, layer, smooth=self.args.smooth, alpha=self.args.alpha, min=self.args.min, act_quant=False, act_bits=self.args.act_bits)

            full = find_layers(layer)

            if self.args.true_sequential:
                sequential = [
                    ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                    ['self_attn.o_proj'],
                    ['mlp.up_proj', 'mlp.gate_proj'],
                    ['mlp.down_proj']
                ]
            else:
                sequential = [list(full.keys())]

            # quantize weights and activation in layer
            for names in sequential:
                subset = {n: full[n] for n in names}

                # define awrq 
                awrq = {}
                for name in subset:
                    awrq[name] = AWRQ(subset[name], self.args.method)
                    awrq[name].quantizer = Quantizer()
                    awrq[name].quantizer.configure(self.args.wbits, perchannel=True, sym=False, mse=False)

                def add_batch(name):
                    def tmp(_, inp, out):
                        awrq[name].add_batch(inp[0].data, out.data)

                    return tmp

                # smooth and quantize activations
                if self.args.smooth:
                    self.smoothing_layer(layer, subset, awrq, inps, attention_mask, position_ids, outs)

                # Hessian matrix H
                if self.args.method in ['gptq', 'awrq']:
                    handles = []
                    for name in subset:
                        handles.append(subset[name].register_forward_hook(add_batch(name)))
                    for j in range(self.args.nsamples):
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                    for h in handles:
                        h.remove()

                # quantize weights
                tick = time.time()
                for name in subset:
                    if self.args.method in ['gptq', 'rtn', 'smoothquant', 'awrq']:
                        print(i, name)
                        print('Quantizing ...')
                        awrq[name].quant_weight(percdamp=self.args.percdamp, blocksize=self.args.blocksize, groupsize=self.args.groupsize, method=self.args.method, actorder=self.args.actorder) 
                        quantizers['model.decoder.layers.%d.%s' % (i, name)] = awrq[name].quantizer
                    subset[name].act_quant = self.args.act_quant # act_quant 
                    awrq[name].free()
                quant_time.append(time.time() - tick)

            # quantized results
            for j in range(self.args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            layers[i] = layer.cpu()
            del layer
            del awrq
            torch.cuda.empty_cache()

            inps, outs = outs, inps

        model.config.use_cache = use_cache
        import numpy as np
        if len(quant_time) > 0:
            print(f'total quant time={np.sum(quant_time)/60} min')

        return quantizers


# for backwards compatibility
LLAMA = LLAMAClass
