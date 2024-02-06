# AWRQ: Activation-aware Weight Reformulation Quantization for Large Language Models


# Quick start
## Dependency
- `torch`:  tested on 1.13.1+cu117
- `transformers`: tested on version 4.34.0

## Support
- Models:
  - LLaMA, LLaMA-2
  - OPT
 
- Datasets:
  - Calibration: C4
  - Evaluation:
    - Accuracy of tasks: Piqa, ARC-e, ARC-c, BoolQ, COPA, StoryCloze
    - PPL: Wikitext2, PTB, C4
   
- Quantuzation configurations:
  - Weights: per-channel quantization
  - Activations: per-tensor dynamic quantization
  - Group quantization in weights: optional
  - Bit-widths: W4A8 (4-bit per-channel weight, 8-bit per-tensor activation), W4A6, W3A8

All experiments were run on a single NVIDIA A100-40GB.

 ## Usage
 ### LLaMA and LLaMA-2: zero_shot
 1. Full precision (FP16)
```
 CUDA_VISIBLE_DEVICES=0 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --method full
```
 3. AWRQ
```
# with smoothing
CUDA_VISIBLE_DEVICES=0 python main.py meta-llama/Llama-2-7b --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --act_bits 8 --groupsize -1 --blocksize 1 --method awrq --smooth --alpha 0.50 --min 0.01
# without smoothing
CUDA_VISIBLE_DEVICES=0 python main.py meta-llama/Llama-2-7b --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --act_bits 8 --groupsize -1 --blocksize 1 --method awrq
```
 2. SmoothQuant
```
CUDA_VISIBLE_DEVICES=0 python main.py meta-llama/Llama-2-7b --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --act_bits 8 --groupsize -1 --method smoothquant --alpha 0.50 --min 0.01
```
 4. RTN
```
CUDA_VISIBLE_DEVICES=0 python main.py meta-llama/Llama-2-7b --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --act_bits 8 --groupsize -1 --method rtn
```
 5. Weight only (GPTQ)
```
# with smoothing
CUDA_VISIBLE_DEVICES=0 python main.py meta-llama/Llama-2-7b --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --groupsize -1 --blocksize 1 --method gptq --smooth --alpha 0.50 --min 0.01
# without smoothing
CUDA_VISIBLE_DEVICES=0 python main.py meta-llama/Llama-2-7b --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --groupsize -1 --blocksize 1 --method gptq
```
 6. Activation only
```
# with smoothing
CUDA_VISIBLE_DEVICES=0 python main.py meta-llama/Llama-2-7b --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --act_bits 8 --method act_only --smooth --alpha 0.50 --min 0.01
# without smoothing
CUDA_VISIBLE_DEVICES=0 python main.py meta-llama/Llama-2-7b --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --act_bits 8 --method act_only
```


### OPT: ppl
 1. Full precision (FP16)
```
CUDA_VISIBLE_DEVICES=0 python opt.py $MODEL_DIR --calib_data c4 --method full
```
 2. AWRQ
```
# with smoothing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m --calib_data c4 --wbits 4 --act_bits 8 --groupsize -1 --blocksize 1 --method awrq --smooth --alpha 0.50 --min 0.10
# without smoothing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m --calib_data c4 --wbits 4 --act_bits 8 --groupsize -1 --blocksize 1 --method awrq
```
 3. SmoothQuant
```
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m --calib_data c4 --wbits 4 --act_bits 8 --groupsize -1 --method smoothquant --alpha 0.50 --min 0.10
```
 4. RTN
```
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m --calib_data c4 --wbits 4 --act_bits 8 --groupsize -1 --method rtn
```
 5. Weight only (GPTQ)
```
# with smoothing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m --calib_data c4 --wbits 4 --groupsize -1 --blocksize 1 --method gptq --smooth --alpha 0.50 --min 0.10
# without smoothing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m --calib_data c4 --wbits 3 --groupsize -1 --blocksize 1 --method gptq
```
 6.  Activation only
```
# with smoothing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m --calib_data c4 --act_bits 8 --method act_only --smooth --alpha 0.5 --min 0.1
# without smoothing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m --calib_data c4 --act_bits 8 --method act_only
```


# Main Results
## LLaMA and LLaMA-2 families
- Results of LLaMA and LLaMA-2 families on zero-shot tasks at W4A8 (4-bit per-channel weight, 8-bit per-tensor activation) quantization.
![image](https://github.com/linzhao-ai/AWRQ/assets/17473403/af68cb71-bdd1-4279-a9a8-a63082e2aed8)

- Speedup on LLaMA-1-30B and LLaMA-2-13B.
![image](https://github.com/zl200881/AWRQ/assets/17473403/586ee1ee-047e-48be-a701-bcc19d45bcaf)


## OPT families
- Results of OPT families via perplexity evaluation on WikiText2 at W4A8 quantization.
![image](https://github.com/zl200881/AWRQ/assets/17473403/46287f96-f9f6-4741-9630-8ae9449370c4)

- Speedup on OPT-30Bat W4A8 quantization.
![image](https://github.com/zl200881/AWRQ/assets/17473403/0c14ddfb-90da-461f-89ea-85a14a1df664)

# Acknowledgements
[GPTQ](https://github.com/IST-DASLab/gptq)

[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)

[SmoothQuant](https://github.com/mit-han-lab/smoothquant)

[OmniQuant](https://github.com/OpenGVLab/OmniQuant?tab=readme-ov-file)
