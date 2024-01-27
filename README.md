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
  - Group quantization in weights
  - bit-widths: W4A8 (4-bit per-channel weight, 8-bit per-tensor activation), W4A6, W3A8

All experiments were run on a single NVIDIA A100-40GB.

 ## Usage
```
python 
```

# Main Results
## LLaMA and LLaMA-2 families
- Results of LLaMA and LLaMA-2 families on zero-shot tasks at W4A8 (4-bit per-channel weight, 8-bit per-tensor activation) quantization.
![image](https://github.com/zl200881/AWRQ/assets/17473403/2b264eaf-a6d4-458b-9f8e-c49ad2a8595b)

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
