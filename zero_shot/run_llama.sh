#########################################################################

# LLAMA: 7b, 13b, 30b, 65b
MODEL_DIR="/PATH/TO/LLaMA/llama-7b"
start=`date +%s`

# FULL (W16A16)
# CUDA_VISIBLE_DEVICES=3 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --method full

# AWRQ
# with smoothing
CUDA_VISIBLE_DEVICES=2 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --act_bits 8 --groupsize -1 --blocksize 1 --method awrq --smooth --alpha 0.50 --min 0.01
# without smoothing
# CUDA_VISIBLE_DEVICES=2 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --act_bits 8 --groupsize -1 --blocksize 1 --method awrq 

# SmoothQuant
# CUDA_VISIBLE_DEVICES=5 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --act_bits 8 --groupsize -1 --method smoothquant --alpha 0.50 --min 0.01 

# RTN: W4A8, W4A6, W3A8
# CUDA_VISIBLE_DEVICES=0 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --act_bits 8 --groupsize -1 --method rtn

# weight only: GPTQ W4A16
# with smoothing
# CUDA_VISIBLE_DEVICES=1 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --groupsize -1 --blocksize 1 --method gptq --smooth --alpha 0.50 --min 0.01
# without smoothing
# CUDA_VISIBLE_DEVICES=3 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --wbits 4 --groupsize -1 --blocksize 1 --method gptq

# activation only: W16A8
# with smoothing
# CUDA_VISIBLE_DEVICES=1 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --act_bits 8 --method act_only --smooth --alpha 0.50 --min 0.01
# without smoothing
# CUDA_VISIBLE_DEVICES=4 python main.py $MODEL_DIR --calib_data c4 --tasks piqa,arc_easy,arc_challenge,boolq,copa,storycloze --table_results --act_bits 8 --method act_only 

end=`date +%s`
echo "Execution Time is: $(($((end-start))/60)) min"

