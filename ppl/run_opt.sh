
# OPT families
# MODEL_NAME: 125m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b
MODEL="opt"
MODEL_NAME="125m"
MODEL_DIR="/PATH/TO/OPT/opt-125m"
start=`date +%s`

# FULL: W16A16
# CUDA_VISIBLE_DEVICES=0 python ./$MODEL.py $MODEL_DIR --calib_data c4 --method full

# AWRQ: W4A8, W4A6
# with smoothing
CUDA_VISIBLE_DEVICES=0 python ./$MODEL.py $MODEL_DIR --calib_data c4 --wbits 4 --act_bits 8 --groupsize -1 --blocksize 1 --method awrq --smooth --alpha 0.50 --min 0.10
# without smoothing
# CUDA_VISIBLE_DEVICES=0 python ./$MODEL.py $MODEL_DIR --calib_data c4 --wbits 4 --act_bits 8 --groupsize -1 --blocksize 1 --method awrq 

# SmoothQuant
# CUDA_VISIBLE_DEVICES=0 python ./$MODEL.py $MODEL_DIR --calib_data c4 --wbits 4 --act_bits 8 --groupsize -1 --method smoothquant --alpha 0.50 --min 0.10

# RTN: W4A8
# CUDA_VISIBLE_DEVICES=0 python ./$MODEL.py $MODEL_DIR --calib_data c4 --wbits 4 --act_bits 8 --groupsize -1 --method rtn 

# weight only: W4A16, W3A16
# with smoothing
# CUDA_VISIBLE_DEVICES=0 python ./$MODEL.py $MODEL_DIR --calib_data c4 --wbits 4 --groupsize -1 --blocksize 1 --method gptq --smooth --alpha 0.50 --min 0.10
# without smoothing
# CUDA_VISIBLE_DEVICES=0 python ./$MODEL.py $MODEL_DIR --calib_data c4 --wbits 3 --groupsize -1 --blocksize 1 --method gptq

# act only: W16A8, W16A6
# with smoothing
# CUDA_VISIBLE_DEVICES=0 python ./$MODEL.py $MODEL_DIR --calib_data c4 --act_bits 8 --method act_only --smooth --alpha 0.5 --min 0.1
# without smoothing
# CUDA_VISIBLE_DEVICES=0 python ./$MODEL.py $MODEL_DIR --calib_data c4 --act_bits 8 --method act_only

end=`date +%s`
echo "Execution Time is: $(($((end-start))/60)) min"

