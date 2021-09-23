echo "start training..."

output_dir="./output/"

#一旦不再使用即释放内存垃圾，=1.0 垃圾占用内存大小达到10G时，释放内存垃圾
export FLAGS_eager_delete_tensor_gb=0.0
#启用快速垃圾回收策略，不等待cuda kernel 结束，直接释放显存
export FLAGS_fast_eager_deletion_mode=1
#该环境变量设置只占用0%的显存
export FLAGS_fraction_of_gpu_memory_to_use=0

sleep 5s

for i in $(seq 1 5 20)
do
    echo "================== 第 $i 次 =========================="
    unset CUDA_VISIBLE_DEVICES
    python -m paddle.distributed.launch --gpus "0,1,2,3" cmrc.py \
        --seed 1000 \
        --task_name cmrc2018 \
        --model_type ernie_gram \
        --model_name_or_path ernie-gram-zh \
        --max_seq_length 384 \
        --batch_size 4 \
        --learning_rate 3e-5 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --save_steps 1000 \
        --warmup_proportion 0.1 \
        --weight_decay 0.01 \
        --output_dir ${output_dir}/cmrc/ \
        --do_train \
        --do_predict \
        --device gpu \
        --n_best_size $i \
        --max_answer_length 30 \
    
    mkdir -p $output_dir/$i
    mv $output/cmrc $output_dir/$i/
    sleep 3s
done