
pip uninstall  paddlenlp -y
pip install pypinyin==0.38.1

tar zxf /root/paddlejob/workspace/train_data/datasets/data108799/ChineseBERT-Paddle-code.tar.gz -C ./

echo "current dir $pwd"

mv ChineseBERT-Paddle-code ChineseBERT-Paddle

export PYTHONPATH="/root/paddlejob/workspace/code/ChineseBERT-Paddle/Paddle_ChineseBert/PaddleNLP"

# ai studio 会自动压缩该目录下的所有文件
output_dir="/root/paddlejob/workspace/output/"

cd /root/paddlejob/workspace/code/ChineseBERT-Paddle
mv /root/paddlejob/workspace/code/run.py /root/paddlejob/workspace/code/ChineseBERT-Paddle
echo "start training..."

sleep 5s
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1,2,3" run.py \
    --max_seq_length 512 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --weight_decay 0.001 \
    --epochs 2 \
    --warmup_proportion 0.1 \
    --seed 1000 \
    --device gpu \

