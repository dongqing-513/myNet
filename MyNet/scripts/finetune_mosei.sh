"""
python ./TVLT/run.py with data_root='./Dataset/cmumosei/' gpus=[3] num_nodes=1 task_cls_mosei \
per_gpu_batchsize=1 num_workers=16 val_check_interval=0.2 warmup_steps=100 max_epoch=10 \
load_hub_path='TVLT.ckpt'"""

python ./run.py with data_root='/home/mz/demo/TVLT/Dataset/cmumosei/'  num_nodes=2 task_cls_mosei\
 per_gpu_batchsize=1 num_workers=16 val_check_interval=0.2 warmup_steps=100 max_epoch=10  tokenizer='/home/mz/demo/MyNet/bert'\
 load_local_path='/home/mz/demo/TVLT/TVLT-MOSEI-EA.ckpt' 
# bert_model='/home/mz/demo/MyNet/bert'
# load_local_path='.'


# 杀死进程
ps -aux | grep "python ./run.py" | grep -v "grep" | awk '{print $2}' | xargs kill -9

# [1] 3927496
nohup python ./run.py with data_root='/home/mz/demo/TVLT/Dataset/cmumosei/' \
  num_nodes=1 task_cls_mosei per_gpu_batchsize=1 num_workers=16 \
  val_check_interval=0.2 warmup_steps=100 max_epoch=10  \
  gpus=2 tokenizer='/home/mz/demo/MyNet/bert' log_dir='/home/mz/demo/MyNet/tensorboardlog' \
  load_local_path='/home/mz/demo/MyNet/TVLT-MOSEI-SA.ckpt'  > 0126fordataloader.log 2>&1 &

nohup python ./run.py with data_root='/home/mz/demo/TVLT/Dataset/cmumosei/' \
  task_cls_mosei num_workers=4 per_gpu_batchsize=1\
  val_check_interval=0.5 warmup_steps=3000 max_epoch=10  \
  gpus=2 tokenizer='/home/mz/demo/MyNet/bert' log_dir='/home/mz/demo/MyNet/tensorboardlog' \
  load_local_path='/home/mz/demo/MyNet/TVLT-MOSEI-SA.ckpt'  > 0211fortextlstm.log 2>&1 &

watch 'tail -n5 0211fortextlstm.log'

nvidia-smi

# 清理临时文件
rm -rf /tmp/pymp-*
rm -rf /tmp/tmp*