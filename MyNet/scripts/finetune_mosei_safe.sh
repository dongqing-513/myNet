#!/bin/bash

# Run with safer DataLoader settings
nohup python ./run.py with data_root='/home/mz/demo/TVLT/Dataset/cmumosei/' \
  task_cls_mosei \
  num_workers=4 \
  per_gpu_batchsize=1 \
  val_check_interval=0.5 \
  warmup_steps=3000 \
  max_epoch=10 \
  gpus=2 \
  tokenizer='/home/mz/demo/MyNet/bert' \
  log_dir='/home/mz/demo/MyNet/tensorboardlog' \
  load_local_path='/home/mz/demo/MyNet/TVLT-MOSEI-SA.ckpt' > $(date +%Y%m%d)_training.log 2>&1 &

# Monitor the log
watch 'tail -n5 $(date +%Y%m%d)_training.log'
