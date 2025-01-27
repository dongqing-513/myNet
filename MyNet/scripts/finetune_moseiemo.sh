python ./TVLT/run.py with data_root='/home/mz/demo/TVLT/Dataset/cmumosei/'  num_nodes=1 task_cls_moseiemo\
 per_gpu_batchsize=4 num_workers=16 val_check_interval=0.2 warmup_steps=100 max_epoch=10  tokenizer='/home/mz/demo/TVLT/bert'\
  load_local_path='/home/mz/demo/TVLT/TVLT-MOSEI-EA.ckpt'
# load_local_path='.'  load_hub_path='TVLT.ckpt'

