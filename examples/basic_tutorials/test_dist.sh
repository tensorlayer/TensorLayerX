export MLU_VISIBLE_DEVICES=0,1
python -m tensorlayerx.distributed.launch --nproc_per_node=2 cifar10_cnn_torch_dist.py 
