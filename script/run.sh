# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run torchrun --nproc_per_node=8 cs336_basics/train_ddp.py \
#     model.d_model=768 \
#     model.num_layers=12 \
#     model.d_ff=2048 \
#     optimizer.lr_max=0.001 \
#     training.batch_size=64 \
#     training.eval_interval=1000 \
#     training.n_procs=8 \
#     checkpoint.save_path=checkpoints/openwebtext2 \
#     'training.run_name=owt2-768d-12l-64bs-8p-lr0.001' \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run torchrun --nproc_per_node=8 cs336_basics/train_ddp.py \
    model.d_model=768 \
    model.num_layers=12 \
    model.d_ff=2048 \
    optimizer.lr_max=0.004 \
    training.batch_size=64 \
    training.eval_interval=1000 \
    training.n_procs=8 \
    checkpoint.save_path=checkpoints/openwebtext-owt2-768d-12l-64bs-8p-lr0.004 \
    'training.run_name=owt2-768d-12l-64bs-8p-lr0.004' \