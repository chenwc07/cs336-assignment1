uv run cs336_basics/inference.py \
    model.d_model=768 \
    model.num_layers=12 \
    model.d_ff=2048 \
    optimizer.lr_max=0.004 \
    training.batch_size=64