import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["HYDRA_FULL_ERROR"] = "1"  # Enable full Hydra error messages
from tqdm import tqdm
import wandb
import hydra
import torch
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
from einops import rearrange

from tokenizers import Tokenizer

from cs336_basics.architecture.modules import TransformerLM
from cs336_basics.training.dataloader import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.training.optimizers import cross_entropy_loss, AdamW, cosine_learning_rate_schedule, gradient_clipping

# logger
wandb.login(key=os.getenv("WANDB_API_KEY"))
logger = wandb.init(
    entity="cwc7", 
    project="cs336-basics", 
    name="transformer-lm-training"
)

@torch.no_grad()
def evaluation(model, dataset, cfg, device):
    model.eval()
    losses = []
    for k in tqdm(range(cfg.training.eval_iters), desc="Evaluating", leave=False):
        x, y = get_batch(dataset, cfg.training.batch_size, cfg.model.context_length, device)
        logits = model(x)
        logits = rearrange(logits, "b seq_len vocab_size -> (b seq_len) vocab_size")
        y = rearrange(y, "b seq_len -> (b seq_len)")
        loss = cross_entropy_loss(logits, y)
        losses.append(loss.item())
    model.train()
    mean_loss = np.mean(losses)
    return {
        'val/loss': mean_loss,
        'val/ppl': np.exp(mean_loss),
    }

def setup(cfg):
    if cfg.optimizer.lr_min is None:
        cfg.optimizer.lr_min = cfg.optimizer.lr_max * 0.1
    if cfg.training.max_iters is None:
        cfg.training.max_iters = 327_680_000 // cfg.training.batch_size // cfg.model.context_length
    if cfg.training.eval_interval is None:
        cfg.training.eval_interval = cfg.training.max_iters // 10
    if cfg.optimizer.warmup_iters is None:
        cfg.optimizer.warmup_iters = cfg.training.max_iters // 10

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    setup(cfg)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.training.seed)
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # --- Model and Optimizer ---
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        num_layers=cfg.model.num_layers,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        theta=cfg.model.theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr_max,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps,
    )

    # --- Data Loading ---
    print("Loading data...")
    data_path = Path(cfg.data.path)
    train_data = np.memmap(data_path / 'train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(data_path / 'val.bin', dtype=np.uint16, mode='r')
    print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

    # --- Training Loop ---
    print("Starting training...")
    if cfg.training.resume_from is not None:
        start_iter = load_checkpoint(cfg.training.resume_from, model, optimizer)
    else:
        start_iter = 0

    for it in tqdm(range(start_iter, cfg.training.max_iters), desc="Training"):
        lr = cosine_learning_rate_schedule(
            it,
            lr_max=cfg.optimizer.lr_max,
            lr_min=cfg.optimizer.lr_min,
            T_warmup=cfg.optimizer.warmup_iters,
            T_cos=cfg.training.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # get data
        x, y = get_batch(train_data, cfg.training.batch_size, cfg.model.context_length, device)

        # forward pass
        logits = model(x)
        logits = rearrange(logits, "b seq_len vocab_size -> (b seq_len) vocab_size")
        y = rearrange(y, "b seq_len -> (b seq_len)")
        loss = cross_entropy_loss(logits, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        total_norm = gradient_clipping(
            model.parameters(),
            max_norm=cfg.optimizer.max_l2_norm,
        )

        # optimizer step
        optimizer.step()

        # --- Logging ---
        if it % cfg.training.log_interval == 0 or it == cfg.training.max_iters - 1:
            tqdm.write(f"Iter {it}: Train loss={loss.item():.4f}, LR={lr:.6f}")
            logger.log({
                'train/loss': loss.item(), 
                'train/ppl': loss.exp().item(),
                'train/lr': lr,
                'train/grad_norm': total_norm
            }, step=it)

        # --- Evaluation ---
        if it % cfg.training.eval_interval == 0 or it == cfg.training.max_iters - 1:
            eval_metrics = evaluation(model, val_data, cfg, device)
            tqdm.write(f"Iter {it}: Val loss={eval_metrics['val/loss']:.4f}, Val ppl={eval_metrics['val/ppl']:.4f}")
            logger.log(eval_metrics, step=it)

            # --- Checkpointing ---
            if cfg.checkpoint.save_path is not None:
                save_path = Path(cfg.checkpoint.save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                checkpoint_file = save_path / f"checkpoint_iter_{it}.pt"
                save_checkpoint(model, optimizer, it, checkpoint_file)
                print(f"Saved checkpoint to {checkpoint_file}")
    
    tqdm.write("Training finished.")
    

if __name__ == "__main__":
    main()