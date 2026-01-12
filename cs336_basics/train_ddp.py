import os
os.environ["HYDRA_FULL_ERROR"] = "1"  # Enable full Hydra error messages
from tqdm import tqdm
import wandb
import hydra
import torch
# DDP相关导入：用于分布式数据并行训练
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
from einops import rearrange

from tokenizers import Tokenizer

from cs336_basics.architecture.modules import TransformerLM
from cs336_basics.training.dataloader import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.training.optimizers import cross_entropy_loss, AdamW, cosine_learning_rate_schedule, gradient_clipping

# 初始化分布式训练环境
def setup_ddp():
    """初始化DDP分布式训练环境，设置进程组和设备"""
    # 从环境变量中获取分布式训练参数
    dist.init_process_group(backend="nccl")  # NCCL是NVIDIA GPU的推荐后端
    local_rank = int(os.environ["LOCAL_RANK"])  # 当前进程在节点上的GPU编号
    world_size = dist.get_world_size()  # 总的进程数（GPU数）
    rank = dist.get_rank()  # 当前进程的全局编号
    torch.cuda.set_device(local_rank)  # 设置当前进程使用的GPU
    return local_rank, rank, world_size

def cleanup_ddp():
    """清理DDP分布式训练环境"""
    dist.destroy_process_group()

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
        cfg.training.max_iters = 2_621_440_000 // cfg.training.n_procs // cfg.training.batch_size // cfg.model.context_length
    if cfg.training.eval_interval is None:
        cfg.training.eval_interval = cfg.training.max_iters // 10
    if cfg.optimizer.warmup_iters is None:
        cfg.optimizer.warmup_iters = cfg.training.max_iters // 10

@hydra.main(version_base=None, config_path="../conf", config_name="config_owt")
def main(cfg: DictConfig):
    # 初始化DDP环境
    local_rank, rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}"
    
    setup(cfg)
    
    # 只在主进程（rank 0）打印配置信息
    if rank == 0:
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print(f"World size: {world_size}")
    
    # 为每个进程设置不同的随机种子，确保数据划分的随机性不同
    torch.manual_seed(cfg.training.seed)
    torch.cuda.empty_cache()
    
    if rank == 0:
        print(f"Using device: {device}")

    # --- Model and Optimizer ---
    if rank == 0:
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
    
    # 使用DDP包装模型，实现多卡并行训练
    # find_unused_parameters=False可以提高性能，但要求所有参数都参与反向传播
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr_max,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps,
    )

    # --- Data Loading ---
    if rank == 0:
        print("Loading data...")
    data_path = Path(cfg.data.path)
    train_data = np.memmap(data_path / 'train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(data_path / 'val.bin', dtype=np.uint16, mode='r')
    if rank == 0:
        print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

    # 只在主进程（rank 0）初始化wandb日志记录
    if rank == 0:
        wandb.login(key="ce0521d2e513e642494e70096e7606178ecd2158")
        logger = wandb.init(
            entity="cwc7", 
            project="cs336-basics", 
            name=cfg.training.run_name
        )
    
    # --- Training Loop ---
    if rank == 0:
        print("Starting training...")
    if cfg.training.resume_from is not None:
        # 加载checkpoint时，DDP模型需要访问module属性
        start_iter = load_checkpoint(cfg.training.resume_from, model.module, optimizer)
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
        # 只在主进程（rank 0）进行日志记录，避免重复记录
        if rank == 0 and (it % cfg.training.log_interval == 0 or it == cfg.training.max_iters - 1):
            tqdm.write(f"Iter {it}: Train loss={loss.item():.4f}, LR={lr:.6f}")
            logger.log({
                'train/loss': loss.item(), 
                'train/ppl': loss.exp().item(),
                'train/lr': lr,
                'train/grad_norm': total_norm
            }, step=it)

        # --- Evaluation ---
        # 评估前同步所有进程，防止rank 0评估时其他进程继续训练导致不同步
        if it % cfg.training.eval_interval == 0 or it == cfg.training.max_iters - 1:
            dist.barrier()  # 同步点：确保所有进程都到达这里
            
            # 只在主进程进行评估和检查点保存
            if rank == 0:
                # 评估时使用model.module获取原始模型（未包装的模型）
                eval_metrics = evaluation(model.module, val_data, cfg, device)
                tqdm.write(f"Iter {it}: Val loss={eval_metrics['val/loss']:.4f}, Val ppl={eval_metrics['val/ppl']:.4f}")
                logger.log(eval_metrics, step=it)

                # --- Checkpointing ---
                if cfg.checkpoint.save_path is not None:
                    save_path = Path(cfg.checkpoint.save_path)
                    save_path.mkdir(parents=True, exist_ok=True)
                    checkpoint_file = save_path / f"checkpoint_iter_{it}.pt"
                    # 保存checkpoint时使用model.module获取原始模型参数
                    save_checkpoint(model.module, optimizer, it, checkpoint_file)
                    print(f"Saved checkpoint to {checkpoint_file}")
            
            dist.barrier()  # 同步点：等待rank 0完成评估和保存，所有进程一起继续训练
    
    if rank == 0:
        tqdm.write("Training finished.")
    
    # 清理DDP环境
    cleanup_ddp()

if __name__ == "__main__":
    main()