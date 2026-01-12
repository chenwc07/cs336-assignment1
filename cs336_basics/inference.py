import torch
from tokenizers import Tokenizer
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.architecture.modules import softmax
from cs336_basics.architecture.modules import TransformerLM
from cs336_basics.training.dataloader import load_checkpoint, save_checkpoint
import hydra
from omegaconf import DictConfig, OmegaConf



def softmax_with_temperature(
    logits: Float[Tensor, " ... seq_len vocab_size"],
    temperature: float = 1.0
) -> torch.Tensor:
    """对logits应用温度缩放的softmax函数。
    
    参数:
        logits (torch.Tensor): 形状为(batch_size, vocab_size)的张量，表示未归一化的对数概率。
        temperature (float): 温度参数，控制分布的平滑度。较高的温度会使分布更平滑，较低的温度会使分布更尖锐。
    """
    logits = logits / temperature
    return softmax(logits, dim=-1)

def nucleu_sampling(
    probs: Float[Tensor, " ... vocab_size"],
    top_p: float = 0.9,
) -> Int[Tensor, "..."]:
    """使用Nucleus采样从logits中采样下一个标记。
    
    参数:
        probs (torch.Tensor): 形状为(batch_size, vocab_size)的张量，表示未归一化的对数概率。
        top_p (float): 累积概率阈值，用于选择候选标记的集合。
    返回:
        torch.Tensor: 形状为(batch_size,)的张量，表示采样得到的下一个标记的索引。
    """
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True) # (..., vocab_size)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1) # (..., vocab_size)
    # 保留累积概率小于等于top_p的token，但至少保留第一个token
    mask = cumsum_probs - sorted_probs < top_p
    sorted_probs = sorted_probs * mask
    sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
    next_token_indices = torch.multinomial(sorted_probs, num_samples=1) # (..., 1)
    next_token = torch.gather(sorted_indices, -1, next_token_indices) # (..., 1)
    return next_token

@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    input_ids: Int[Tensor, " batch_size seq_len"],
    max_new_tokens: int,
    max_context_length: int,
    tokenizer: Tokenizer = None,
    temperature: float = 1.0,
    top_p: float = 0.9,
):
    for i in range(max_new_tokens):
        input_ids_cond = input_ids[:, -max_context_length:]  # (batch_size, context_length)
        logits = model(input_ids_cond)  # (batch_size, context_length, vocab_size)
        logits = logits[:, -1, :]  # (batch_size, vocab_size)
        if temperature <= 0:
            # 贪婪采样
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # (batch_size, seq_len + 1)
        else:
            probs = softmax_with_temperature(logits, temperature=temperature)  # (batch_size, vocab_size)
            next_token = nucleu_sampling(probs, top_p=top_p)  # (batch_size, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # (batch_size, seq_len + 1)
        
        # 检查是否生成了结束标记
        if tokenizer is not None:
            eos_token_id = tokenizer.token_to_id("<|endoftext|>")
            if eos_token_id is not None and next_token[0].item() == eos_token_id:
                break
    return input_ids

@hydra.main(version_base=None, config_path="../conf", config_name="config_owt")
def main(cfg: DictConfig):
    tokenizer = Tokenizer.from_file(cfg.data.tokenizer_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        num_layers=cfg.model.num_layers,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        theta=cfg.model.theta,
    ).to(device)

    it = load_checkpoint('checkpoints/openwebtext-owt2-768d-12l-64bs-8p-lr0.004/checkpoint_iter_9999.pt', model, None)
    print(f"Loaded checkpoint at iteration {it}")

    prompt = "Where is the capital of England?"
    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long)  # (1, seq_len)
    input_ids = input_ids.to(device)
    generated_ids = generate(
        model,
        input_ids,
        max_new_tokens=256,
        max_context_length=cfg.model.context_length,
        tokenizer=tokenizer,
        temperature=0.6,
        top_p=0.9,
    )
    generated_text = tokenizer.decode(generated_ids[0].cpu().numpy().tolist())
    print("Generated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()