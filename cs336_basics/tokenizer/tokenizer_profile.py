import cProfile
import pstats
from pstats import SortKey
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

def profile_encode():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="tests/fixtures/gpt2_vocab.json",
        merges_path="tests/fixtures/gpt2_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    
    corpus_path = "tests/fixtures/tinystories_sample.txt"
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = tokenizer.encode(text)
    
    profiler.disable()
    
    # 打印统计信息
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    print("\n=== Top 20 functions by cumulative time ===")
    stats.print_stats(20)
    
    return result

if __name__ == "__main__":
    profile_encode()