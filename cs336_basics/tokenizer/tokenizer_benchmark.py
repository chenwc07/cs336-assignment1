import time
import tiktoken
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

def benchmark():
    print("=" * 60)
    print("Tokenizer Benchmark")
    print("=" * 60)
    
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="tests/fixtures/gpt2_vocab.json",
        merges_path="tests/fixtures/gpt2_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    
    reference_tokenizer = tiktoken.get_encoding(
        "gpt2"
        )
    
    corpus_path = "tests/fixtures/tinystories_sample.txt"
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    total_bytes = len(text.encode('utf-8'))
    
    # Warm up
    _ = tokenizer.encode(text[:1000])
    
    # Your tokenizer
    start = time.perf_counter()
    your_ids = tokenizer.encode(text)
    your_time = time.perf_counter() - start
    
    # Reference tokenizer
    start = time.perf_counter()
    ref_ids = reference_tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    ref_time = time.perf_counter() - start
    
    # Results
    print(f"\nInput size: {total_bytes:,} bytes ({total_bytes / 1024:.2f} KB)")
    print(f"\n{'Tokenizer':<20} {'Time (s)':<12} {'Throughput (KB/s)':<20} {'Tokens':<10}")
    print("-" * 60)
    print(f"{'Your implementation':<20} {your_time:<12.4f} {total_bytes / your_time / 1024:<20.2f} {len(your_ids):<10}")
    print(f"{'tiktoken (ref)':<20} {ref_time:<12.4f} {total_bytes / ref_time / 1024:<20.2f} {len(ref_ids):<10}")
    print(f"\nSpeedup factor: {ref_time / your_time:.2f}x")

if __name__ == "__main__":
    benchmark()