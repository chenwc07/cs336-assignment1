import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

def encode_to_bin_streaming(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    chunk_lines: int = 50000,
    dtype=np.uint16,
):
    """
    Read input_path in chunks, batch encode and write to output_path (binary continuous uint16).
    """
    # If resumable, check the byte size of existing output_path
    start_token_count = 0

    # Open output file in append mode
    fout = output_path.open("ab")  # append in binary
    
    total_written = start_token_count

    # First, count total lines for tqdm
    print(f"Counting lines in {input_path}...")
    with input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with input_path.open("r", encoding="utf-8") as fin:
        lines = []
        # Wrap file iterator with tqdm for progress visualization
        for line in tqdm(fin, total=total_lines, desc=f"Encoding {input_path.name}", unit=" lines"):
            lines.append(line)
            if len(lines) >= chunk_lines:
                # Batch encoding
                encodings = tokenizer.encode_batch(lines)
                for enc in encodings:
                    ids = enc.ids
                    if ids:
                        arr = np.array(ids, dtype=dtype)
                        arr.tofile(fout)
                        total_written += arr.size
                lines = []

        # Handle remaining lines in the last chunk
        if lines:
            encodings = tokenizer.encode_batch(lines)
            for enc in encodings:
                ids = enc.ids
                if ids:
                    arr = np.array(ids, dtype=dtype)
                    arr.tofile(fout)
                    total_written += arr.size

    fout.close()
    print(f"Finished encoding. Total tokens written: {total_written}")
    return total_written

def process_train_val(
    tokenizer_path: Path,
    train_input: Path,
    val_input: Path,
    output_dir: Path,
    chunk_lines: int = 50000,
):
    # load tokenizer
    print(f"Loading tokenizer from {tokenizer_path} ...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print("Tokenizer loaded, vocab size:", tokenizer.get_vocab_size())

    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train.bin"
    val_out = output_dir / "val.bin"

    print("Encoding train (streaming mode)...")
    cnt_train = encode_to_bin_streaming(tokenizer, train_input, train_out, chunk_lines=chunk_lines)
    print("Encoding val (streaming mode)...")
    cnt_val = encode_to_bin_streaming(tokenizer, val_input, val_out, chunk_lines=chunk_lines)

    print("Done. Train tokens:", cnt_train, "Val tokens:", cnt_val)


def main():
    # tokenizer_path = Path("hf_tokenizer/openwebtext-32k/tokenizer.json")
    # train_input_path = Path("data/owt_train.txt")
    # val_input_path = Path("data/owt_valid.txt")
    # output_dir = Path("data/openwebtext")
    
    tokenizer_path = Path("hf_tokenizer/tinystories-10k/tokenizer.json")
    train_input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    val_input_path = Path("data/TinyStoriesV2-GPT4-valid.txt")
    output_dir = Path("data/tinystories")

    # adjust your parameters
    # the number of lines to process in each chunk, 
    # adjust based on your machine memory / tokenizer performance
    chunk_lines = 100000

    process_train_val(
        tokenizer_path,
        train_input_path,
        val_input_path,
        output_dir,
        chunk_lines=chunk_lines
    )


if __name__ == "__main__":
    main()