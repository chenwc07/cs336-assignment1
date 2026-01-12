import os
import time
import regex as re
from typing import BinaryIO

from joblib import Parallel, delayed

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class TrainBPE:
    def __init__(self, corpus_file: str, vocab_size: int, special_tokens: list[str]):
        self.corpus_file = corpus_file
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = self.init_vocab()
        self.merges = []

    def init_vocab(self) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}  # Initialize with byte values
        
        # add special tokens starting from index 256
        vocab_set = set(vocab.values())
        idx = 256
        for token in self.special_tokens:
            if token.encode('utf-8') in vocab_set:
                continue
            vocab[idx] = token.encode('utf-8')
            vocab_set.add(token.encode('utf-8'))
            idx += 1
        
        return vocab

    def run_pre_tokenization(self) -> dict[tuple[bytes], int]:
        with open(self.corpus_file, "rb") as f:
            num_processes = 16
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        def process_chunk(start, end):
            with open(self.corpus_file, "rb") as f_chunk:
                f_chunk.seek(start)
                chunk = f_chunk.read(end - start).decode("utf-8", errors="ignore")
            return self.pre_tokenization(chunk)
                
        # 使用joblib并行处理每个chunk
        token_counts_list = Parallel(n_jobs=num_processes)(
            delayed(process_chunk)(start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        )

        # Combine token counts from all chunks
        combined_counts = {}
        for token_counts in token_counts_list:
            for token, count in token_counts.items():
                combined_counts[token] = combined_counts.get(token, 0) + count

        return combined_counts

    def pre_tokenization(self, chunk: str) -> dict[tuple[bytes], int]:
        token_counts = {}
        sub_chunks = re.split('|'.join(map(re.escape, self.special_tokens)), chunk)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pat = re.compile(PAT)

        for sub_chunk in sub_chunks:
            for mtch in pat.finditer(sub_chunk):
                byte_seq = mtch.group(0).encode('utf-8')
                byte_tuple = tuple([bytes([b]) for b in byte_seq])  # for b in byte_seq会把b转成int，需要bytes([b])转回bytes
                token_counts[byte_tuple] = token_counts.get(byte_tuple, 0) + 1

        return token_counts
    
    def update_word(
            self, 
            word: tuple[bytes], 
            pair: tuple[bytes]
    ) -> tuple[tuple[bytes], list[tuple[bytes]], list[tuple[bytes]]]:
        new_token = pair[0] + pair[1]
        new_word = []

        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        old_pairs = [(word[j], word[j + 1]) for j in range(len(word) - 1)]
        new_pairs = [(new_word[k], new_word[k + 1]) for k in range(len(new_word) - 1)]

        return tuple(new_word), new_pairs, old_pairs

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # pre-tokenization
        time_start = time.time()
        word_freqs = self.run_pre_tokenization()
        time_end = time.time()
        # print(f"Pre-tokenization time: {time_end - time_start} seconds")

        # cal bytes pair frequencies
        time_start = time.time()
        pair_freqs = {}
        pair_index = {}

        for word, freq in word_freqs.items():  # calculate pair frequencies and pair index
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
                if pair not in pair_index:
                    pair_index[pair] = set()
                pair_index[pair].add(word)

        time_end = time.time()
        # print(f"Pair frequency calculation time: {time_end - time_start} seconds")


        time_start = time.time()
        # BPE merging
        while len(self.vocab) < self.vocab_size and pair_freqs:
            max_pair_freq = max(pair_freqs.values())
            max_pair_freq_pairs = [pair for pair, freq in pair_freqs.items() if freq == max_pair_freq]
            # choose the lexicographically max pair in case of tie
            pair = max(max_pair_freq_pairs, key=lambda x: (x[0], x[1]))

            freq = pair_freqs.pop(pair)

            # Add new token to vocab
            new_token = pair[0] + pair[1]
            self.merges.append((pair[0], pair[1]))
            self.vocab[len(self.vocab)] = new_token

            # Update word frequencies
            affected_words = pair_index.pop(pair, [])
            # print(f"Merging pair: {pair} with frequency: {freq}, affected words: {len(affected_words)}")
            for word in affected_words:
                word_freq = word_freqs.pop(word)
                new_word, new_pairs, old_pairs = self.update_word(word, pair)

                ## updates cache
                # word_freqs
                word_freqs[new_word] = word_freqs.get(new_word, 0) + word_freq

                # old_pairs
                for old_pair in old_pairs:
                    if old_pair == pair:
                        continue
                    pair_index[old_pair].discard(word)
                    pair_freqs[old_pair] = pair_freqs.get(old_pair, 0) - word_freq
                    if pair_freqs[old_pair] <= 0:
                        pair_freqs.pop(old_pair)
                        pair_index.pop(old_pair)

                # new_pairs
                for new_pair in new_pairs:
                    if new_pair not in pair_index:
                        pair_index[new_pair] = set()
                    pair_index[new_pair].add(new_word)
                    pair_freqs[new_pair] = pair_freqs.get(new_pair, 0) + word_freq

        time_end = time.time()
        # print(f"BPE merging time: {time_end - time_start} seconds")

        return self.vocab, self.merges


if __name__ == "__main__":
    import json

    ## tinystories
    # bpe = TrainBPE(
    #     corpus_file='data/TinyStoriesV2-GPT4-train.txt',
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"]
    # )
    # vocab, merges = bpe.train()
    
    # with open('data/tiny_story_train_vocab.json', 'w', encoding='utf-8') as f:
    #     json.dump({idx: token.decode('utf-8', errors='replace') for idx, token in vocab.items()}, f, ensure_ascii=False, indent=4)

    # with open('data/tiny_story_train_merges.txt', 'w', encoding='utf-8') as f:
    #     for pair in merges:
    #         f.write(f"{pair[0].decode('utf-8', errors='replace')} {pair[1].decode('utf-8', errors='replace')}\n")


    ## owt
    bpe = TrainBPE(
        corpus_file='data/owt_train.txt',
        vocab_size=32000,
        special_tokens=["<|endoftext|>"]
    )
    vocab, merges = bpe.train()
    
    with open('data/owt_train_train_vocab.json', 'w', encoding='utf-8') as f:
        json.dump({idx: token.decode('utf-8', errors='replace') for idx, token in vocab.items()}, f, ensure_ascii=False, indent=4)

    with open('data/owt_train_train_merges.txt', 'w', encoding='utf-8') as f:
        for pair in merges:
            f.write(f"{pair[0].decode('utf-8', errors='replace')} {pair[1].decode('utf-8', errors='replace')}\n")

    


            

                

        
        
            