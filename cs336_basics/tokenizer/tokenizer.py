from typing import Iterable
import regex as re

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        # self.merges = merges
        self.merges_priority = {mer: i for i, mer in enumerate(merges)}
        self.special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True) if special_tokens else []
        
        for special_token in self.special_tokens:
            if special_token.encode('utf-8') not in self.vocab.values():
                self.vocab[max(self.vocab.keys()) + 1] = special_token.encode('utf-8')
        
        self.token_to_id = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        import json
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            vocab = json.load(vf)
            vocab = {int(k): v.encode('utf-8') for k, v in vocab.items()}

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            for line in mf:
                parts = line.strip('\n').rsplit(' ', 1)
                if len(parts) == 2:
                    merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)
    
    def pre_tokenize(self, text: str) -> list[tuple[bytes]]:
        if self.special_tokens:
            pattern = f"({'|'.join(map(re.escape, self.special_tokens))})"   # 使用括号捕获特殊标记
            chunks = re.split(pattern, text)
        else:
            chunks = [text]
        pre_tokens = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                pre_tokens.append(chunk.encode('utf-8'))
            else:
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                pat = re.compile(PAT)
                for mtch in pat.finditer(chunk):
                    byte_seq = mtch.group(0).encode('utf-8')
                    byte_tuple = tuple([bytes([b]) for b in byte_seq])
                    pre_tokens.append(byte_tuple)
        return pre_tokens
    
    # def encode(self, text: str) -> list[int]:
    #     pre_tokens = self.pre_tokenize(text)
    #     token_ids = []
    #     for pre_token in pre_tokens:
    #         if pre_token in self.token_to_id:
    #             token_ids.append(self.token_to_id[pre_token])
    #         else:
    #             temp_pre_token = list(pre_token)
    #             for merge in self.merges:
    #                 if len(temp_pre_token) < 2:
    #                     break
    #                 new_word = []
    #                 i = 0
    #                 while i < len(temp_pre_token):
    #                     if i < len(temp_pre_token) - 1 and (temp_pre_token[i], temp_pre_token[i + 1]) == merge:
    #                         new_word.append(merge[0] + merge[1])
    #                         i += 2
    #                     else:
    #                         new_word.append(temp_pre_token[i])
    #                         i += 1
    #                 temp_pre_token = tuple(new_word)

    #             for token in temp_pre_token:
    #                 if token in self.token_to_id:
    #                     token_ids.append(self.token_to_id[token])
    #                 else:
    #                     token_ids.append(max(self.vocab.keys()) + 1)  # unknown token id
    #     return token_ids
    
    def encode(self, text: str) -> list[int]:
        pre_tokens = self.pre_tokenize(text)
        token_ids = []
        for pre_token in pre_tokens:
            if pre_token in self.token_to_id:
                token_ids.append(self.token_to_id[pre_token])
            else:
                temp_pre_token = list(pre_token)
                changed = True
                while changed and len(temp_pre_token) > 1:
                    changed = False

                    # 寻找最佳匹配
                    best_merge = None
                    best_idx = -1
                    best_priority = len(self.merges_priority)
                    for i in range(len(temp_pre_token) - 1):
                        pair = (temp_pre_token[i], temp_pre_token[i + 1])
                        if pair in self.merges_priority:
                            priority = self.merges_priority[pair]
                            if priority < best_priority:
                                best_priority = priority
                                best_merge = pair
                                best_idx = i
                    
                    if best_merge:
                        changed = True
                        # 执行合并
                        new_word = temp_pre_token[:best_idx] + [best_merge[0] + best_merge[1]] + temp_pre_token[best_idx + 2:]
                        temp_pre_token = new_word      

                for token in temp_pre_token:
                    token_ids.append(self.token_to_id.get(token, max(self.vocab.keys()) + 1))  # unknown token id
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        token_bytes = []
        for token_id in ids:
            if token_id not in self.vocab:
                token_bytes.append("<|unknown|>".encode('utf-8'))
            else:
                token_bytes.append(self.vocab[token_id])
        return b''.join(token_bytes).decode('utf-8', errors='replace')
    
    def half_decode(self, ids: list[int]) -> str:
        token_bytes = []
        for token_id in ids:
            if token_id not in self.vocab:
                token_bytes.append("<|unknown|>".encode('utf-8'))
            else:
                token_bytes.append(self.vocab[token_id])
        return token_bytes


if __name__ == "__main__":
    import tiktoken
    from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
    reference_tokenizer = tiktoken.get_encoding("gpt2")

    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="tests/fixtures/gpt2_vocab.json",
        merges_path="tests/fixtures/gpt2_merges.txt"
    )

    test_text = "Hello, world! This is a test."

    ids = tokenizer.encode(test_text)
    half_decoded = tokenizer.half_decode(ids)

    for i in range(256):
        print(i, tokenizer.vocab.get(i, "<|unknown|>").decode('utf-8', errors='replace'))



