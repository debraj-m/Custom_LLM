import re
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

import re
from collections import Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_counts = Counter()
    def train(self, texts):
        for text in texts:
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            self.word_counts.update(words)
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        self.word_to_id = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }
        self.id_to_word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        for i, (word, _) in enumerate(most_common):
            self.word_to_id[word] = i + 4
            self.id_to_word[i + 4] = word
    def encode(self, text):
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return [self.word_to_id.get(word, 1) for word in words]
    def decode(self, ids):
        words = []
        for id in ids:
            word = self.id_to_word.get(id, '<unk>')
            if word not in ['<pad>', '<bos>', '<eos>']:
                words.append(word)
        return ' '.join(words)

class SubwordTokenizer:
    def __init__(self, vocab_size=5000):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
        self.decoder = decoders.ByteLevel()
        self.tokenizer.decoder = self.decoder
    def train(self, texts):
        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids)
