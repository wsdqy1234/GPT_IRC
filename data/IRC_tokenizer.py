import re
from collections import Counter

from langchain import PromptTemplate

from utils.utils import act_history2list, load_hands


# Special seqs -> Special token
SPECIAL_SEQUENCES_TO_TOKEN = {
    # Role
    "small blind": "<small_blind>",
    "big blind": "<big_blind>",
    # Action
    "no action": "<no_action>",
    "blind bet": "<blind_bet>",
    "all-in": "<all_in>",
    "quits game": "<quit>",
    "kicked from game": "<kicked>",
}
MAX_NUM_PLAYERS = 22
for i in range(1, MAX_NUM_PLAYERS+1):
    SPECIAL_SEQUENCES_TO_TOKEN["position {}".format(str(i))] = "<pos_{}>".format(str(i))

# Special token -> Special seqs
SPECIAL_TOKEN_TO_SEQUENCES = {token: sequence for sequence, token in SPECIAL_SEQUENCES_TO_TOKEN.items()}

class IRCTokenizer:
    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size
        self.act_eos_token = "<eos-act>"    # Action Ending Token
        self.con_eos_token = "<eos-con>"    # Context Ending Token
        self.pad_token = "<pad>"            # Padding Token
        self.unknown_token = "<unk>"        # Unknown Token
        self.word2idx = {}
        self.idx2word = {}
        
    
    def tokenize(self, text):
        for seq, token in SPECIAL_SEQUENCES_TO_TOKEN.items():
            # text = [re.sub(r'\b' + re.escape(seq) + r'\b', token, t) for t in text]
            text = re.sub(r'\b' + re.escape(seq) + r'\b', token, text)
        
        # tokens = re.findall(r"\w+|[^\w\s]|\s", text) # 空格也算token
        # tokens = re.findall(r"\w+|[^\w\s]", text) # 空格不算token
        
        # tokens = [re.findall(r"<[^>]+>|\w+|[^\w\s]|\s", t) for t in text] # 匹配<>
        tokens = re.findall(r"<[^>]+>|\w+|[^\w\s]|\s", text) # 匹配<>
        return tokens
        
    
    def build_vocab(self, dataset):
        word_freq = Counter(word for text in dataset for word in self.tokenize(text))
        words = word_freq.most_common(self.vocab_size)
        
        self.word2idx = {word: idx+2 for idx, (word, _) in enumerate(words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # <unk> and <pad>
        self.word2idx[self.unknown_token] = 0
        self.word2idx[self.pad_token] = 1
        self.idx2word[0] = self.unknown_token
        self.idx2word[1] = self.pad_token
        
        self.vocab_size = len(self.word2idx)
        
        
    def encode(self, text, max_length=None):
        tokens = self.tokenize(text)
        token_ids = [self.word2idx.get(token, self.word2idx.get('<unk>')) for token in tokens]
        attention_mask = [1]*len(token_ids)
        
        # Padding to max_length
        if max_length:
            padding_length = max_length - len(token_ids)
            if padding_length > 0:
                token_ids += [self.word2idx[self.pad_token]] * padding_length
                attention_mask += [0] * padding_length
        
        return {"input_ids": token_ids, "attention_mask": attention_mask}


    def decode(self, indices):
        return ''.join(self.idx2word.get(idx, '<unk>') for idx in indices)

    
    def batch_encode(self, text_list, max_length=None):
        return [self.encode(text, max_length) for text in text_list]
    
    
    def batch_decode(self, indices_list):
        return [self.decode(indices) for indices in indices_list]
        