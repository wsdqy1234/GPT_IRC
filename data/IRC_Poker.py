import json
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer

from utils.utils import hand2list, act_history2list



# Action Template
ROLE_LIST = ['small blind', 'big blind', 'position 1']
STAGE_LIST = ['preflop', 'flop', 'turn', 'river', 'showdown']
ACTION_LIST = ['no action', 'blind bet', 'fold', 'check', 'bet', 'call', 'raise', 'all-in', 'quits game', 'kicked from game']



class IRC_Poker_Dataset(Dataset):
    """
    Read the Parsed Hands from IRC Database
    """
    def __init__(self, data_root=None, tokenizer=None):
        json_path = data_root
        hand = hand2list(json_path)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2') if tokenizer is None else tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data = hand
        self.encodings = []
        
        for _, sample in enumerate(self.data):
            context = sample["context"]
            action_history_dict = sample["action_history"]
            action_history = act_history2list(action_history_dict) # Turn action_history into a list
            next_action = sample["next_action"]
            
            # Tokenizer + Embedding
            encode_context = self.tokenizer.encode(str(context))
            encode_action_history = self.tokenizer.encode(str(action_history))
            encode_next_action = self.tokenizer.encode(str(next_action))


            self.encodings.append({
                "encode_context": encode_context,
                "encode_action_history": encode_action_history,
                "encode_next_action": encode_next_action
            })
        
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        encode_context = self.encodings[idx]['encode_context']
        encode_action_history = self.encodings[idx]['encode_action_history']
        encode_next_action = self.encodings[idx]['encode_next_action']

        return encode_context, encode_action_history, encode_next_action