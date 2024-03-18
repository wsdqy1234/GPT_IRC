import json
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Tokenizer

from utils.utils import act_history2list, load_hands



# Action Template
ROLE_LIST = ['small blind', 'big blind', 'position 1']
STAGE_LIST = ['preflop', 'flop', 'turn', 'river', 'showdown']
ACTION_LIST = ['no action', 'blind bet', 'fold', 'check', 'bet', 'call', 'raise', 'all-in', 'quits game', 'kicked from game']



class IRC_Poker_Dataset(Dataset):
    """
    Dataset for Texas Hold'em in IRC Database 
    Read the Parsed Hands
    """
    def __init__(self, dataset_root=None, max_hands=0, tokenizer=None, train_flag=True):
        self.train_flag = train_flag
        
        # json_path = data_root
        # hand = hand2list(json_path)
        all_hands = load_hands(dataset_root, max_hands)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2') if tokenizer is None else tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data = all_hands
        self.encodings = []
        
        # Assuming self.tokenizer.encode can handle batch inputs efficiently
        context_list = []
        action_history_list = []
        next_action_list = []

        # Prepare the data
        for sample in self.data:
            context = sample["context"]
            action_history_dict = sample["action_history"]
            action_history = act_history2list(action_history_dict)  # Convert to list once and store
            next_action = sample["next_action"]
            
            context_list.append(str(context))
            action_history_list.append(str(action_history))
            next_action_list.append(str(next_action))
        
        # Batch Tokenize
        encoded_contexts = self.tokenizer(context_list, return_attention_mask=False)
        encoded_action_histories = self.tokenizer(action_history_list, return_attention_mask=False)
        encoded_next_actions = self.tokenizer(next_action_list, return_attention_mask=False)
        
        # Now, populate self.encodings
        self.encodings = [
            {
                "encoded_context": encoded_context,
                "encoded_action_history": encoded_action_history,
                "encoded_next_action": encoded_next_action
            }
            for encoded_context, encoded_action_history, encoded_next_action in zip(encoded_contexts["input_ids"], encoded_action_histories["input_ids"], encoded_next_actions["input_ids"])
        ]


        
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        encoded_context = self.encodings[idx]['encoded_context']
        encoded_action_history = self.encodings[idx]['encoded_action_history']
        encoded_next_action = self.encodings[idx]['encoded_next_action']

        return encoded_context, encoded_action_history, encoded_next_action