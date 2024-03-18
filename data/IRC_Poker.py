import json
import torch
from torch.utils.data import Dataset, DataLoader


from utils.utils import hand2list, act_history2list

# Action Template
ROLE_LIST = ['small blind', 'big blind', 'position 1']
STAGE_LIST = ['preflop', 'flop', 'turn', 'river', 'showdown']
ACTION_LIST = ['no action', 'blind bet', 'fold', 'check', 'bet', 'call', 'raise', 'all-in', 'quits game', 'kicked from game']

class IRC_Poker_Dataset(Dataset):
    """
    Read the Parsed Hands from IRC Database
    """
    def __init__(self, data_root=None):
        json_path = data_root
        hand = hand2list(json_path)
        
        self.data = hand
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Split
        context = sample["context"]
        action_history_dict = sample["action_history"]
        next_action = sample["next_action"]
        
        # Turn action_history into a list
        action_history = act_history2list(action_history_dict)
        
        return context, action_history, next_action


if __name__ == "__main__":
    import os
    print("Current working directory:", os.getcwd())
    
    # test
    data_root = "./hands_valid_1.json"
    
    irc_poker = IRC_Poker_Dataset(data_root)
    
    context, action_history, next_action = irc_poker[28]
    length = len(irc_poker)
    
    print("Length: {}".format(length))
    print("Context: {}".format(context))
    print("Action_history: {}".format(action_history))
    print("Next_action: {}".format(next_action))