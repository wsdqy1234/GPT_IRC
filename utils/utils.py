import json
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
from mingpt.model import GPT



def hand2list(json_path):
    """
    Input: json_path (once a file)
    return: hand = [h1, h2, ..., hn], h1 = {dict}
    """
    hand = []
    with open(json_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            hand.append(data)
    return hand


def act_history2list(action_history_dict):
    """
    Input: action_history_dict
    Return: action_history = [act1, act2, ..., actn]
    """
    action_history = [action_history_dict[str(i)] for i in range(1, len(action_history_dict)+1)]
    return action_history


def hand_parser(hand_list):
    
    
    return


class Hand_Dataset(Dataset):
    """ Dataset for Texas Hold'em in IRC Database 

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        
        
        
    
    def __len__(self):
        return
    
    def __getitem__(self, idx):
        
        return super().__getitem__(idx)
    
    def get_vocab_size(self):
        
        return 1
    
    def get_block_size(self):
        
        return 2
    

# class Hand_DataLoader(DataLoader):
    
    
    """
    先想想GPT应该怎么训练，输入/输出 分别是什么，
    """


if __name__ == "__main__":
    set_seed(3407)
    
    h = hand2list("hands_valid_1.json")
    print(h[0])