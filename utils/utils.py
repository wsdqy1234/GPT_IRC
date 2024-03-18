import os
import json
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
from mingpt.model import GPT



# def hand2list(json_path):
#     """
#     Input: json_path (once a file)
#     return: hand = [h1, h2, ..., hn], h1 = {dict}
#     """
#     hand = []
#     with open(json_path, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             hand.append(data)
#     return hand


def act_history2list(action_history_dict):
    """
    Input: action_history_dict
    Return: action_history = [act1, act2, ..., actn]
    """
    action_history = [action_history_dict[str(i)] for i in range(1, len(action_history_dict)+1)]
    return action_history


def load_hands(dataset_path, max_hands):
    """
    Input:  Parsed Dataset Path and 
    Return: max_hands*10000 hands in the Dataset
            hand = [h1, h2, ..., hn], h1 = {dict}
    """
    all_hands = []
    
    for i, subdir in enumerate(os.listdir(dataset_path)):
        if (i+1) > max_hands: break
        
        subdir_path = os.path.join(dataset_path, subdir) # example: ./data/Parser/parser_1-10000
        
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):    # example: ./data/Parser/parser_1-10000/hands_valid_1.json
                if file.endswith(".json"):
                    json_path = os.path.join(subdir_path, file)
                    
                    with open(json_path, 'r') as f:
                        for line in f:
                            data = json.loads(line)
                            all_hands.append(data)
    
    return all_hands
    


# class Hand_DataLoader(DataLoader):
    
    
    """
    先想想GPT应该怎么训练，输入/输出 分别是什么，
    """


if __name__ == "__main__":
    set_seed(3407)
    
    h = hand2list("hands_valid_1.json")
    print(h[0])