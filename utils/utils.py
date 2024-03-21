import os
import re
import json
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
from mingpt.model import GPT


ROLE_LIST = ['<small_blind>', '<big_blind>']
MAX_NUM_PLAYERS = 22
for i in range(1, MAX_NUM_PLAYERS+1):
    ROLE_LIST.append('<pos_{}>'.format(str(i)))

STAGE_LIST = ['preflop', 'flop', 'turn', 'river', 'showdown']
ACTION_LIST = ['<no_action>', '<blind_bet>', 'fold', 'check', 'bet', 'call', 'raise', '<all_in>', '<quit>', '<kicked>']



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


def encodings_decode(tokenizer, encoding):
    encoded_context = encoding["encoded_context"]
    encoded_action_history = encoding["encoded_action_history"]
    encoded_next_action = encoding["encoded_next_action"]
    
    # Decoding without <pad> and <eos> token
    context_tokens = [tokenizer.idx2word.get(idx, '<unk>') for idx in encoded_context]
    context = "".join(token for token in context_tokens if (token != tokenizer.pad_token) and (token != tokenizer.act_eos_token) and (token != tokenizer.con_eos_token))
    
    action_history_tokens = [tokenizer.idx2word.get(idx, '<unk>') for idx in encoded_action_history]
    action_history = "".join(token for token in action_history_tokens if (token != tokenizer.pad_token) and (token != tokenizer.act_eos_token) and (token != tokenizer.con_eos_token))
    
    next_action_tokens = [tokenizer.idx2word.get(idx, '<unk>') for idx in encoded_next_action]
    next_action = "".join(token for token in next_action_tokens if (token != tokenizer.pad_token) and (token != tokenizer.act_eos_token) and (token != tokenizer.con_eos_token))
    
    return context.strip(), action_history.strip(), next_action.strip()


def parse_role(context):
    match = re.search(r'role:(<[^>]+>)', context)
    if match:
        extracted_role = match.group(1)
        return extracted_role
    else:
        return None


def parse_next_action(pred_next_action):
    next_role, next_stage, next_action = pred_next_action.split()
    return next_role, next_stage, next_action


def parse_action_history(action_history):
    pattern = re.compile(r'(<[^>]+> \w+ [^<]+)')
    actions = pattern.findall(action_history)
    
    role_history = []
    stage_history = []
    act_history = []
    
    for act in actions:
        role, stage, action = parse_next_action(act)
        role_history.append(role)
        stage_history.append(stage)
        act_history.append(action)
    
    return role_history, stage_history, act_history


def is_valid_move(context, pred_next_action, action_history):
    extracted_role = parse_role(context)
    next_role, next_stage, next_action = parse_next_action(pred_next_action)
    
    # The action must be one of the allowed actions
    if next_action not in ACTION_LIST:
        return False
    
    # The stage must be one of the defined stages
    if next_stage not in STAGE_LIST:
        return False
    
    # The role must be the same as the role in the context
    if extracted_role != next_role:
        return False
    
    # If action_history is empty, the next_stage must be 'preflop'
    if action_history == "":
        if next_stage != 'preflop':
            return False

        # On the "pre-flop", the first player to act can't "check";
        if next_action == "check":
            return False
    else:
        role_history, stage_history, act_history = parse_action_history(action_history)
        
        last_role, last_stage, last_action = role_history[-1], stage_history[-1], act_history[-1]
        
        # If the stages are not consecutive
        if STAGE_LIST.index(next_stage) - STAGE_LIST.index(last_stage) not in [0, 1]:
            return False
        
        # If the last action was a "bet" or "raise", the next action cannot be a "check".
        if last_action in ["bet", "raise"] and next_action == "check":
            return False
        
        # If there has been a "bet", the next player cannot "check" and must "fold", "call", or "raise".
        if last_action == "bet" and next_action == "check":
            return False

        # Note: More rules can be added here as needed.
    
    return True




if __name__ == "__main__":
    set_seed(3407)
    dataset_path = "./data/Parser"
    max_hands = 1
    a = load_hands(dataset_path, max_hands)
    print("hello")