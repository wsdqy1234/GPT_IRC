import ast
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from langchain import PromptTemplate
from transformers import GPT2Tokenizer

from utils.utils import act_history2list, load_hands
from data.IRC_tokenizer import IRCTokenizer


# Action Template
ROLE_LIST = ['small blind', 'big blind', 'position 1']
STAGE_LIST = ['preflop', 'flop', 'turn', 'river', 'showdown']
ACTION_LIST = ['no action', 'blind bet', 'fold', 'check', 'bet', 'call', 'raise', 'all-in', 'quits game', 'kicked from game']


# Delete bankroll to reduce the vocab size
# Delete other info
CONTEXT_TEMPLATE = """num_players:{num_players} role:{role} position:{position} board:{board} pocket_cards:{pocket_cards}"""
CONTEXT_TEMPLATE = PromptTemplate.from_template(CONTEXT_TEMPLATE)


def save_dataset(dataset, save_path):
    """
    Save dataset.encodings and dataset.attention_mask to save_path
    """
    data_to_save = {
        "encodings": dataset.encodings,
        "attention_mask": dataset.attention_mask,
        "tokenizer": dataset.tokenizer
    }
    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)


def load_dataset(file_path, tokenizer=None, train_flag=True):
    """
    Load dataset.encodings and recover IRC_Poker_Dataset
    """
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    dataset = IRC_Poker_Dataset()
    dataset.encodings = loaded_data["encodings"]
    dataset.attention_mask = loaded_data["attention_mask"]
    dataset.tokenizer = loaded_data["tokenizer"]
    return dataset


class IRC_Poker_Dataset(Dataset):
    """
    Dataset for Texas Hold'em in IRC Database 
    Read the Parsed Hands
    """
    def __init__(self, dataset_root=None, max_hands=0, tokenizer=None, train_flag=True):
        if not dataset_root:
            # Load Dataset from pkl
            self.train_flag = train_flag
            self.tokenizer = None
            self.encodings = None
            self.attention_mask = None
        else:
            # Read Data and Construct Dataset
            self.train_flag = train_flag
            self.tokenizer = tokenizer if tokenizer else IRCTokenizer()
            
            context_list = []
            action_history_list = []
            next_action_list = []

            # Prepare the data
            all_hands = load_hands(dataset_root, max_hands)
            for sample in all_hands:
                context = sample["context"]
                action_history_dict = sample["action_history"]
                action_history = act_history2list(action_history_dict)
                next_action = sample["next_action"]
                
                # Context use Prompt Template
                context_prompt = CONTEXT_TEMPLATE.format(
                    num_players = context["num_players"],
                    role = context["role"],
                    position = context["position"],
                    # bankroll = context["bankroll"],
                    board = context["board"],
                    pocket_cards = context["pocket_cards"]
                )
                context_prompt = context_prompt + self.tokenizer.con_eos_token
                
                context_list.append(context_prompt)
                
                action_history_norm = [f"{d['role']} {d['stage']} {d['action']} {self.tokenizer.act_eos_token}" for d in action_history]
                action_history_list.append(" ".join(action_history_norm))
                
                next_action_norm = [next_action["role"], next_action["stage"], next_action["action"], self.tokenizer.act_eos_token]
                next_action_list.append(" ".join(next_action_norm))
            
            ## 把字符串list拼接起来成为一个list，扔进去tokenizer，先建个表
            all = context_list + action_history_list + next_action_list
            self.tokenizer.build_vocab(all)
            
            # max_hands=1时的最大长度
            max_length_context = None
            max_length_act_his = None
            max_length_next_act = None
            
            encoded_contexts = self.tokenizer.batch_encode(context_list, max_length_context)
            encoded_action_histories = self.tokenizer.batch_encode(action_history_list, max_length_act_his)
            encoded_next_actions = self.tokenizer.batch_encode(next_action_list, max_length_next_act)
            
            self.encodings = [
                {
                    "encoded_context": encoded_context["input_ids"],
                    "encoded_action_history": encoded_action_history["input_ids"],
                    "encoded_next_action": encoded_next_action["input_ids"]
                }
                for encoded_context, encoded_action_history, encoded_next_action in zip(encoded_contexts, encoded_action_histories, encoded_next_actions)
            ]
            
            self.attention_mask = [
                {
                    "atten_mask_context": atten_mask_context["attention_mask"],
                    "atten_mask_action_history": atten_mask_action_history["attention_mask"],
                    "atten_mask_next_action": atten_mask_next_action["attention_mask"]
                }
                for atten_mask_context, atten_mask_action_history, atten_mask_next_action in zip(encoded_contexts, encoded_action_histories, encoded_next_actions)
            ]
            
            self.calculate_max_length()


    def __len__(self):
        return len(self.encodings)


    def __getitem__(self, idx):
        encoded_context = self.encodings[idx]['encoded_context']
        encoded_action_history = self.encodings[idx]['encoded_action_history']
        encoded_next_action = self.encodings[idx]['encoded_next_action']
        
        atten_mask_context = self.attention_mask[idx]['atten_mask_context']
        atten_mask_action_history = self.attention_mask[idx]['atten_mask_action_history']
        atten_mask_next_action = self.attention_mask[idx]['atten_mask_next_action']

        return encoded_context, encoded_action_history, encoded_next_action, atten_mask_context, atten_mask_action_history, atten_mask_next_action
    
    
    def calculate_max_length(self):
        max_length_context = 0
        max_length_act_his = 0
        max_length_next_act = 0
        
        for encoding in self.encodings:
            context_length = len(encoding["encoded_context"])
            max_length_context = max(max_length_context, context_length)
            
            action_history_length = len(encoding["encoded_action_history"])
            max_length_act_his = max(max_length_act_his, action_history_length)
            
            next_action_length = len(encoding["encoded_next_action"])
            max_length_next_act = max(max_length_next_act, next_action_length)
        
        self.max_length_context = max_length_context
        self.max_length_act_his = max_length_act_his
        self.max_length_next_act = max_length_next_act