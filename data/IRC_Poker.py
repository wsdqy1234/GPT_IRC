import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from langchain import PromptTemplate
from transformers import GPT2Tokenizer

from utils.utils import act_history2list, load_hands


# Action Template
ROLE_LIST = ['small blind', 'big blind', 'position 1']
STAGE_LIST = ['preflop', 'flop', 'turn', 'river', 'showdown']
ACTION_LIST = ['no action', 'blind bet', 'fold', 'check', 'bet', 'call', 'raise', 'all-in', 'quits game', 'kicked from game']


CONTEXT_TEMPLATE = """\
Context Information: This is a Texas Hold'em game with {num_players} players. You are playing in the role of <{role}> at position {position} on the board. Each player's bankroll is {bankroll}. The current community cards on the board are {board}, and your private cards are {pocket_cards}.
"""
CONTEXT_TEMPLATE = PromptTemplate.from_template(CONTEXT_TEMPLATE)


# ACTION_TEMPLATE = """\
# At {stage} stage, player <{role}> adopted action <{action}>
# """

# ACTION_HISTORY_TEMPLATE = """\
# Historical Actions: 
# """

def save_dataset(dataset, save_path):
    """
    Save dataset.encodings and dataset.attention_mask to save_path
    """
    data_to_save = {
        "encodings": dataset.encodings,
        "attention_mask": dataset.attention_mask
    }
    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)


def load_dataset(file_path, tokenizer=None, train_flag=True):
    """
    Load dataset.encodings and recover IRC_Poker_Dataset
    """
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    # Recover dataset
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2') if tokenizer is None else tokenizer
    dataset = IRC_Poker_Dataset(tokenizer=tokenizer, train_flag=train_flag)
    dataset.encodings = loaded_data["encodings"]
    dataset.attention_mask = loaded_data["attention_mask"]
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
            self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2') if tokenizer is None else tokenizer
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.encodings = None
            self.attention_mask = None
        else:
            # Read Data and Construct Dataset
            self.train_flag = train_flag
            self.tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2') if tokenizer is None else tokenizer
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            context_list = []
            action_history_list = []
            next_action_list = []

            # Prepare the data
            all_hands = load_hands(dataset_root, max_hands)
            for sample in all_hands:
                context = sample["context"]
                action_history_dict = sample["action_history"]
                action_history = act_history2list(action_history_dict)  # Convert to list once and store
                next_action = sample["next_action"]
                
                # Context use Prompt Template
                context_prompt = CONTEXT_TEMPLATE.format(
                    num_players = context["num_players"],
                    role = context["role"],
                    position = context["position"],
                    bankroll = context["bankroll"],
                    board = context["board"],
                    pocket_cards = context["pocket_cards"]
                )
                
                context_list.append(context_prompt)
                action_history_list.append(str(action_history))
                next_action_list.append(str(next_action))
            
            # Batch Tokenize
            flag_attention_mask = True # Padding -> mask: 0
            flag_padding = True # Default: padding_side="right"
            
            encoded_contexts = self.tokenizer(
                context_list,
                padding=flag_padding,
                return_attention_mask=flag_attention_mask)
            
            encoded_action_histories = self.tokenizer(
                action_history_list,
                padding=flag_padding,
                return_attention_mask=flag_attention_mask)
            
            encoded_next_actions = self.tokenizer(
                next_action_list,
                padding=flag_padding,
                return_attention_mask=flag_attention_mask)
            
            # Now, populate self.encodings
            self.encodings = [
                {
                    "encoded_context": encoded_context,
                    "encoded_action_history": encoded_action_history,
                    "encoded_next_action": encoded_next_action
                }
                for encoded_context, encoded_action_history, encoded_next_action in zip(encoded_contexts["input_ids"], encoded_action_histories["input_ids"], encoded_next_actions["input_ids"])
            ]
            
            self.attention_mask = [
                {
                    "atten_mask_context": atten_mask_context,
                    "atten_mask_action_history": atten_mask_action_history,
                    "atten_mask_next_action": atten_mask_next_action
                }
                for atten_mask_context, atten_mask_action_history, atten_mask_next_action in zip(encoded_contexts["attention_mask"], encoded_action_histories["attention_mask"], encoded_next_actions["attention_mask"])
            ]


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