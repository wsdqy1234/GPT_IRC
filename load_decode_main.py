import json
from data.IRC_Poker import load_dataset
from utils.utils import encodings_decode

# Load encodings to form Dataset
save_path = "./data/irc_poker_dataset_new.pkl"
loaded_dataset = load_dataset(save_path)
loaded_encodings = loaded_dataset.encodings
loaded_attention_mask = loaded_dataset.attention_mask

print(loaded_encodings[-1000])
print(loaded_attention_mask[-1000])

# Try to Decode the encodings
context, action_history, next_action = encodings_decode(loaded_dataset.tokenizer, loaded_encodings[-1000])

print("#context#: ", context)
print("#action_history#: ", action_history)
print("#next_action#: ", next_action)


# Try to Decode json files
def json_read(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    context_history_list = []
    next_action_list = []
    pred_next_action_list = []
    
    for item in data:
        context_history_list.append(item["context_history"])
        next_action_list.append(item["next_action"])
        pred_next_action_list.append(item["pred_next_action"])
    
    return context_history_list, next_action_list, pred_next_action_list


def json_decode(tokenizer, json_path):
    context_history_list, next_action_list, pred_next_action_list = json_read(json_path)
    
    n = len(context_history_list)
    context_decode = []
    action_history_decode = []
    next_action_decode = []
    pred_next_action_decode = []
    
    for i in range(n):
        
        # Extract content before pad token
        tmp_context_history = context_history_list[i]
        first_pad_idx = tmp_context_history.index(tokenizer.word2idx[tokenizer.pad_token])
        extracted_context_history = tmp_context_history[:first_pad_idx]
        
        # Separate extracted_context_history into "context" and "action_history" (two parts)
        eos_con_idx = extracted_context_history.index(tokenizer.word2idx[tokenizer.con_eos_token])
        extracted_context = extracted_context_history[:eos_con_idx]
        extracted_history = extracted_context_history[eos_con_idx+1:]
        
        # Context Decoding
        context_tokens = [tokenizer.idx2word.get(idx, '<unk>') for idx in extracted_context]
        context = "".join(token for token in context_tokens if (token != tokenizer.pad_token) and (token != tokenizer.act_eos_token) and (token != tokenizer.con_eos_token))
        context_decode.append(context.strip())
        
        # Action History Decoding
        action_history_tokens = [tokenizer.idx2word.get(idx, '<unk>') for idx in extracted_history]
        action_history = "".join(token for token in action_history_tokens if (token != tokenizer.pad_token) and (token != tokenizer.act_eos_token) and (token != tokenizer.con_eos_token))
        action_history_decode.append(action_history.strip())
    
        # Next Action Decoding
        next_action_tokens = [tokenizer.idx2word.get(idx, '<unk>') for idx in next_action_list[i]]
        next_action = "".join(token for token in next_action_tokens if (token != tokenizer.pad_token) and (token != tokenizer.act_eos_token) and (token != tokenizer.con_eos_token))
        next_action_decode.append(next_action.strip())
        
        # Predicted Next Action Decoding
        pred_next_action_tokens = [tokenizer.idx2word.get(idx, '<unk>') for idx in pred_next_action_list[i]]
        pred_next_action = "".join(token for token in pred_next_action_tokens if (token != tokenizer.pad_token) and (token != tokenizer.act_eos_token) and (token != tokenizer.con_eos_token))
        pred_next_action_decode.append(pred_next_action.strip())
    
    return context_decode, action_history_decode, next_action_decode, pred_next_action_decode


def json_write_decode(save_path, context_decode, action_history_decode, next_action_decode, pred_next_action_decode):
    data = []
    for context, action_history, next_action, pred_next_action in zip(context_decode, action_history_decode, next_action_decode, pred_next_action_decode):
        data.append({
            "context": context.strip(),
            "action_history": action_history.strip(),
            "next_action": next_action.strip(),
            "pred_next_action": pred_next_action.strip()
        })
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)




json_path = "./1000.json"
context_decode, action_history_decode, next_action_decode, pred_next_action_decode = json_decode(loaded_dataset.tokenizer, json_path)

json_save_path = "./1000_decode.json"
json_write_decode(json_save_path, context_decode, action_history_decode, next_action_decode, pred_next_action_decode)