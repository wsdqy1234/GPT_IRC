from utils.utils import is_valid_move
import re
import json


ROLE_LIST = ['<small_blind>', '<big_blind>']
MAX_NUM_PLAYERS = 22
for i in range(1, MAX_NUM_PLAYERS+1):
    ROLE_LIST.append('<pos_{}>'.format(str(i)))

STAGE_LIST = ['preflop', 'flop', 'turn', 'river', 'showdown']
ACTION_LIST = ['<no_action>', '<blind_bet>', 'fold', 'check', 'bet', 'call', 'raise', '<all_in>', '<quit>', '<kicked>']


def json_decode_read(json_decode_path):
    with open(json_decode_path, 'r') as file:
        data = json.load(file)
    
    context_list = []
    action_history_list = []
    next_action_list = []
    pred_next_action_list = []
    
    for item in data:
        context_list.append(item["context"])
        action_history_list.append(item["action_history"])
        next_action_list.append(item["next_action"])
        pred_next_action_list.append(item["pred_next_action"])
    
    return context_list, action_history_list, next_action_list, pred_next_action_list


json_decode_path = "./1000_decode.json"
context_list, action_history_list, next_action_list, pred_next_action_list = json_decode_read(json_decode_path)

n = len(context_list)
true_cnt = 0
for i in range(n):
    if is_valid_move(context_list[i], pred_next_action_list[i], action_history_list[i]):
        true_cnt += 1

print("Accuracy: {}".format(true_cnt/n))