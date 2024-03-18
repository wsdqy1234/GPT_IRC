from data.IRC_Poker import IRC_Poker_Dataset

# test
# data_root = "./hands_valid_1.json"
dataset_root = "./data/Parser"
max_hands = 1

irc_poker = IRC_Poker_Dataset(dataset_root, max_hands)


length = len(irc_poker)
encode_context, encode_action_history, encode_next_action = irc_poker[length-1]


print("Length: {}".format(length))
print("Context: {}".format(encode_context))
print("Action_history: {}".format(encode_action_history))
print("Next_action: {}".format(encode_next_action))