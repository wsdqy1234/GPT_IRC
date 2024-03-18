from data.IRC_Poker import IRC_Poker_Dataset

# test
data_root = "./hands_valid_1.json"

irc_poker = IRC_Poker_Dataset(data_root)


length = len(irc_poker)
encode_context, encode_action_history, encode_next_action = irc_poker[length-1]


print("Length: {}".format(length))
print("Context: {}".format(encode_context))
print("Action_history: {}".format(encode_action_history))
print("Next_action: {}".format(encode_next_action))