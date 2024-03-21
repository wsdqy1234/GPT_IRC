from data.IRC_Poker import IRC_Poker_Dataset, save_dataset

# Initalized Dataset
dataset_root = "./data/Parser"
max_hands = 1
irc_poker_dataset = IRC_Poker_Dataset(dataset_root, max_hands)

# Save Dataset
save_path = "./data/irc_poker_dataset_new.pkl"
save_dataset(irc_poker_dataset, save_path)