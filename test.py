from data.IRC_Poker import IRC_Poker_Dataset, save_dataset, load_dataset

# dataset_root = "./data/Parser"
# max_hands = 1
# irc_poker_dataset = IRC_Poker_Dataset(dataset_root, max_hands)

# Save Dataset encodings
save_path = "./data/irc_poker_dataset.pkl"
# save_dataset(irc_poker_dataset, save_path)

# Load encodings to form Dataset
loaded_dataset = load_dataset(save_path)
loaded_encodings = loaded_dataset.encodings
loaded_attention_mask = loaded_dataset.attention_mask

print(loaded_encodings[0])
print(loaded_attention_mask[0])

print("Finished")