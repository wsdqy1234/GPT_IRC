


class IRC_POKER:
    """
    Read the Parsed Hands from IRC Database
    """
    def __init__(self, data_root=None, num_hands=10000):
        self.num_hands = num_hands
        
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return