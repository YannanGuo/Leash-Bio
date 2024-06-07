
def generate_maccsfp(molecule):
    if molecule is None:
        return None
    return list(MACCSkeys.GenMACCSKeys(molecule))


def generate_ecfp(molecule, radius=4, bits=1024):
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))


class LeashDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

