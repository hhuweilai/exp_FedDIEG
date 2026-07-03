import os
import torch
import pandas as pd
import pickle
import random
import numpy as np  # <--- 新增导入
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. 专供 非序列模型 (MF, GPFedRec) 使用的训练 Dataset
# ==========================================
class MF_TrainDataset(Dataset):
    def __init__(self, train_df, num_items):
        self.users = train_df['userId'].values
        self.pos_items = train_df['itemId'].values
        self.num_items = num_items
        self.user_items_map = train_df.groupby('userId')['itemId'].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]

        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_items_map[user]:
                break

        return torch.tensor(user, dtype=torch.long), \
            torch.tensor(pos_item, dtype=torch.long), \
            torch.tensor(neg_item, dtype=torch.long)


# ==========================================
# 2. 专供 序列模型 (SASRec, Ours) 使用的训练 Dataset
# ==========================================
class Seq_TrainDataset(Dataset):
    def __init__(self, user_df, num_items, max_len=50):
        self.num_items = num_items
        self.max_len = max_len
        # 【关键】将所有 itemId + 1，为了让 0 可以作为 Padding 填充符
        self.items = user_df['itemId'].values + 1
        self.user_items_set = set(self.items)

    def __len__(self):
        return 1  # 在本地训练中，一个用户的数据我们视为一条完整的序列数据

    def __getitem__(self, idx):
        seq = np.zeros([self.max_len], dtype=np.int64)
        pos = np.zeros([self.max_len], dtype=np.int64)
        neg = np.zeros([self.max_len], dtype=np.int64)
        nxt = self.items[-1]
        idx_pos = self.max_len - 1

        # 倒序遍历用户的历史记录，构建滑动窗口式的自回归序列
        for i in reversed(self.items[:-1]):
            seq[idx_pos] = i
            pos[idx_pos] = nxt
            while True:
                neg_item = random.randint(1, self.num_items)
                if neg_item not in self.user_items_set:
                    break
            neg[idx_pos] = neg_item
            nxt = i
            idx_pos -= 1
            if idx_pos == -1: break

        return torch.tensor(seq, dtype=torch.long), \
            torch.tensor(pos, dtype=torch.long), \
            torch.tensor(neg, dtype=torch.long)


# ==========================================
# 3. 专供 非序列模型 100% 通用的评测 Dataset
# ==========================================
class EvalDataset(Dataset):
    def __init__(self, eval_df, candidates_dict):
        self.eval_data = []
        for _, row in eval_df.iterrows():
            user = int(row['userId'])
            pos_item = int(row['itemId'])
            neg_items = candidates_dict[user]
            items = [pos_item] + neg_items
            self.eval_data.append((user, items))

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        user, items = self.eval_data[idx]
        return torch.tensor(user, dtype=torch.long), \
            torch.tensor(items, dtype=torch.long)


# ==========================================
# 4. 专供 序列模型 评测的 Dataset
# ==========================================
class Seq_EvalDataset(Dataset):
    def __init__(self, train_df, eval_df, candidates_dict, max_len=50):
        self.eval_data = []
        # 构建每个用户的历史序列字典 (注意 +1 对齐 padding)
        user_hist = train_df.groupby('userId')['itemId'].apply(lambda x: list(x + 1)).to_dict()

        for _, row in eval_df.iterrows():
            user = int(row['userId'])
            pos_item = int(row['itemId']) + 1
            # 负样本候选集也需要 +1
            neg_items = [i + 1 for i in candidates_dict[user]]
            items = [pos_item] + neg_items

            hist = user_hist.get(user, [])
            seq = np.zeros([max_len], dtype=np.int64)
            idx_pos = max_len - 1
            # 用用户在 train 中的历史行为填充满这 50 个槽位
            for i in reversed(hist):
                seq[idx_pos] = i
                idx_pos -= 1
                if idx_pos == -1: break

            self.eval_data.append((user, seq, items))

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        user, seq, items = self.eval_data[idx]
        return torch.tensor(user, dtype=torch.long), \
            torch.tensor(seq, dtype=torch.long), \
            torch.tensor(items, dtype=torch.long)


# 统一的 DataLoader 工厂函数
def get_dataloaders(dataset_name="movielens-100k", batch_size=64, model_type="MF"):
    base_dir = "datasets"
    splits_dir = os.path.join(base_dir, "splits", dataset_name)
    processed_dir = os.path.join(base_dir, "processed", dataset_name)

    import json
    with open(os.path.join(processed_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    num_users = meta["num_users"]
    num_items = meta["num_items"]

    train_df = pd.read_csv(os.path.join(splits_dir, "loo_train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "loo_val.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "loo_test.csv"))

    with open(os.path.join(splits_dir, "eval_candidates.pkl"), "rb") as f:
        candidates_dict = pickle.load(f)

    client_train_dataloaders = {}
    for user, user_df in train_df.groupby('userId'):
        if model_type == "MF":
            client_dataset = MF_TrainDataset(user_df, num_items)
        elif model_type == "SASRec":
            client_dataset = Seq_TrainDataset(user_df, num_items, max_len=50)

        client_train_dataloaders[user] = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

    if model_type == "MF":
        val_dataset = EvalDataset(val_df, candidates_dict)
        test_dataset = EvalDataset(test_df, candidates_dict)
    elif model_type == "SASRec":
        val_dataset = Seq_EvalDataset(train_df, val_df, candidates_dict, max_len=50)
        test_dataset = Seq_EvalDataset(train_df, test_df, candidates_dict, max_len=50)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return client_train_dataloaders, val_loader, test_loader, num_users, num_items