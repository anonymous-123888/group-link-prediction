from helper import *
from torch.utils.data import Dataset
import random


class TrainDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:	The triples used for training the model
    params:		Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, triples, params, target_index):
        self.triples = triples
        self.p = params
        self.target_index = target_index

    def __len__(self):
        return len(self.triples)

    # 如何取batch中的一条数据
    def __getitem__(self, idx):
        source,object = list(self.triples.keys())[idx]
        label = self.triples[(source,object)]
        source, label = torch.LongTensor([source]), np.int32(list(label))
        trp_label = self.get_label(label) # 格式为所有实体的one_hot向量

        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / len(self.target_index))

        return source, trp_label, None, None  # 返回(s\r\o), 及(s,r)对应的所有o的label

    @staticmethod
    # 如何取一个batch的数据
    def collate_fn(data):
        #print(data)
        source = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        return source, trp_label

    def get_label(self, label):
        if self.target_index is not None:
            y = np.zeros([len(self.target_index)], dtype=np.float32)
            # print('组织实体',self.target_index)
            # print('组织实体长度', len(self.target_index))
            # print('与e2有关的组织实体',label)
            # 如果有target_index，这里的label应该是表示属于target_index的第几个而不是所有实体列表中的第几个
            for e2 in label:
                targe_index = self.target_index.index(e2)
                y[targe_index] = 1.0
            # print('最终生产的标签', y)
            return torch.FloatTensor(y)
        else:
            y = np.zeros([self.p.num_ent], dtype=np.float32)
            for e2 in label: y[e2] = 1.0
            return torch.FloatTensor(y)

    def get_neg_ent(self, triple, label):
        def get(triple, label):
            pos_obj = label
            mask = np.ones([self.p.num_ent], dtype=np.bool)
            mask[label] = 0
            neg_ent = np.int32(
                np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
            neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))

            return neg_ent

        neg_ent = get(triple, label)
        return neg_ent


class TestDataset(Dataset):
    """
    Evaluation Dataset class.

    Parameters
    ----------
    triples:	The triples used for evaluating the model
    params:		Parameters for the experiments

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """

    def __init__(self, triples, params, target_index):
        self.triples = triples
        self.p = params
        self.target_index = target_index

    def __len__(self):
        return len(self.triples)

    # 如何取batch中的一条数据
    def __getitem__(self, idx):
        source, object = list(self.triples.keys())[idx]
        label = self.triples[(source,object)]
        source, object, label = torch.LongTensor([source]), torch.LongTensor([object]), np.int32(list(label))
        trp_label = self.get_label(label) # 格式为所有实体的one_hot向量

        return source, trp_label, object  # 返回(s\r\o), 及(s,r)对应的所有o的label

    @staticmethod
    # 如何取一个batch的数据
    def collate_fn(data):
        #print(data)
        source = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        object = torch.stack([_[2] for _ in data], dim=0)
        return source, trp_label, object

    def get_label(self, label):
        if self.target_index is not None:
            y = np.zeros([len(self.target_index)], dtype=np.float32)
            # print('组织实体',self.target_index)
            # print('组织实体长度', len(self.target_index))
            # print('与e2有关的组织实体',label)
            # 如果有target_index，这里的label应该是表示属于target_index的第几个而不是所有实体列表中的第几个
            for e2 in label:
                targe_index = self.target_index.index(e2)
                y[targe_index] = 1.0
            # print('最终生产的标签', y)
            return torch.FloatTensor(y)
        else:
            y = np.zeros([self.p.num_ent], dtype=np.float32)
            for e2 in label: y[e2] = 1.0
            return torch.FloatTensor(y)