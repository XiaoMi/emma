# -*- coding: utf-8 -*-
# Some functions come from the Internet, if you violate your rights, please contact us.
import os
from itertools import chain

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]",
#                   "<no_emotion>", "<happiness>", "<like>", "<sadness>", "<disgust>", "<anger>", "<fear>"]
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]", "[sad]", "[None]", "[Keywords]"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


class WBDataset(Dataset):

    def __init__(self, data, tokenizer, max_history=15, batch_first=True, lm_labels=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.lm_labels:
            history = self.data[index][-2 * self.max_history:-1]
            resposne = self.data[index][-1]
        else:
            history = self.data[index][-2 * self.max_history:-1]
            resposne = []
        return self.process(history, resposne)

    def process(self, history, resposne, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels


class DatasetBase(Dataset):

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data_files = list()
        self.data_files_offset = list()
        self.data_len = 0
        self._check_files()

    def _check_files(self):
        if self.data_path is None:
            raise RuntimeError("Data path cannot be \
                empty at same time.")

        if self.data_path:
            if not os.path.exists(self.data_path):
                raise RuntimeError("Training files does not exist at " + self.data_path)
            prepare_files_offset(self.data_path, self.data_files,
                                 self.data_files_offset)
            self.data_len = len(self.data_files_offset)

    def __len__(self):
        return self.data_len

    def _get_line(self, index):
        tup = self.data_files_offset[index]
        target_file = self.data_files[tup[0]]
        with open(target_file, "r", encoding="utf-8") as f:
            f.seek(tup[1])
            line = f.readline()
        return line


class WBdistDataset(DatasetBase):

    def __init__(self, tokenizer, max_history=15, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(WBdistDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        dialog = dialog.strip().split("\t")[:2]

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dialog = tokenize(dialog)
        history = dialog[:-1]
        candidates = dialog[-1]
        return self.process(history, candidates)

    def process(self, history, resposne, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels

class XiaoAiDataset(DatasetBase):

    def __init__(self, tokenizer, max_history=15, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(XiaoAiDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        dialog = dialog.strip().split("\t")

        def tokenize(obj):
            # print(obj)
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
                # if obj in SPECIAL_TOKENS:
                #     return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))[0]
                # else:
                #     return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        # dialog = tokenize(dialog)
        history = [tokenize(dialog[0])]
        candidates = tokenize(dialog[1])

        emotions = [tokenize(dialog[2])[0],tokenize(dialog[3])[0]]
        true_emotion = tokenize(dialog[3])[0]
        # print("history:", history)
        # print("candidates:",candidates)
        # print("emotions:", emotions)
        # print("true_emotion:", true_emotion)
        return self.process(history, candidates, emotions, true_emotion)

    def get_emotion_label(self,candidate_emotion):
        _, _, _, _, no_emotion_id, happiness_id, surprise_id, sadness_id, disgust_id, anger_id, fear_id, = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS)
        # print(" no_emotion_id, happiness_id, surprise_id, sadness_id, disgust_id, anger_id, fear_id:",  no_emotion_id, happiness_id, surprise_id, sadness_id, disgust_id, anger_id, fear_id)
        if candidate_emotion == happiness_id:
            return 0
        elif candidate_emotion == surprise_id:
            return 1
        elif candidate_emotion == sadness_id:
            return 2
        elif candidate_emotion == disgust_id:
            return 3
        elif candidate_emotion == anger_id:
            return 4
        elif candidate_emotion == fear_id:
            return 5
        elif candidate_emotion == no_emotion_id:
            return 6

    def process(self, history, resposne, emotions, true_emotion,with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
        """
        sequence = [[bos] + history[0]] + history[1:] + [resposne + ([eos] if with_eos else [])]
        sequence = [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence)]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["token_emotion_ids"] = [emotions[i] for i, s in enumerate(sequence[:-1]) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["mc_labels"] = self.get_emotion_label(true_emotion)
        """
        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["token_emotion_ids"] = [bos] + [emotions[i] for i,s in enumerate(sequence[1:-1]) for _ in s]
        instance['mc_token_ids'] = len(instance["input_ids"]) - 1
        instance['mc_labels'] = self.get_emotion_label(true_emotion)
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)

        token_emotion_ids = pad_sequence(
            [torch.tensor(instance["token_emotion_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad
        )
        mc_token_ids = torch.tensor([instance["mc_token_ids"] for instance in batch])
        # print("mc_labels:", [instance["mc_labels"] for instance in batch])
        mc_labels =torch.tensor([instance["mc_labels"] for instance in batch])

        return input_ids, token_type_ids, labels, token_emotion_ids, mc_token_ids, mc_labels

class MobanDataset(DatasetBase):

    def __init__(self, tokenizer, max_history=50, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(MobanDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        # print("old dialog:", dialog)
        dialog = dialog.strip().split("\t")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        query = dialog[0].split("+")
        query_text, query_label, query_keyword = query[0], query[1].strip(), query[2].strip()
        if query_keyword == "[None]":
            hasKeyword = "[False]"
        else:
            hasKeyword = "[True]"
        query_text = query_text + " [SEP] " + query_label + " [SEP] " + hasKeyword + " [SEP] " + query_keyword + " [SEP]"
        query = tokenize(query_text.split("|"))
        response = tokenize(dialog[-1])
        # print("query ids:", query)
        # print("response:", dialog[-1])
        # print("response_ids:", response)
        return self.process(query, response)

    def process(self, history, resposne, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        # print("seq1:", sequence)
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        # print("seq2:", sequence)
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        # print("input_ids:", instance['input_ids'])
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels

class MobanNoKeyDataset(DatasetBase):

    def __init__(self, tokenizer, max_history=50, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(MobanNoKeyDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        # print("old dialog:", dialog)
        dialog = dialog.strip().split("\t")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        query = dialog[0].split("+")
        query_text, query_label, query_keyword = query[0], query[1].strip(), query[2].strip()
        if query_keyword == "[None]":
            hasKeyword = "[False]"
        else:
            hasKeyword = "[True]"
        query_text = query_text + " [SEP] " + query_label +  " [SEP]"
        query = tokenize(query_text.split("|"))
        response = tokenize(dialog[-1])
        # print("query ids:", query)
        # print("response:", dialog[-1])
        # print("response_ids:", response)
        return self.process(query, response)

    def process(self, history, resposne, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        # print("seq1:", sequence)
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s 
                                    for i, s in enumerate(sequence[1:])]
        # print("seq2:", sequence)
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        # print("input_ids:", instance['input_ids'])
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels

class MobanNoLabelDataset(DatasetBase):

    def __init__(self, tokenizer, max_history=50, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(MobanNoLabelDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        # print("old dialog:", dialog)
        dialog = dialog.strip().split("\t")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        query = dialog[0].split("+")
        query_text, query_label, query_keyword = query[0], query[1].strip(), query[2].strip()
        if query_keyword == "[None]":
            hasKeyword = "[False]"
        else:
            hasKeyword = "[True]"
        query_text = query_text +  " [SEP] " + hasKeyword + " [SEP] " + query_keyword + " [SEP]"

        query = tokenize(query_text.split("|"))
        response = tokenize(dialog[-1])
        # print("query ids:", query)
        # print("response:", dialog[-1])
        # print("response_ids:", response)
        return self.process(query, response)

    def process(self, history, resposne, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        # print("seq1:", sequence)
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s 
                                    for i, s in enumerate(sequence[1:])]
        # print("seq2:", sequence)
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        # print("input_ids:", instance['input_ids'])
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels

class MobanNoAllDataset(DatasetBase):

    def __init__(self, tokenizer, max_history=50, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(MobanNoAllDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        # print("old dialog:", dialog)
        dialog = dialog.strip().split("\t")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        query = dialog[0].split("+")
        query_text, query_label, query_keyword = query[0], query[1].strip(), query[2].strip()
        if query_keyword == "[None]":
            hasKeyword = "[False]"
        else:
            hasKeyword = "[True]"
        query_text = query_text +  " [SEP]"

        query = tokenize(query_text.split("|"))
        response = tokenize(dialog[-1])
        # print("query ids:", query)
        # print("response:", dialog[-1])
        # print("response_ids:", response)
        return self.process(query, response)

    def process(self, history, resposne, with_eos=True):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:4])
        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        # print("seq1:", sequence)
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s 
                                    for i, s in enumerate(sequence[1:])]
        # print("seq2:", sequence)
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        # print("input_ids:", instance['input_ids'])
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels


def prepare_files_offset(path, files_list, offset_list):
    """Fill the file index and offsets of each line in files_list in offset_list
    Args:
        path: string of file path, support single file or file dir
        files_list: the list contains file names
        offset_list: the list contains the tuple of file name index and offset
    """
    if os.path.isdir(path):  # for multi-file, its input is a dir
        files_list.extend([os.path.join(path, f) for f in os.listdir(path)])
    elif os.path.isfile(path):  # for single file, its input is a file
        files_list.append(path)
    else:
        raise RuntimeError(path + " is not a normal file.")
    for i, f in enumerate(files_list):
        offset = 0
        with open(f, "r", encoding="utf-8") as single_file:
            for line in single_file:
                tup = (i, offset)
                offset_list.append(tup)
                offset += len(bytes(line, encoding='utf-8'))
