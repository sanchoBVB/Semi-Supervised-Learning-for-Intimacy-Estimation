import MeCab
import sqlite3



def labeld_padding(datarow, tokenizer, max_len):
    index_datasets_title_tmp = []
    index_datasets_category = []

    for index, data in datarow.iterrows():
        data = list(data)
        index_title = tokenizer(data[0])["input_ids"]
        index_datasets_title_tmp.append(index_title)
        index_datasets_category.append(int(data[1]))

    index_datasets_title = []
    for title in index_datasets_title_tmp:
        if len(title) <= max_len:
            for i in range(max_len - len(title)):
                title.insert(0, 0)
        else:
            title = title[len(title)-max_len:]
        index_datasets_title.append(title)

    return index_datasets_title, index_datasets_category


def notlebeled_padding(datarow, tokenizer, data_split, max_len):
    index_datasets_text_tmp = []
    index_datasets_aug1_tmp = []
    index_datasets_aug2_tmp = []
    index_datasets_utt_tmp = [[] for _ in range(data_split)]
    index_datasets_utt1_tmp = [[] for _ in range(data_split)]
    index_datasets_utt2_tmp = [[] for _ in range(data_split)]

    for index, data in datarow.iterrows():
        data = list(data)
        index_text = tokenizer(data[0])["input_ids"]
        index_datasets_text_tmp.append(index_text)
        index_aug1 = tokenizer(data[1])["input_ids"]
        index_datasets_aug1_tmp.append(index_aug1)
        index_aug2 = tokenizer(data[2])["input_ids"]
        index_datasets_aug2_tmp.append(index_aug2)
        for i in range(data_split):
            index_utt = tokenizer(data[3+i])["input_ids"]
            index_datasets_utt_tmp[i].append(index_utt)
        for i in range(data_split):
            index_utt1 = tokenizer(data[3+data_split+i])["input_ids"]
            index_datasets_utt1_tmp[i].append(index_utt1)
        for i in range(data_split):
            index_utt2 = tokenizer(data[3+(data_split*2)+i])["input_ids"]
            index_datasets_utt2_tmp[i].append(index_utt2)

    index_datasets_text = []
    index_datasets_aug1 = []
    index_datasets_aug2 = []
    for text, aug1, aug2 in zip(index_datasets_text_tmp, index_datasets_aug1_tmp, index_datasets_aug2_tmp):
        if len(text) <= max_len:
            for _ in range(max_len - len(text)):
                text.insert(0, 0)
        else:
            text = text[len(text)-max_len:]
        index_datasets_text.append(text)

        if len(aug1) <= max_len:
            for _ in range(max_len - len(aug1)):
                aug1.insert(0, 0)
        else:
            aug1 = aug1[len(aug1)-max_len:]
        index_datasets_aug1.append(aug1)

        if len(aug2) <= max_len:
            for _ in range(max_len - len(aug2)):
                aug2.insert(0, 0)
        else:
            aug2 = aug2[len(aug2)-max_len:]
        index_datasets_aug2.append(aug2)

    index_datasets_utt = [[] for _ in range(data_split)]
    index_datasets_utt1 = [[] for _ in range(data_split)]
    index_datasets_utt2 = [[] for _ in range(data_split)]
    for i in range(data_split):
        for utt, utt1, utt2 in zip(index_datasets_utt_tmp[i], index_datasets_utt1_tmp[i], index_datasets_utt2_tmp[i]):
            if len(utt) <= max_len:
                for _ in range(max_len - len(utt)):
                    utt.insert(0, 0)
            else:
                utt = utt[len(utt)-max_len:]
            index_datasets_utt[i].append(utt)

            if len(utt1) <= max_len:
                for _ in range(max_len - len(utt1)):
                    utt1.insert(0, 0)
            else:
                utt1 = utt1[len(utt1)-max_len:]
            index_datasets_utt1[i].append(utt1)

            if len(utt2) <= max_len:
                for _ in range(max_len - len(utt2)):
                    utt2.insert(0, 0)
            else:
                utt2 = utt2[len(utt2)-max_len:]
            index_datasets_utt2[i].append(utt2)

    return index_datasets_text, index_datasets_aug1, index_datasets_aug2, index_datasets_utt, index_datasets_utt1, index_datasets_utt2
