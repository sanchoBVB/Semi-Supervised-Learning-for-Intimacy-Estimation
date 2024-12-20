from sklearn.utils import shuffle


def train2batch(title, category, batch_size):
    title_batch = []
    category_batch = []
    title_shuffle, category_shuffle = shuffle(title, category, random_state=0)
    for i in range(0, len(title), batch_size):
        title_batch.append(title_shuffle[i:i+batch_size])
        category_batch.append(category_shuffle[i:i+batch_size])
    return title_batch, category_batch


def not2batch(text, aug1, aug2, utt, utt1, utt2, data_split):
    text_batch = []
    aug1_batch = []
    aug2_batch = []
    utt_batch = [[] for _ in range(data_split)]
    utt1_batch = [[] for _ in range(data_split)]
    utt2_batch = [[] for _ in range(data_split)]
    shuffle_list = shuffle(list(range(len(aug1))), random_state=0)
    for i in shuffle_list:
        text_batch.append([text[i]])
        aug1_batch.append([aug1[i]])
        aug2_batch.append([aug2[i]])
        for s in range(data_split):
            utt_batch[s].append([utt[s][i]])
            utt1_batch[s].append([utt1[s][i]])
            utt2_batch[s].append([utt2[s][i]])
    return text_batch, aug1_batch, aug2_batch, utt_batch, utt1_batch, utt2_batch


def test2batch(ind, title, category, batch_size):
    index_batch = []
    title_batch = []
    category_batch = []
    index_shuffle, title_shuffle, category_shuffle = shuffle(ind, title, category, random_state=0)
    for i in range(0, len(title), batch_size):
        index_batch.append(index_shuffle[i:i+batch_size])
        title_batch.append(title_shuffle[i:i+batch_size])
        category_batch.append(category_shuffle[i:i+batch_size])
    return index_batch, title_batch, category_batch