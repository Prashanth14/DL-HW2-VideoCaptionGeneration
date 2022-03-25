# build dict #
# python sequence.py /Manish/MLDS_hw2_data/training_data/feat/ /home/menishe/Manish/MLDS_hw2_data/training_label.json 
import sys
from os import listdir
import numpy as np
import json
import pickle as pk


max_decoder_steps = 20

def build_bld_dict(sentences, n_min):
    n_word = {}
    n_seq = 0
    for sentence in sentences:
        n_seq += 1
        for word in sentence.lower().split(' '):
            n_word[word] = n_word.get(word, 0) + 1
    
    bld_dict = [word for word in n_word if n_word[word] >= n_min]
    print ('From %d words filtered %d words to dictionary with minimum count [%d]' % (len(n_word), len(bld_dict), n_min) ,'\n')

    keywordtrans = {}
    wordkeytrans = {}
    keywordtrans[0] = '<pad>'
    keywordtrans[1] = '<bos>'
    keywordtrans[2] = '<eos>'
    keywordtrans[3] = '<unk>'    
    wordkeytrans['<pad>'] = 0
    wordkeytrans['<bos>'] = 1
    wordkeytrans['<eos>'] = 2
    wordkeytrans['<unk>'] = 3

    for key, word in enumerate(bld_dict):
        wordkeytrans[word] = key + 4
        keywordtrans[key + 4] = word

    n_word['<pad>'] = n_seq
    n_word['<bos>'] = n_seq
    n_word['<eos>'] = n_seq
    n_word['<unk>'] = n_seq
    
    return wordkeytrans, keywordtrans, bld_dict

def pad_seqs(seqs, maxLen=None, padStr='pre', truncStr='pre', value=0):    
  
    len_seq = list()
    for se in seqs:
        len_seq.append(len(se))

    n_sample = len(seqs)
    if maxLen is None:
        maxLen = np.max(len_seq)

    seq_shape = tuple()
    for shp in seqs:
        if len(shp) > 0:
            seq_shape = np.asarray(shp).shape[1:]
            break

    padseq = (np.zeros((n_sample, maxLen) + seq_shape) * value).astype('int32')
    for k, se in enumerate(seqs):
        if not len(se):
            continue  # empty list/array was found
        if truncStr == 'pre':
            trunc = se[-maxLen:]
        elif truncStr == 'post':
            trunc = se[:maxLen]

        trunc = np.asarray(trunc, dtype='int32')
        if trunc.shape[1:] != seq_shape:
            raise ValueError('Shape of sample %s of sequence at position %s ''is different from expected shape %s' %(trunc.shape[1:], k, seq_shape))

        if padStr == 'post':
            padseq[k, :len(trunc)] = trunc
        elif padStr == 'pre':
            padseq[k, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padStr)
    return padseq
    
def filter_token(string):
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    for c in filters:
        string = string.replace(c,'')
    return string



if __name__ == "__main__":
    np.random.seed(2020)

    feat_folder = sys.argv[1]
    training_label_json = sys.argv[2]

    feat_filenames = listdir(feat_folder)
    feat_filepaths = [(feat_folder + filename) for filename in feat_filenames]

    
    vid_id = [filename[:-4] for filename in feat_filenames]# Remove '.avi' from filename

    dict_feat = {}
    for filepath in feat_filepaths:
        video_feat = np.load(filepath)
        video_ID = filepath[: -4].replace(feat_folder, "")
        dict_feat[video_ID] = video_feat
    
    video_caption = json.load(open(training_label_json, 'r'))
    dict_caption={}
    captions_corpus = list()
    for video in video_caption:
        filtered_captions = [filter_token(sentence) for sentence in video["caption"]]
        dict_caption[video["id"]] = filtered_captions
        captions_corpus += filtered_captions


    wordkeytrans, keywordtrans, bld_dict = build_bld_dict(captions_corpus, n_min=3)
    
    pk.dump(wordkeytrans, open('./wordkeytrans.obj', 'wb'))
    pk.dump(keywordtrans, open('./keywordtrans.obj', 'wb'))

    ID_caption = list()
    captions_words = list()

    words_list = list()
    for ID in vid_id:
        for caption in dict_caption[ID]:
            ID_caption.append((dict_feat[ID], caption))
            words = caption.split()
            captions_words.append(words)
            for word in words:
                words_list.append(word)

    caption_set = np.unique(words_list, return_counts=True)[0]
    max_captions_length = max([len(words) for words in captions_words])
    avg_captions_length = np.mean([len(words) for words in captions_words])
    num_unique_tokens_captions = len(caption_set)

    print("Caption dimenstion is:", np.shape(ID_caption))
    print("Caption's max length is:", max_captions_length)
    print("Average length of the captions:", avg_captions_length)
    print("Unique tokens:", num_unique_tokens_captions)
    
    print("ID of 11th video:", vid_id[10])
    print("Shape of features of 11th video:", ID_caption[10][0].shape)
    print("Caption of 11th video:", ID_caption[10][1])
   
    pk.dump(vid_id, open('vid_id.obj', 'wb'))
    pk.dump(dict_caption, open('dict_caption.obj', 'wb'))
    pk.dump(dict_feat, open('dict_feat.obj', 'wb'))
