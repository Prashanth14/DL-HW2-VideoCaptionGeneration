import math
import operator
import sys
import json
from functools import reduce 
def countNgram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculating the precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Building dictionary of n_gram counts
        for ref in references:
            ref_sentence = ref[si]
            n_gram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentence by considering the n_gram length
            for i in range(limits):
                n_gram = ' '.join(words[i:i+n]).lower()
                if n_gram in n_gram_d.keys():
                    n_gram_d[n_gram] += 1
                else:
                    n_gram_d[n_gram] = 1
            ref_counts.append(n_gram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            n_gram = ' '.join(words[i:i + n]).lower()
            if n_gram in cand_dict:
                cand_dict[n_gram] += 1
            else:
                cand_dict[n_gram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each n_gram by considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(s,t,flag = False):

    score = 0.  
    count = 0
    candidate = [s.strip()]
    if flag:
        references = [[t[i].strip()] for i in range(len(t))]
    else:
        references = [[t.strip()]] 
    precisions = []
    pr, bp =countNgram(candidate, references, 1)
    precisions.append(pr)
    score = geometric_mean(precisions) * bp
    return score
### Usage: python bleu_eval.py caption.txt
### Ref : https://github.com/vikasnar/Bleu
if __name__ == "__main__" :
    test = json.load(open('/home/menishe/Manish/MLDS_hw2_data/testing_label.json','r'))
    output = sys.argv[1]
    result = {}
    with open(output,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    #count by average
    bleu=[]
    for item in test:
        score_per_video = []
        for caption in item['caption']:
            caption = caption.rstrip('.')
            score_per_video.append(BLEU(result[item['id']],caption))
        bleu.append(sum(score_per_video)/len(score_per_video))
    average = sum(bleu) / len(bleu)
    ##>=0.25
    print("Originally, average bleu score is " + str(average))
    #count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
    bleu=[]
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    ## >= 0.65
    print("By another method, average bleu score is " + str(average))



