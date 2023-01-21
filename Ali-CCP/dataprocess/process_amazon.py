import gzip
from collections import defaultdict
from datetime import datetime
import numpy as np
import random
import sys

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
if __name__ == '__main__':
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    dataset_name = sys.argv[1]
    raw_file = 'reviews_' + dataset_name + '_5.json.gz'
    f = open('reviews_' + dataset_name + '.txt', 'w')
    for l in parse(raw_file):
        line += 1
        f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        countU[rev] += 1
        countP[asin] += 1
    f.close()

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    for l in parse(raw_file):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        rating = float(l['overall'])
        if countU[rev] < 5 or countP[asin] < 5:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
            usernum += 1
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemid = itemnum
            itemmap[asin] = itemid
            itemnum += 1
        User[userid].append([itemid, rating])
    # sort reviews in User according to time

    data_out = []
    for userid in User.keys():
        review_len = len(User[userid])
        pos_set = [e[0] for e in User[userid]]
        for i in range(review_len):
            is_like = 1.0 if User[userid][i][1] >=4.0 else 0.0
            data_out.append((1.0, is_like, userid, User[userid][i][0]))
            neg = random_neq(0, itemnum-1, pos_set)
            data_out.append((0.0,0.0,userid,neg))
            
    random.shuffle(data_out)     
    total_size = len(data_out)   
    print (usernum, itemnum)

    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    i = 0
    with open('%s_train.txt' % dataset_name, 'w') as f:
        while i < train_size:
            f.write('%.1f,%.1f,%d,%d\n'%(data_out[i][0], data_out[i][1], data_out[i][2], data_out[i][3]))
            i = i + 1

    with open('%s_val.txt' % dataset_name, 'w') as f:
        while i < train_size + val_size:
            f.write('%.1f,%.1f,%d,%d\n'%(data_out[i][0], data_out[i][1], data_out[i][2], data_out[i][3]))
            i = i + 1
            
    with open('%s_test.txt' % dataset_name, 'w') as f:
        while i < total_size:
            f.write('%.1f,%.1f,%d,%d\n'%(data_out[i][0], data_out[i][1], data_out[i][2], data_out[i][3]))
            i = i + 1        