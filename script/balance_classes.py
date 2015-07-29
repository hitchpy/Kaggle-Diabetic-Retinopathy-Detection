import pandas as pd
import os
import multiprocessing 


def split_index(row):
    # split row to 6 processes for parallel processing
    res = []
    chunk = row/6
    for i in range(1, 6):
        res.append([(i-1)*chunk, i*chunk])
    res.append([5*chunk, row])
    return res

def helper(part):
    res = {'image':[], 'level': []}
    for i in range(part[0], part[1]):
        if labels.level[i] != 0:
            lev = labels.level[i]
            for j in range(aug_dict[lev]):
                os.system("cp ~/diabetic/input/processed/run-normal/train/" + labels.image[i] + '.jpeg ~/diabetic/input/processed/run-normal/train/' + labels.image[i] + str(j) + '.jpeg')
                res['image'].append(labels.image[i] + str(j))
                res['level'].append(lev)
    return res

if __name__ == "__main__":
    labels = pd.read_csv('../input/trainLabels.csv')
    parts = split_index(labels.shape[0])
    x = labels.level.value_counts()
    n_train = labels.shape[0] * 0.2
    aug_dict = {}

    for i in x.index:
        if i != 0:
            aug_dict[i] = int(round(n_train /x.ix[i]))
    
    pool = multiprocessing.Pool(processes = 6)
    print "Augmenting data"
    jobs = [pool.apply(helper, args=(x,)) for x in parts]
    
    temp = jobs.pop()
    for it in jobs:
        for key, values in it.items():
            temp[key].extend(values)
    aug = pd.DataFrame.from_dict(temp)
    labels = pd.concat([labels, aug])
    labels = labels.reset_index(drop = True)
    labels.to_csv("trainLabels.csv", index = False)
    
     


    