import glob, os, shutil
import tqdm
import pandas as pd
import numpy as np 

from sklearn.mixture import GaussianMixture

BASE_DIR = os.getcwd()

TRAIN_PATH = os.path.join(BASE_DIR, 'origin_data', 'bboxes', 'train', 'jaywalks')
TEST_PATH = os.path.join(BASE_DIR, 'origin_data', 'bboxes', 'test', 'jaywalks')

SAVE_TRAIN_PATH = os.path.join(BASE_DIR, 'preprocessed_data', 'train', 'jaywalks')
SAVE_TEST_PATH = os.path.join(BASE_DIR, 'preprocessed_data', 'test', 'jaywalks')


train_bbox_list = glob.glob(os.path.join(TRAIN_PATH, '*.txt'))
test_bbox_list = glob.glob(os.path.join(TEST_PATH, '*.txt'))


train = train_bbox_list[1000]

target = pd.read_csv(train, header = None, sep=' ').dropna(axis=1)
target.columns = ['label', 'x', 'y', 'w', 'h']
target_p = target.loc[target.loc[:, 'label'] == 3, ['x', 'y']].copy()

target_p

def preprocessing(target_p):
    target_p.loc[:,'x'] = target_p.loc[:,'x'].apply(lambda x: x / 1280)
    target_p.loc[:,'y'] = target_p.loc[:,'x'].apply(lambda x: x / 720)
    return target_p
    


# training
train_i = 0

_weight = 0.8

for train in tqdm.tqdm(train_bbox_list, desc = 'preprocessing...'):
    train_img_path = train.replace('bboxes', 'images').replace('txt', 'jpg')
    file_name = os.path.split(train_img_path)[-1]
    target = pd.read_csv(train, header = None, sep=' ').dropna(axis=1)
    target.columns = ['label', 'x', 'y', 'w', 'h']
    target = target.loc[target.loc[:, 'label'] == 3, ['x', 'y']].copy()
    target = preprocessing(target)

    
    if target.__len__() >2 :
        gmm = GaussianMixture(n_components=2, random_state=2)
        _ = gmm.fit(target)


    if np.abs(gmm.weights_[0] - gmm.weights_[1]) < _weight :
        train_i += 1
        _ = shutil.copy(train_img_path, os.path.join(SAVE_TRAIN_PATH, file_name))

    else: 
        pass 





# testing
test_i = 0 

for test in tqdm.tqdm(test_bbox_list, desc = 'preprocessing...'):
    test_img_path = test.replace('bboxes', 'images').replace('txt', 'jpg')
    file_name = os.path.split(test_img_path)[-1]
    target = pd.read_csv(test, header = None, sep=' ').dropna(axis=1)
    target.columns = ['label', 'x', 'y', 'w', 'h']
    target = target.loc[target.loc[:, 'label'] == 3, ['x', 'y']].copy()
    target = preprocessing(target)

    if target.__len__() >2 :
        gmm = GaussianMixture(n_components=2, random_state=2)
        _ = gmm.fit(target)


    if np.abs(gmm.weights_[0] - gmm.weights_[1]) < _weight :
        test_i += 1
        _ = shutil.copy(test_img_path, os.path.join(SAVE_TEST_PATH, file_name))

    else: 
        pass 

print(train_i, test_i )
# weight_ (0.8) -> 1378, 557
# weight_ (0.7) -> 1309, 486
# weight_ (0.6) -> 945, 393
# weight_ (0.5) -> 690, 340