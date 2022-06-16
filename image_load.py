import tqdm, os, glob, shutil, argparse
import urllib.request

import requests, json
import pandas as pd

from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
ABS_DIR = os.path.dirname(os.path.abspath(__file__))

# images path
JAYWALK_IMG_PATH = os.path.join(BASE_DIR, 'origin_data', 'images', 'jaywalks')
OTHER_IMG_PATH = os.path.join(BASE_DIR, 'origin_data', 'images', 'others')

JAY_IMG_TRAIN_PATH = os.path.join(os.path.split(JAYWALK_IMG_PATH)[0], 'train', 'jaywalks')
JAY_IMG_TEST_PATH = os.path.join(os.path.split(JAYWALK_IMG_PATH)[0], 'test', 'jaywalks')

OHR_IMG_TRAIN_PATH = os.path.join(os.path.split(OTHER_IMG_PATH)[0], 'train', 'others')
OHR_IMG_TEST_PATH = os.path.join(os.path.split(OTHER_IMG_PATH)[0], 'test', 'others')



def save_image(URL = 'http://apis.data.go.kr/C100006/zerocity/getCctvList/event/2DBoundingBox'):
    i = 1
    detect_df = pd.DataFrame()
    while True:
        params ={'serviceKey' : '2mFilBsibMy//cPnneUpOHu1krwdYECrJUS0v2jyGyBMtTh7joFCd08AMeV72hs9c+vW+uh/xSIhCrypUjq9VQ==',
                'type' : 'json', 'numOfRows' : '1000', 'pageNo' : f'{i}'}

        response = requests.get(URL, params=params)
        data_set = json.loads(response.content)
        if data_set[0]['cctvfileList'] == []:
            detect_df.reset_index(inplace=True, drop=True)
            break

        sample = pd.json_normalize(data_set[0]['cctvfileList'])
        detect_df = pd.concat([detect_df, sample], axis=0)
        i += 1
    
    jaywalks_img = detect_df.loc[detect_df.loc[:,'event_type'] == '05', 'image_flph'].tolist()
    others_img = detect_df.loc[detect_df.loc[:,'event_type'] == '07', 'image_flph'].tolist()

    # save jaywalks images, bboxes
    for url in tqdm.tqdm(jaywalks_img, desc='saving jaywalker images...'):
        if not os.path.isdir(JAYWALK_IMG_PATH):
            os.makedirs(JAYWALK_IMG_PATH)

        urllib.request.urlretrieve(url, os.path.join(JAYWALK_IMG_PATH, os.path.split(url)[-1]))


    # save others images, bboxes
    for url in tqdm.tqdm(others_img, desc='saving other images...'):
        if not os.path.isdir(OTHER_IMG_PATH):
            os.makedirs(OTHER_IMG_PATH)
        
        urllib.request.urlretrieve(url, os.path.join(OTHER_IMG_PATH, os.path.split(url)[-1]))

    print('complete!')


jaywalk_list = glob.glob(os.path.join(JAYWALK_IMG_PATH, '*.jpg'))
other_list = glob.glob(os.path.join(OTHER_IMG_PATH, '*.jpg'))

def dataset_split(jaywalk_list=jaywalk_list, other_list=other_list):

    jaywalk_train = jaywalk_list[:1400]
    jaywalk_test = jaywalk_list[1400:]

    other_train = other_list[:1648]
    other_test = other_list[1648:]

    if not os.path.exists(JAY_IMG_TRAIN_PATH):
        os.makedirs(JAY_IMG_TRAIN_PATH)

    if not os.path.exists(JAY_IMG_TEST_PATH):
        os.makedirs(JAY_IMG_TEST_PATH)

    if not os.path.exists(OHR_IMG_TRAIN_PATH):
        os.makedirs(OHR_IMG_TRAIN_PATH)

    if not os.path.exists(OHR_IMG_TEST_PATH):
        os.makedirs(OHR_IMG_TEST_PATH)

    for j_train_path in jaywalk_train:
        shutil.copy(j_train_path, JAY_IMG_TRAIN_PATH)

    for j_test_path in jaywalk_test:
        shutil.copy(j_test_path, JAY_IMG_TEST_PATH)

    for o_train_path in other_train:
        shutil.copy(o_train_path, OHR_IMG_TRAIN_PATH)

    for o_test_path in other_test:
        shutil.copy(o_test_path, OHR_IMG_TEST_PATH)
    
    return print('complete to make train_test_datasets!')




if __name__ == '__main__':
#    save_image()

    dataset_split()