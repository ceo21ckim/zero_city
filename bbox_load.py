import tqdm, os, glob, shutil
import requests, json
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
ABS_DIR = os.path.dirname(os.path.abspath(__file__))


# image path
JAYWALK_IMG_PATH = os.path.join(BASE_DIR, 'origin_data', 'images', 'jaywalks')
OTHER_IMG_PATH = os.path.join(BASE_DIR, 'origin_data', 'images', 'others')


# bboxes path
JAYWALK_BBOX_PATH = os.path.join(BASE_DIR, 'origin_data', 'bboxes', 'jaywalks')
OTHER_BBOX_PATH = os.path.join(BASE_DIR, 'origin_data', 'bboxes', 'others')

JAY_BBOX_TRAIN_PATH = os.path.join(os.path.split(JAYWALK_BBOX_PATH)[0], 'train', 'jaywalks')
JAY_BBOX_TEST_PATH = os.path.join(os.path.split(JAYWALK_BBOX_PATH)[0], 'test', 'jaywalks')

OHR_BBOX_TRAIN_PATH = os.path.join(os.path.split(OTHER_BBOX_PATH)[0], 'train', 'others')
OHR_BBOX_TEST_PATH = os.path.join(os.path.split(OTHER_BBOX_PATH)[0], 'test', 'others')




# dict
detection_labels = [
    '버스', '택시', '신호등', '사람', '표지판', '승용차', '트럭', '신호등_초록', '신호등_방향표시', '신호등_주황', 
    '신호등_빨강', '오토바이', '자전거', '기타_특수차량', '킥보드', '구급차', '소방차', 'Zero shuttle']

kor_to_eng = dict({
    '버스' : 'bus', 
    '택시' : 'taxi', 
    '신호등' : 'traffic',
    '사람' : 'person', 
    '표지판' : 'sign', 
    '승용차' : 'car', 
    '트럭' : 'truck', 
    '신호등_초록' : 'traffic_green', 
    '신호등_방향표시' : 'traffic_direction', 
    '신호등_주황' : 'traffic_orange', 
    '신호등_빨강' : 'traffic_red', 
    '오토바이' : 'motorcycle', 
    '자전거' : 'bicycle', 
    '기타_특수차량' : 'etc', 
    '킥보드' : 'kickboard', 
    '구급차' : 'ambulance', 
    '소방차' : 'fire_truck',
    'Zero shuttle' : 'zero_shuttle'
})

eng_to_kor = dict({})

for key in kor_to_eng.keys(): eng_to_kor[kor_to_eng[key]] = key


label_to_idx = dict([(label, idx) for idx, label in enumerate(detection_labels)])

def make_dir():
    if not os.path.exists(JAYWALK_BBOX_PATH):
        os.makedirs(JAYWALK_BBOX_PATH)
    
    if not os.path.exists(OTHER_BBOX_PATH):
        os.makedirs(OTHER_BBOX_PATH)



def save_bboxes(URL = 'http://apis.data.go.kr/C100006/zerocity/getCctvList/event/2DBoundingBox'):
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
    
    jaywalks_json = detect_df.loc[detect_df.loc[:,'event_type'] == '05', 'cctv_json_stor_flph'].tolist()
    others_json = detect_df.loc[detect_df.loc[:,'event_type'] == '07', 'cctv_json_stor_flph'].tolist()
    
    for j_json in tqdm.tqdm(jaywalks_json, desc = 'saving jaywalkers bbox'):
        file_name = os.path.split(j_json)[-1][:-5] + '.txt'

        response = requests.get(j_json)
        jaywalks_dict = json.loads(response.text)
        jaywalks_df = pd.json_normalize(jaywalks_dict['annotations'])
        jaywalks_df.loc[:,'label_id'] = jaywalks_df.loc[:,'label_name'].apply(lambda x: label_to_idx[x])

        for label, bbox in jaywalks_df.loc[:,['label_id', 'info']].values:
            x_center, y_center, width, height = bbox     

            with open(os.path.join(JAYWALK_BBOX_PATH, file_name), 'a') as f :
                f.write(f'{label} {x_center} {y_center} {width} {height} \n')


    for o_json in tqdm.tqdm(others_json, desc = 'saving others bbox'):
        file_name = os.path.split(o_json)[-1][:-5] + '.txt'

        response = requests.get(o_json)
        others_dict = json.loads(response.text)
        others_df = pd.json_normalize(others_dict['annotations'])
        others_df.loc[:,'label_id'] = others_df.loc[:,'label_name'].apply(lambda x: label_to_idx[x])

        for label, bbox in others_df.loc[:,['label_id', 'info']].values:
            x_center, y_center, width, height = bbox     

            with open(os.path.join(OTHER_BBOX_PATH, file_name), 'a') as f :
                f.write(f'{label} {x_center} {y_center} {width} {height} \n')

    print('complete!')

jaywalk_list = glob.glob(os.path.join(JAYWALK_BBOX_PATH, '*.txt'))
other_list = glob.glob(os.path.join(OTHER_BBOX_PATH, '*.txt'))


def dataset_split(jaywalk_list=jaywalk_list, other_list=other_list):

    jaywalk_train = jaywalk_list[:1400]
    jaywalk_test = jaywalk_list[1400:]

    other_train = other_list[:1648]
    other_test = other_list[1648:]

    if not os.path.exists(JAY_BBOX_TRAIN_PATH):
        os.makedirs(JAY_BBOX_TRAIN_PATH)

    if not os.path.exists(JAY_BBOX_TEST_PATH):
        os.makedirs(JAY_BBOX_TEST_PATH)

    if not os.path.exists(OHR_BBOX_TRAIN_PATH):
        os.makedirs(OHR_BBOX_TRAIN_PATH)

    if not os.path.exists(OHR_BBOX_TEST_PATH):
        os.makedirs(OHR_BBOX_TEST_PATH)

    for j_train_path in jaywalk_train:
        shutil.copy(j_train_path, JAY_BBOX_TRAIN_PATH)

    for j_test_path in jaywalk_test:
        shutil.copy(j_test_path, JAY_BBOX_TEST_PATH)

    for o_train_path in other_train:
        shutil.copy(o_train_path, OHR_BBOX_TRAIN_PATH)

    for o_test_path in other_test:
        shutil.copy(o_test_path, OHR_BBOX_TEST_PATH)
    
    return print('complete to make train_test_datasets!')


if __name__ == '__main__':
    save_bboxes()

    dataset_split()