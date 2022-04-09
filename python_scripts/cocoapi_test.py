from pathlib import Path

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import math
import json
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

def examine_data(coco, history_wh, history_hw, history_area):
    dictionary={}
    for cat in nms:
        catIds = coco.getCatIds(catNms=[cat])
        annIds = coco.getAnnIds(catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        dictionary[cat]={'area_segm' : 0, 'area_box': 0, 'ratio': 0, 'count': 0, 'id': None}
        for dict in anns:
            id = dict['category_id']
            area_segm = dict['area']
            bbox = dict['bbox']
            area_box = bbox[2]*bbox[3]
            try:
                ratio = bbox[2]/bbox[3]
            except:
                ratio = 0
            if ratio != 0:
                dictionary[cat]['area_segm'] += area_segm
                dictionary[cat]['area_box'] += area_box
                dictionary[cat]['ratio'] += ratio
                dictionary[cat]['count'] += 1
                inverseratio = bbox[3]/bbox[2]
                try:
                    history_wh[cat].append(ratio)
                    history_hw[cat].append(inverseratio)
                    history_area[cat].append(area_segm)
                except:
                    history_wh[cat] = []
                    history_hw[cat] = []
                    history_area[cat] = []
                    history_wh[cat].append(ratio)
                    history_hw[cat].append(inverseratio)
                    history_area[cat].append(area_segm)



        dictionary[cat]['id'] = id
    return dictionary

def merge_dictionaries():
    dictionary_by_class= {}
    dictionary_final ={'area_segm' : 0, 'area_box': 0, 'ratio': 0, 'count': 0}
    for cat in nms:
        dictionary_by_class[cat]={}
        count_tot = dictionary_train[cat]['count'] + dictionary_val[cat]['count']
        dictionary_by_class['count'] = count_tot
        dictionary_by_class[cat]['area_segm'] = (dictionary_train[cat]['area_segm'] + dictionary_val[cat]['area_segm']) / count_tot
        dictionary_by_class[cat]['area_box'] = (dictionary_train[cat]['area_box'] + dictionary_val[cat]['area_segm']) / count_tot
        dictionary_by_class[cat]['ratio'] = (dictionary_train[cat]['ratio'] + dictionary_val[cat]['ratio']) / count_tot

        mean_h = math.sqrt(dictionary_by_class[cat]['area_box'] / dictionary_by_class[cat]['ratio'])
        mean_w = mean_h * dictionary_by_class[cat]['ratio']
        dictionary_by_class[cat]['mean_w'] = mean_w
        dictionary_by_class[cat]['mean_h'] = mean_h

        dictionary_final['count'] += dictionary_train[cat]['count'] + dictionary_val[cat]['count']
        dictionary_final['area_segm'] += dictionary_train[cat]['area_segm'] + dictionary_val[cat]['area_segm']
        dictionary_final['area_box'] += dictionary_train[cat]['area_box'] + dictionary_val[cat]['area_box']
        dictionary_final['ratio'] += dictionary_train[cat]['ratio'] + dictionary_val[cat]['ratio']

    dictionary_final['area_segm'] /= dictionary_final['count']
    dictionary_final['area_box'] /= dictionary_final['count']
    dictionary_final['ratio'] /= dictionary_final['count']
    mean_h = math.sqrt(dictionary_final['area_box'] / dictionary_final['ratio'])
    mean_w = mean_h * dictionary_final['ratio']
    dictionary_final['mean_w'] = mean_w
    dictionary_final['mean_h'] = mean_h
    return dictionary_by_class, dictionary_final


history_wh = {}
history_hw = {}
history_area = {}

dataDir='coco'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
with open(annFile, "r") as f:
    d = json.load(f)

coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
dictionary_train = examine_data(coco, history_wh, history_hw, history_area)

dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
dictionary_val = examine_data(coco, history_wh, history_hw, history_area)

dictionary_by_class, dictionary_final = merge_dictionaries()

#print(dictionary_train['person'])
#print(dictionary_val['person'])
#print(dictionary_by_class['person'])
#print(dictionary_final)

dataset_folder = "../datasets/"

with open(f'{dataset_folder}wh.txt', 'w') as outfile:
    json.dump(history_wh, outfile)

with open(f'{dataset_folder}hw.txt', 'w') as outfile:
    json.dump(history_hw, outfile)

with open(f'{dataset_folder}area.txt', 'w') as outfile:
    json.dump(history_area, outfile)

print("Files correctly exported")

