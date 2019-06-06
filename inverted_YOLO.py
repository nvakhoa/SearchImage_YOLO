import time
import numpy as np
import cv2, os
import darknet as dn
from cache import cache
path = 'D:/data/coco/val2017/'

def set_dir_path(newPath):
    global path
    path = newPath


def Image2Vector(thr):
    global path
    configPath = './cfg/yolov3-openimages.cfg'
    weightPath = './weight/yolov3-openimages.weights'
    metaPath = './cfg/openimages.data'
    filenames = os.listdir(path)
    text, vector_file = dn.performDetect(listPath=path,filenames=filenames, showImage=False, thresh=thr,
                                         weightPath=weightPath, metaPath=metaPath, configPath=configPath
                                         )

    a, b = vector_file.shape
    vector_object = vector_file.T
    invert = []
    for vt in vector_object:
        temp = []
        for index in range(a):
            if vt[index] != 0:
                temp.append(index)
        invert.append(temp)
    records = [(filenames[i], vector_file[i]) for i in range(a)]

    file_names, vector = zip(*records)
    f = open('data/List_name.txt','w')
    content = '\n'.join(filenames)
    f.write(content)


    f = open('data/vector_image.txt','w')
    content =''
    for i in vector:
        content += ' '.join(map(str,i)) + '\n'
    f.write(content)

    f = open('data/inverted.txt','w')
    content = '\n'.join(map(str,invert))
    f.write(content)




    return file_names, vector , invert


def load_name_vector_invert(thresh=0.25):
    cache_filename = 'records_file.ktp'
    records = cache(cache_path=cache_filename,
                    fn=Image2Vector,
                    thr=thresh)
    return records


if __name__ == '__main__':
    img = 'D:\source\PythonProject\project_IR\image/PUBG-New-Vehicle.jpg'
    _, vector = dn.performDetect(imagePath=img)
    print(vector)