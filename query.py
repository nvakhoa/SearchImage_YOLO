import numpy as np, re, math, time
import cv2
import inverted_YOLO as i2v

name_object = 'data/openimages.names'
path_image =  'D:/data/coco/val2017/'
i2v.set_dir_path(path_image)


def combine(vector, invert):
    temp = []
    for index in range(len(vector)):
        if vector[index] != 0:
            temp.extend(invert[index])
    return list(set(temp))


def negate(arr, number_file):
    temp = [i
            for i in range(number_file)
            if i not in arr]
    return list(temp)


def special(str):
    begin = re.search(r"\"\b", str)
    end = re.search(r"\"\b", str[:][:][::-1])
    if begin != None: begin = begin.end()
    if end != None:  end = len(str) - end.start() - 1
    return (begin, end)


def vector_intersect(vector, invert):
    arr = np.arange(1, len(vector), 1)
    for index in range(len(vector)):
        if vector[index] != 0:
            arr = np.intersect1d(np.array(arr), np.array(invert[index]))
    return arr


def find_tf(index, file):
    global invert, tf

    line_invert = invert[index]
    line_tf = tf[index]

    if file in line_invert:
        return line_tf[line_invert.index(file)]
    else:
        return 0


def Dist_question(Query, ListFile):
    global voc, n_voc

    start = time.time()
    tf_query = np.zeros(n_voc)

    for word in range(n_voc):
        if voc[word] in Query:
            count = Query.count(voc[word])
            tf_query[word] = 1 + math.log(count)

    dist = dict()
    for file in range(len(ListFile)):
        d = 0.0
        for i in range(n_voc):
            d += ((tf_query[i] - tf[ListFile[file]][i])) ** 2
        dist[ListFile[file]] = math.sqrt(d)

    dist = sorted(dist, key=dist.get)

    t = time.time() - start
    return dist, t


def show_image(pathImage):
    f = cv2.imread(pathImage, 1)
    cv2.imshow('img', f)
    cv2.waitKey()
    cv2.destroyAllWindows()


def distance(vector_query, vectors_file, listID):
    Norm2 = dict()
    for i in listID:
        Norm2[i] = np.linalg.norm(vector_query - vectors_file[i])
    Norm2 = sorted(Norm2, key=Norm2.get)
    return Norm2


def Text2vector(search_query, voc):
    vector_query = np.zeros(len(voc), dtype=np.int)
    for index in range(len(voc)):
        if voc[index] in search_query:
            vector_query[index] = search_query.count(voc[index])
    return vector_query


def query_decode(search_query, vectors_file, voc, inverted):
    spe = special(search_query)
    vector_query = Text2vector(search_query, voc)
    if spe[0] != None and spe[1] != None:
        vector = Text2vector(search_query[spe[0]: spe[1]], voc)
        plus = vector_intersect(vector, inverted)
        if search_query[spe[0] - 2] == '-':
            plus = negate(plus, len(vectors_file))
    else:
        plus = combine(vector_query, inverted)

    return distance(vector_query, vectors_file, plus)


if __name__ == '__main__':
    tic = time.time()
    print('loading...')
    f = open(name_object, 'r')
    voc = [i.strip() for i in f.readlines()]
    f.close()

    filenames, vectorfiles, invert = i2v.load_name_vector_invert()
    vectorfiles = np.array(vectorfiles)
    n_voc, n_files = len(voc), len(filenames)
    print(n_voc, n_files)
    print('time to load: ', round(time.time() - tic, 3))
    query = ''

    k = 5

    while True:
        query = input('Input query: ')
        tic = time.time()
        if query == '--exit':
            break
        else:
            ranking = query_decode(query.lower(), vectorfiles, voc, invert)
            N_find = len(ranking)
            print('Có ', N_find, ' kết quả (', round(time.time() - tic, 2), 's)', sep='')
            for i in range(N_find):
                if i > k: break
                print(path_image + filenames[ranking[i]])
                show_image(path_image + filenames[ranking[i]])
