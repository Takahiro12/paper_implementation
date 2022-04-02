import pickle, os
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def translate(file):
    x = unpickle(file)
    datas = x[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8') 
    labels = x[b'labels']
    filenames = x[b'filenames']
    tgt_dir = './dataset'
    if (not os.path.isdir(tgt_dir)):
        os.mkdir(tgt_dir)

    infos = []
    for label, data, filename in zip(labels, datas, filenames):
        info = [label, os.path.join(tgt_dir, filename.decode(encoding='utf-8'))]
        infos.append(info)
        img = Image.fromarray(data)
        img.save(os.path.join(tgt_dir, filename.decode(encoding='utf-8')))
    
    return infos


if __name__ == '__main__':
    infos_1 = translate('./data/cifar-10-batches-py/data_batch_1')
    infos_2 = translate('./data/cifar-10-batches-py/data_batch_2')
    infos_3 = translate('./data/cifar-10-batches-py/data_batch_3')
    infos_4 = translate('./data/cifar-10-batches-py/data_batch_4')
    infos_5 = translate('./data/cifar-10-batches-py/data_batch_5')
    infos_t = translate('./data/cifar-10-batches-py/test_batch')

    train_info = infos_1 + infos_2 + infos_3 + infos_4 + infos_5
    train_info, valid_info = train_test_split(train_info, train_size=0.6)

    test_info = infos_t

    label_names = unpickle('./data/cifar-10-batches-py/batches.meta')[b'label_names']
    meta_info = [[idx, name] for idx, name in enumerate(label_names)]

    if (not os.path.isdir('./info')):
        os.mkdir('./info')

    with open('./info/train_info.pkl', 'wb') as f:
        pickle.dump(train_info, f)

    with open('./info/valid_info.pkl', 'wb') as f:
        pickle.dump(valid_info, f)

    with open('./info/test_info.pkl', 'wb') as f:
        pickle.dump(test_info, f)

    with open('./info/meta_info.pkl', 'wb') as f:
        pickle.dump(meta_info, f)
    

