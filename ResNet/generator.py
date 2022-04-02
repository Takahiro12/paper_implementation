import pickle
import tensorflow as tf
import config

def preprocess_image(path):
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=config.channels)
    # img_tensor = tf.image.decode_jpeg(img_raw, channels=config.channels, expand_animations=False)
    img_tensor = tf.image.resize(img_tensor, [config.image_height, config.image_width])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor / 255.0
    return img_tensor

def generate_dataset(path):
    with open(path, 'rb') as f:
        objs = pickle.load(f)
    paths = []
    labels = []
    count = 0
    for label, path in objs:
        count += 1
        labels.append(label)
        paths.append(path)    

    path_dataset = tf.data.Dataset.from_tensor_slices(paths)
    img_dataset = path_dataset.map(preprocess_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((img_dataset, label_dataset))
    dataset =dataset.shuffle(buffer_size=count).batch(batch_size=config.BATCH_SIZE, drop_remainder=True)

    return dataset, count

