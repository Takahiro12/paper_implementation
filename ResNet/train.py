import config
from generator import generate_dataset
import tensorflow as tf
from tensorflow.python.compiler.mlcompute import mlcompute
from tensorflow.python.framework.ops import disable_eager_execution
from models.resnet import Resnet18
import math
import time



def get_model():
    model = Resnet18()
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.summary()
    return model

def main():
    a = time.time()
    print('start: ', a)
    # GPUのセット
    gpu = mlcompute.is_apple_mlc_enabled()
    print('m1 mac is gpu available:', gpu)
    if (gpu):
        mlcompute.set_mlc_device(device_name='any') # 'any' or 'cpu' or 'gpu'
        

    train_dataset, train_count= generate_dataset('./info/train_info.pkl')
    valid_dataset, valid_count = generate_dataset('./info/valid_info.pkl')

    model = get_model()
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adadelta()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)
        valid_loss(v_loss)
        valid_accuracy(labels, predictions)
        
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()  
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch+1,
                                                                                    config.EPOCHS,
                                                                                    step,
                                                                                    math.ceil(train_count / config.BATCH_SIZE),
                                                                                    train_loss.result(),
                                                                                    train_accuracy.result()))
            if step % 10 == 0:
                print('elapsed time: ', time.time() - a)
        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train_loss: {:.5f}, train_accuracy: {:.5f}, valid_loss:{:.5f}, valid_accuracy: {:.5f}".format(
            epoch+1, config.EPOCHS, train_loss.result(), train_accuracy.result(), valid_loss.result(), valid_accuracy.result()
        ))
        print('elapsed time: ', time.time() - a)
    b = time.time()
    print('end: ', b)
    print('interval: ', b - a)
    model.save_weights(filepath=config.save_model_dir, save_format='tf')


if __name__ == '__main__':
    main()
    

    