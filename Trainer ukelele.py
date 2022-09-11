from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import load_model

image_x, image_y = 200, 200
batch_size = 64
train_dir = "chords_ukelele"


def keras_model(image_x, image_y):
    model = load_model('guitar_learner_1.h5')
    filepath = "ukelele_learner_1.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def main():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=False,
        validation_split=0.2,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="training")

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="validation")

    no_epochs = 10

    model, callbacks_list = keras_model(image_x, image_y)
    history = model.fit(train_generator, epochs=no_epochs, validation_data=validation_generator)
    scores = model.evaluate_generator(generator=validation_generator, steps=64)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('ukelele_learner_1.h5')

    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,no_epochs  +1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('Train_Validation Loss.png')

    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    epochs = range(1,no_epochs+1)
    plt.plot(epochs, loss_train, 'g', label='Training Accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation Accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('Train_Validation Accuracy.png')


main()
