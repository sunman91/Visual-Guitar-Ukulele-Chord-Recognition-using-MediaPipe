from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

image_x, image_y = 200, 200
batch_size = 64
train_dir = "chords"


def keras_model(image_x, image_y):
    num_of_classes = 8
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='valid'))
    model.add(Conv2D(128, (5, 5), activation='relu')) #
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='valid')) #
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "guitar_learner.h5"
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

    model.save('guitar_learner.h5')

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
