from collections import Counter
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

train_loc = 'dataset/output/filtered/train/'
test_loc = 'dataset/output/filtered/val/'
target_size = (224, 224)

tr_data = ImageDataGenerator()
train_data = tr_data.flow_from_directory(
    directory=train_loc,
    target_size=target_size)
ts_data = ImageDataGenerator()
test_data = ts_data.flow_from_directory(
    directory=test_loc,
    target_size=target_size)

vgg16 = VGG16(weights='imagenet')
vgg16.summary()

x = vgg16.get_layer('fc2').output
prediction = Dense(5, activation='softmax', name='predictions')(x)

model = Model(inputs=vgg16.input, outputs=prediction)

for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-16:]:
    layer.trainable = True
    print("Layer '%s' is trainable" % layer.name)

opt = Adam(lr=0.000001)
model.compile(optimizer=opt, loss=categorical_crossentropy,
              metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint("DR_detection_model.h5", monitor='val_accuracy', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

counter = Counter(train_data.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}
print(class_weights)

hist = model.fit(train_data, steps_per_epoch=train_data.samples//train_data.batch_size, validation_data=test_data,
                 class_weight=class_weights, validation_steps=test_data.samples//test_data.batch_size,
                 epochs=80, callbacks=[checkpoint, early])

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.title('VGG16: Loss and Validation Loss (0.000001 = Adam LR)')
plt.legend()
plt.show()

plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='val')
plt.title('VGG16: Accuracy and Validation Accuracy (0.000001 = Adam LR)')
plt.legend()
plt.show()
