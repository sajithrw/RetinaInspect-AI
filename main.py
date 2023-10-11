from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# specify the directory containing the labeled subdirectories
image_directory = 'dataset/'

# data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    fill_mode='nearest'
    # rescale=1./255,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest'
)

batch_size = 64
train_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = VGG16(weights='imagenet', include_top=False)

# custome layers for classification task
layer = base_model.output
layer = GlobalAveragePooling2D()(layer)
layer = Dense(1024, activation='relu')(layer)
predictions = Dense(5, activation='softmax')(layer) #5 classes for DR levels

# compile and train model with dataset
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

num_epochs = 20
model.fit(train_generator, epochs=num_epochs)

# save trained model
model.save('DR_detection_model_2.h5')
