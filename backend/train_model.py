import tensorflow as tf
import os
import json

# Set memory growth to avoid GPU memory issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Parameters
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 2

# Load the dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    tf.keras.utils.get_file(
        'cats_vs_dogs',
        'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
        extract=True,
        cache_dir='.'
    ) + '/cats_and_dogs_filtered/train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    tf.keras.utils.get_file(
        'cats_vs_dogs',
        'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
        extract=True,
        cache_dir='.'
    ) + '/cats_and_dogs_filtered/validation',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Print class names and indices
class_names = train_ds.class_names
print("Class names:", class_names)
print("Class indices - 0:", class_names[0], "1:", class_names[1])

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

# Create the model with data augmentation
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Train the model
initial_epochs = 10
print("\nInitial training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs
)

# Fine-tune the model
print("\nFine-tuning the model...")
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Continue training
fine_tune_epochs = 5
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs
)

# Save the class mapping
class_mapping = {i: name for i, name in enumerate(class_names)}
print("\nClass mapping:", class_mapping)

# Save the model and class mapping
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cat_dog_model')
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, 'model.keras'))

# Save class mapping to a file
with open(os.path.join(model_dir, 'class_mapping.json'), 'w') as f:
    json.dump(class_mapping, f)

print("Model and class mapping saved successfully!") 