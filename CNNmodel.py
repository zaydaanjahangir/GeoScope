import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import kagglehub

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 7
LEARNING_RATE = 0.001


def download_kaggle_dataset():
    print("Downloading dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("ayuseless/streetview-image-dataset")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def find_images_in_download(path, df):
    image_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    index = int(os.path.splitext(file)[0])
                    image_files.append((index, os.path.join(root, file)))
                except ValueError:
                    continue

    image_files.sort()
    image_paths = [path for _, path in image_files]

    print(f"\nFound {len(image_paths)} images in downloaded dataset")

    if len(image_paths) < len(df):
        print(f"Warning: More CSV entries ({len(df)}) than images ({len(image_paths)})")
        df = df.iloc[:len(image_paths)].copy()

    df['image_path'] = image_paths[:len(df)]

    print(f"Matched {len(df)} images to CSV rows")
    return df


def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print("\nData loaded successfully. First few rows:")
        print(df.head())

        kaggle_path = download_kaggle_dataset()

        df = find_images_in_download(kaggle_path, df)

        label_encoder = LabelEncoder()
        df['continent_encoded'] = label_encoder.fit_transform(df['continent'])

        print("\nClass distribution:")
        print(df['continent'].value_counts())

        return df, label_encoder

    except Exception as e:
        print(f"\nError in load_data(): {str(e)}")
        raise


def create_data_pipeline(df, image_size, batch_size, is_training=False):

    def parse_image(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size)
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image, label

    filenames = df['image_path'].values
    labels = df['continent_encoded'].values

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df))

    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_model(input_shape, num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'data', 'coordinates_with_continents_mapbox.csv')

    print(f"Script directory: {script_dir}")
    print(f"CSV path: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df, label_encoder = load_data(csv_path)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = create_data_pipeline(train_df, IMAGE_SIZE, BATCH_SIZE, is_training=True)
    val_dataset = create_data_pipeline(val_df, IMAGE_SIZE, BATCH_SIZE, is_training=False)

    model = create_model(IMAGE_SIZE + (3,), NUM_CLASSES)
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
        callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
    ]

    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks_list
    )

    model.save('continent_classifier.h5')
    print("\nTraining complete! Model saved as 'continent_classifier.h5'")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()


if __name__ == '__main__':
    main()