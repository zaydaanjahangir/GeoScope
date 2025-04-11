import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import kagglehub

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
INITIAL_LR = 0.0001


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

        unique_classes = df['continent'].unique()
        num_classes = len(unique_classes)
        print("\nClass distribution:")
        print(df['continent'].value_counts())
        print(f"\nUnique classes: {unique_classes}")
        print(f"Number of classes: {num_classes}")

        # Compute class weights
        class_weights = compute_class_weight('balanced',
                                             classes=np.unique(df['continent_encoded']),
                                             y=df['continent_encoded'])
        class_weights = dict(enumerate(class_weights))
        print("\nClass weights:", class_weights)

        return df, label_encoder, num_classes, class_weights

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
        image = tf.image.random_brightness(image, max_delta=0.3)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.central_crop(image, np.random.uniform(0.8, 1.0))
        image = tf.image.resize(image, image_size)
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
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs), base_model


def plot_training_history(history):
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
    plt.savefig('training_history.png')
    plt.show()


def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'data', 'coordinates_with_continents_mapbox.csv')

    print(f"Script directory: {script_dir}")
    print(f"CSV path: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df, label_encoder, num_classes, class_weights = load_data(csv_path)
    print(f"\nSetting NUM_CLASSES to: {num_classes}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = create_data_pipeline(train_df, IMAGE_SIZE, BATCH_SIZE, is_training=True)
    val_dataset = create_data_pipeline(val_df, IMAGE_SIZE, BATCH_SIZE, is_training=False)

    model, base_model = create_model(IMAGE_SIZE + (3,), num_classes)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        INITIAL_LR, decay_steps=1000, decay_rate=0.96, staircase=True)

    model.compile(
        optimizer=optimizers.Adam(lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs('checkpoints', exist_ok=True)
    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        callbacks.ModelCheckpoint(
            'checkpoints/best_model.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        callbacks.ModelCheckpoint(
            'checkpoints/model_epoch_{epoch:02d}.keras',
            save_freq='epoch'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7
        ),
        callbacks.CSVLogger('training_log.csv'),
        callbacks.TensorBoard(log_dir='./logs')
    ]

    print("\nPhase 1: Training top layers...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        callbacks=callbacks_list,
        class_weight=class_weights
    )

    print("\nPhase 2: Fine-tuning base model...")
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(INITIAL_LR / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=10,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        class_weight=class_weights
    )

    model.save('continent_classifier_final.keras')
    print("\nModel saved as 'continent_classifier_final.keras'")

    plot_training_history(history)

    visualize_predictions(model, val_dataset, label_encoder)


def visualize_predictions(model, dataset, label_encoder):
    plt.figure(figsize=(15, 10))
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            pred_label = label_encoder.inverse_transform([np.argmax(predictions[i])])[0]
            true_label = label_encoder.inverse_transform([labels[i]])[0]
            plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
            plt.axis('off')
        plt.savefig('sample_predictions.png')
        plt.show()


if __name__ == '__main__':
    main()