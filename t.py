import os
import tensorflow as tf
from tensorflow.keras import layers, applications
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Step 1: Enhanced Data Loading and Preprocessing
def load_data(train_dir, val_dir, img_size=(160, 160), batch_size=32):
    # Load datasets with potential resize for transfer learning
    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=True
    )
    
    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )

    class_names = train_ds.class_names
    print(f"Class Names: {class_names}")

    # Preprocessing for transfer learning
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

    # Performance optimizations
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

# Step 2: Transfer Learning Model with Fine-Tuning Capability
def build_model(input_shape=(160, 160, 3), num_classes=5):
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ])

    # Base model with transfer learning
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model initially

    # Custom head
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model, base_model

# Step 3: Enhanced Training with Fine-Tuning
def train_model(model, base_model, train_ds, val_ds, epochs=10):
    # Initial training phase
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2,
            patience=2,
            verbose=1
        )
    ]

    print("\nTraining initial model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Fine-tuning phase
    print("\nFine-tuning model...")
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False  # Only unfreeze last 4 layers

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=history.epoch[-1] + 1,
        callbacks=callbacks
    )

    return history, history_fine

# Step 4: Save the Model
def save_model(model, save_path='plant_disease_model.keras'):
    model.save(save_path)
    print(f"Model saved to {save_path}")

# Main Workflow
if __name__ == "__main__":
    # Define paths
    train_dir = '/home/prormrxcn/Documents/internal_hackathon/train'
    val_dir = '/home/prormrxcn/Documents/internal_hackathon/val'

    # Load data
    train_ds, val_ds, class_names = load_data(train_dir, val_dir, img_size=(160, 160))

    # Build model
    model, base_model = build_model(input_shape=(160, 160, 3), num_classes=len(class_names))

    # Train and fine-tune model
    history, history_fine = train_model(model, base_model, train_ds, val_ds, epochs=10)

    # Save final model
    save_model(model)