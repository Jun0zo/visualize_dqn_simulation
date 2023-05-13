import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model, Input

# Define Attention Layer
class AttentionLayer(layers.Layer):
    def __init__(self, input_shape, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.input_shape = input_shape

    def build(self, input_shape):
        self.attention = self.add_weight(shape=(self.input_shape[1], self.input_shape[2], 1), name='attention')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_weights = tf.math.softmax(self.attention, axis=1)
        return attention_weights * inputs

# Define Attention CNN
def create_attention_cnn(input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    # Attention layers
    a = layers.Dense(1)(x)
    a = layers.Flatten()(a)
    a = layers.Activation('softmax')(a)
    a = layers.Reshape((1, 1, -1))(a)
    x = AttentionLayer(input_shape)(x)
    x = layers.Multiply()([x, a])
    
    # Flatten and apply fully connected layers with dropout
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Load CIFAR-10 dataset and normalize pixel values
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create model and compile
model = create_attention_cnn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and visualize attention masks for a sample image
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

sample_image = x_test[0]
attention_model = Model(inputs=model.input, outputs=[model.output, model.get_layer('attention').output])
predictions, attention_map = attention_model.predict(sample_image[np.newaxis, ...])

plt.imshow(sample_image)
plt.imshow(np.squeeze(attention_map), cmap='gray', alpha=0.5)
plt.show()
