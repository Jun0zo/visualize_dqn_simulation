import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Conv2D, Activation, Lambda
from keras.models import Model
import cv2 
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(img):
    # Resize the image to the desired shape
    img = cv2.resize(img, (256, 256))
    # Convert the pixel values to the range [0, 1]
    img = img.astype('float32') / 255.0
    img = img.reshape((1, 256, 256, 3))
    return img

# Define the BAMModel class
class BAMModel:
    def __init__(self, input_shape, num_classes, channels=32, reduction_ratio=16):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Build the model
        inputs = Input(shape=self.input_shape)
        x = Conv2D(self.channels, kernel_size=(3, 3), activation='relu')(inputs)
        print(x.shape)
        x, attention_maps = self.BAM(x)
        print(x.shape)
        x = Conv2D(self.channels, kernel_size=(3, 3), activation='relu')(x)
        print(x.shape)
        x, attention_maps2 = self.BAM(x)
        print(x.shape)
        x = GlobalAveragePooling2D()(x)
        print(x.shape)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        print(x.shape)
        self.model = Model(inputs=inputs, outputs=[outputs, [attention_maps, attention_maps2]])
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        
    def BAM(self, inputs, return_attention_maps=False):
        # Channel attention module
        channel_avg = GlobalAveragePooling2D()(inputs)
        channel_fc1 = Dense(self.channels//self.reduction_ratio)(channel_avg)
        channel_relu = Activation('relu')(channel_fc1)
        channel_fc2 = Dense(self.channels)(channel_relu)
        channel_sigmoid = Activation('sigmoid')(channel_fc2)
        channel_sigmoid = Reshape((1, 1, self.channels))(channel_sigmoid)

        # Spatial attention module
        spatial_conv = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(inputs)
        spatial_conv = Lambda(lambda x: x[0] * x[1])([inputs, spatial_conv])

        # Feature recalibration
        output = Multiply()([channel_sigmoid, spatial_conv])
        output = Add()([output, inputs])
        
        attention_map = Lambda(lambda x: tf.reduce_sum(x, axis=-1))(spatial_conv)
        attention_map = Lambda(lambda x: tf.expand_dims(x, axis=-1))(attention_map)
        return output, attention_map
        
    def predict_with_attention_maps(self, X):
        output, attention_maps = self.model.predict(x)
        return output, attention_maps
    
    def predict(self, X):
        return self.model.predict(X)
        
    def train(self, X_train, y_train, batch_size=32, epochs=10):
        num_samples = X_train.shape[0]
        num_batches = num_samples // batch_size
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = (batch + 1) * batch_size
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                loss, accuracy = self.model.train_on_batch(X_batch, y_batch)
                print('Batch {}/{} - loss: {:.4f} - accuracy: {:.4f}'.format(batch + 1, num_batches, loss, accuracy))
    
if __name__ == '__main__':
    img = cv2.imread('./seo_in.jpg', cv2.IMREAD_COLOR)
    x = preprocess_image(img)
    
    bam_model = BAMModel((256, 256, 3), num_classes=1)
    
    outputs, attention_maps = bam_model.predict_with_attention_maps(x)

    # Print the shape of the outputs and attention maps
    print('Outputs shape:', outputs.shape)
    print('Attention maps shape:', attention_maps[0].shape)
    print('Attention maps shape:', attention_maps[1].shape)
    
    attention_map = attention_maps[0]
    
    fig, axs = plt.subplots(nrows=1, ncols=len(attention_maps))

    for i, map in enumerate(attention_maps):
        if len(map.shape) == 4:
            map = map.squeeze()
        axs[i].imshow(map)
        axs[i].set_title(f'Layer {i} Attention Map')

    plt.show()