import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Lambda, Dense, Conv2D, Cropping2D, Flatten, Dropout
from keras.optimizers import Adam


im_shape = 160, 320, 3  # Shape of each image
lr = 3e-4               # Learning Rate
batch_size = 32         # Batch Size
epochs = 2**8           # Number of epochs

# Reading data
df = pd.read_csv('driving_log.csv')

# Validation split
train_samples, validation_samples = train_test_split(df, test_size=0.2)

# Function to generate model data
def generator(samples, batch_size=32): 
    
    num_samples = len(samples)
    
    # Infinite loop
    while True:
        # Shuffling the data
        samples = shuffle(samples)
        
        # Create batches
        for offset in range(0, num_samples, batch_size):
            
            # Fetching batch data
            batch_samples = samples[offset:offset+batch_size]

            images = [] # Empty list to store processed batch images
            angles = [] # Empty list to store processed batch sterring angle
            
            # Fetching and processing each point in batch
            for _, batch_sample in batch_samples.iterrows():
                
                # Randomly picking the orient of image and using correction accordingly
                choice = np.random.choice(['left','center','right'])
                correction = {'left':0.25, 'center':0, 'right':-0.25}
                
                name = batch_sample[choice].strip()
                
                image = plt.imread(name)
                angle = batch_sample['steering'] + correction[choice]
                
                # Augmenting the data randomly flipping the image vertically and inverting the steering angle
                if np.random.choice(['flip','not flip']) == 'flip':
                    image, angle = np.fliplr(image), -angle
                
                images.append(image)
                angles.append(angle)
                
            # Converting lists to numpy array
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)
            

# Defing train and validation data generators
train_generator      = generator(train_samples,      batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# Creating Model
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = im_shape))      # Cropping the image
model.add(Lambda(lambda x: (x/255 - 0.5)))                                      # Normalization Layer
model.add(Conv2D(24, (5, 5), activation = 'elu', strides = (2, 2)))             # Conv Layer with 24 5x5 filters of stride 2
model.add(Conv2D(36, (5, 5), activation = 'elu', strides = (2, 2)))             # Conv Layer with 36 5x5 filters of stride 2
model.add(Conv2D(48, (5, 5), activation = 'elu', strides = (2, 2)))             # Conv Layer with 48 5x5 filters of stride 2
model.add(Conv2D(64, (3, 3), activation = 'elu'))                               # Conv Layer with 64 3x3 filters of stride 1
model.add(Conv2D(64, (3, 3), activation = 'elu'))                               # Conv Layer with 64 3x3 filters of stride 1
model.add(Dropout(0.33))                                                        # Dropout for regularization
model.add(Flatten())                                                            # Flattening the 3D matrix
model.add(Dense(100, activation = 'elu'))                                       # Dense Layer with 100 nodes
model.add(Dense(50,  activation = 'elu'))                                       # Dense Layer with 50 nodes
model.add(Dense(10,  activation = 'elu'))                                       # Dense Layer with 10 nodes
model.add(Dense(1))                                                             # Final output node

# Compiling model
model.compile(loss = 'mse', optimizer = Adam(lr))

# Training the model
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch = len(train_samples)//batch_size,
                                     validation_data = validation_generator, 
                                     validation_steps = len(validation_samples)//batch_size, 
                                     epochs = epochs)
# Saving the model
model.save('model.h5')




