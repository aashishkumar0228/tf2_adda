import tensorflow as tf

def get_source_encoder():
    model = tf.keras.models.Sequential()
    # Block 1
    model.add(tf.keras.layers.Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(32,3, padding  ="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # Block 2
    model.add(tf.keras.layers.Conv2D(64,3, padding  ="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(64,3, padding  ="same"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())

    print('Source/Target Encoder Summary')
    print(model.summary())
    return model

def get_source_classifier(classes):
    input_layer = tf.keras.layers.Input(shape=(3136,))
    x = tf.keras.layers.Dense(256, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    
    source_classifier_model = tf.keras.models.Model(inputs=input_layer, outputs=(x))
    print('Source/Target Classifier Summary')
    print(source_classifier_model.summary())

    return source_classifier_model

def get_discriminator():
    input_layer = tf.keras.layers.Input(shape=(3136,))
    x = tf.keras.layers.Dense(256, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    discriminator_model = tf.keras.models.Model(inputs=input_layer, outputs=(x))
    print('Discriminator summary')
    discriminator_model.summary()
    return discriminator_model