import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # use CPU only, I have a newer version of CUDA installed that TF doesn't like

from model import create_model

# hyperparameters
epochs = 50
batch_size = 5
lr = 0.001

# make our datasets
classes = ['coke', 'pepsi']
size = (28, 28)  # we're using google and flickr thumbnails with a shallow CNN, so 28x28 will do
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,  # training generator with some mild augmentations
                                                                  rotation_range=10,
                                                                  horizontal_flip=True)
val_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_dataset = train_generator.flow_from_directory(directory='data/train',
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    target_size=size,
                                                    classes=classes,
                                                    class_mode='sparse')
val_dataset = val_generator.flow_from_directory(directory='data/val',
                                                batch_size=batch_size,
                                                shuffle=True,
                                                target_size=size,
                                                classes=classes,
                                                class_mode='sparse')

# initialize our model
model = create_model()

# define loss, optimizer, and metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr=lr)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


# our train and test step functions
def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
        grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # update metrics
    train_loss(loss)
    train_accuracy(y_train, predictions)


def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    test_loss(loss)
    test_accuracy(y_test, predictions)


for epoch in range(epochs):
    # train
    for _ in range(len(train_dataset) - 1):
        x_train, y_train = next(train_dataset)
        train_step(model, optimizer, x_train, y_train)

    # validate
    for _ in range(len(val_dataset) - 1):
        x_test, y_test = next(val_dataset)
        test_step(model, x_test, y_test)

    # print some info for this epoch
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

    # reset metrics
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

# save our model
tf.saved_model.save(model, 'saved_model')

