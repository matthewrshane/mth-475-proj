import tensorflow as tf
pinn = tf.keras.models.load_model('pinn_model.keras')

def predict(u):
    pinn.summary()

    y = 1000   # TODO: !!
    return y

predict(1)