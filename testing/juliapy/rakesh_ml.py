import numpy as np
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class EnhancedPINN(tf.keras.Model):
    def __init__(self, Nx, **kwargs):
        super(EnhancedPINN, self).__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(100, activation='swish',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.hidden2 = tf.keras.layers.Dense(80, activation='swish',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.hidden3 = tf.keras.layers.Dense(60, activation='swish',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.output_layer = tf.keras.layers.Dense(Nx)
    
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.bn1(x)
        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.hidden3(x)
        return self.output_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({'Nx': self.Nx})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(101, **config)

# Initializing the model
pinn = EnhancedPINN(tf.keras.models.load_model('enhanced_pinnmodel.keras', custom_objects={'EnhancedPINN': EnhancedPINN}))

# init u
x = np.linspace(0, 2 * np.pi, 101)
u = np.sin(x)

output = pinn.call(u)
print(output)