import tensorflow as tf
import numpy as np

#Run the model eagerly
class Maml(tf.keras.Model):
    def outter_data(self,x,y, k = 5, evenly = False):
        if evenly:
            space = np.ceil(len(y) / k).astype('int32')
            arg = np.argsort(y, axis=0)
            indexes = arg[::space].flatten()
        else:
            indexes = np.random.randint(0,y.shape[0], k)
        self.outter_x = tf.convert_to_tensor(x[indexes])
        self.outter_y = tf.convert_to_tensor(y[indexes])
    
    def copy_model(self, x):
        
        copied_model = Maml(self.inputs, self.outputs)
        copied_model.set_weights(self.get_weights())
        return copied_model

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as test_tape:
            with tf.GradientTape() as train_tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            trainable_vars = self.trainable_variables
            gradients = train_tape.gradient(loss, trainable_vars)
        
            model_copy = self.copy_model(x)
            self.optimizer.apply_gradients(zip(gradients, model_copy.trainable_variables))

            test_y_pred = model_copy(self.outter_x, training=True)
            test_loss = self.compiled_loss(self.outter_y, test_y_pred, regularization_losses=self.losses)
            
        gradients = test_tape.gradient(test_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}