# Model-Agnostic Meta-Learning (MAML) Tensorflow 2 Implementation

Model-Agnostic Meta-Learning (MAML) is a remarkable Deep Learning technique but sometimes their implementation and use comes a bit fuzzy. Below is a consise, fast and simple to use Tensorflow 2 MAML implementation designed by the course of the work.

### Requirements

- `Tensorflow >= 2.3.0`
- `Numpy >= 1.10.0`

### Example

`Fast, simple, easy to use`

```python3
inp = ResNet10V2(X_val[0,:,:,:].shape)
y_hat = tf.keras.layers.Dense(1)(inp.layers[-2].output)

model = Maml(inp.input, y_hat)
model.compile(optimizer = optimizer, loss = "mse", run_eagerly=True)
model.outter_data(X_val, Y_val, k = 5, evenly = True)
```


### Acknowledgements

Official implementation by Chelsea Finn: [here](https://github.com/cbfinn/maml)
