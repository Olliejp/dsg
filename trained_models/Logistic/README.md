## Folder with the trained Logistic Models. 

The models take all measurements from a single time sample to predict failure (output of 1) or non-failure (output of 0). Two models are trained with different number of training samples. The trained Models are save using the function using tf.keras.save() in tensorflow 2.0. They can be loaded by calling tf.keras.load() 
