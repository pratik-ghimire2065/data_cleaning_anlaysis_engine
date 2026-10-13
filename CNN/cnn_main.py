import tensorflow as tf
from tensorflow.keras import datasets,models,layers

class Imageclassifier():
  def __init__(self,train_label,test_label,train_image,test_image):
    self.train_label=train_label
    self.test_label=test_label
    self.train_image=train_image
    self.test_images=test_image
  def train_model(self):
    self.train_image,self.test_images=self.train_images/255.0,self.train_images/255.0
    base_model=tf.keras.applications.MobileNetV2(
        input_shape=(32,32,3),
        include_top=False,
        weights="imagenet"
    )
    data_aug=models.Sequential([
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2)
    ])
    model=models.Sequential([
        base_model,
        data_aug,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(10,activation="softmax")
    ])

    base_model.trainable=True
    for layer in base_model.layers[:120]:
      layer.trainable=False
    model.compile(
        optimizer=tf.keras.optimizers.Adam.learning_rate(0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    history=model.fit(
        self.train_image,
        self.train_label,
        epochs=10,
        batch_size=32,
        verbose=1,
        validation_data=(self.test_images,self.test_images)
    )
    print("cnn model training complete let's proceed to another one model")