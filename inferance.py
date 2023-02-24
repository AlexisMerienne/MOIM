import tensorflow as tf

class Inferance():


    def __init__(self):
        pass


    def predict(self,bash):
        model = tf.keras.models.load_model("model")
        tr_predict = model(bash)
        return tr_predict.numpy()[0]



    