from lucid.modelzoo.vision_base import Model
from lucid.optvis import render
import tensorflow as tf
from lucid.misc.io import load, save


class CLIPImage(Model):
    image_value_range = (0, 255)
    input_name = 'input_image'
    def __init__(self):
        self.model_name = "RN50_4x"
        self.image_shape = [288, 288, 3]
        self.model_path = "https://openaipublic.blob.core.windows.net/clip/tf/RN50_4x/084ee9c176da32014b0ebe42cd7ca66e/image32.pb"

    def load(self, inp = None):
        import tensorflow as tf
        if inp == None:
            self.inp = tf.placeholder(shape = (None,self.image_shape[0], self.image_shape[1], 3), dtype = tf.float32)   
        else:
            self.inp = inp
        self.T   = render.import_model(self, self.inp, self.inp)
        return self.inp, self.T


class CLIPText(Model):
    input_name = 'tokens'

    def __init__(self):
        self.model_name = f"RN50_4x_text"
        self.model_path = "https://openaipublic.blob.core.windows.net/clip/tf/RN50_4x/da21bc82c7bba068aa8163333438354c/text32.pb"

    def load(self, O = None):
        import tensorflow as tf
        if O == None:
            self.O = tf.placeholder(tf.int32, [None, None])  
        else:
            self.O = O
        tf.import_graph_def(self.graph_def, {self.input_name: self.O}, name = "text")
        gph = tf.get_default_graph()
        self.T = lambda x: gph.get_tensor_by_name("text/" + x + ":0")
        return self.O, self.T
