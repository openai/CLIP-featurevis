from tokenizer import SimpleTokenizer
from model import CLIPImage, CLIPText
import tensorflow as tf
from lucid.misc.io import load
import numpy as np

def imresize(img, size, scale=255):
    from PIL import Image
    im = Image.fromarray((img*scale).astype(np.uint8) )
    return np.array(im.resize(size, Image.BICUBIC)).astype(np.float32)/scale

tokenizer = SimpleTokenizer()

tf.reset_default_graph()
inp_text, T_text = CLIPText().load()
inp_img,  T_img  = CLIPImage().load()

sess = tf.Session()

captions = ["This is a dog", "This is a cat", "This is a dog and a cat"]
tokens   = []
for caption in captions:
	tokens.append(tokenizer.tokenize(caption)[0])

img    = imresize(load("https://openaipublic.blob.core.windows.net/clarity/dog_cat.jpeg"), [288,288])

text_embd = sess.run(T_text("text_post/l2_normalize"), {inp_text: tokens})
img_embd  = sess.run(T_img("l2_normalize"), {inp_img: [img]})

scores = (text_embd @ img_embd.T)[:,0]

for score, caption in zip(scores, captions):
	print(caption, score)