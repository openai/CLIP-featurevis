from tqdm import tqdm
from model import CLIPImage, CLIPText
import tensorflow as tf
import os
import numpy as np

from lucid.optvis import objectives, param
import lucid.optvis.render as render
from lucid.optvis.objectives import wrap_objective, diversity
import lucid.optvis.transform as transform
from lucid.misc.io import load, save


@wrap_objective()
def l2(batch=None):
  def inner(T):
    return -tf.reduce_mean((T("input") - 0.5)**2)
  return inner

@wrap_objective()
def vector(layer, d, batch=None):
  def inner(T):
    channel_obj = tf.reduce_mean( tf.einsum( "ijkl,j->ikl", tf.nn.relu(T(layer)), tf.constant(d) ), [1,2])
    channel_obj_weighted = tf.reduce_mean(channel_obj)**(1/1)
    return channel_obj_weighted
  return inner

@wrap_objective()
def attr(obj, style_attrs, layers, strength):
    def inner(T):
        style = tf.constant(style_attrs)
        obj_t = obj(T)  
        layer_t = T(layers[0])
        w = tf.linspace(strength[0], strength[1], tf.shape(layer_t)[0])
        batch_n, _, _, _ = layer_t.get_shape().as_list()
        style = tf.transpose(style, (0,2,3,1))
        style = tf.image.resize(style, (tf.shape(layer_t)[2],tf.shape(layer_t)[3]))
        style = tf.transpose(style, (0,3,1,2))
        flat_attrs = []
        grads = tf.gradients(obj_t, [T(layer) for layer in layers])
        for layer, grad_t in zip(layers, grads):
            layer_t = T(layer)
            attr_t = layer_t * tf.nn.relu(tf.stop_gradient(grad_t))
            if len(style_attrs.shape) == 2:
                flat_attr_t = tf.reduce_sum(attr_t, axis=(2,3))
            elif len(style_attrs.shape) == 4:
                flat_attr_t = attr_t
            flat_attrs.append(flat_attr_t)
        flat_attr_t = tf.concat(flat_attrs, -1)
        return tf.reduce_sum(w[:,None,None,None]*flat_attr_t*style)
    return inner

def render_facet(model, neuron_obj, layers, style_attrs, strength = (0.1, 0.3), l2_weight = 10.0, resolution = 128, alpha = False):

    def mean_alpha():
        def inner(T):
            input_t = T("input")
            return tf.sqrt(tf.reduce_mean(input_t[..., 3:] ** 2))
        return objectives.Objective(inner)

    standard_transforms = [
        transform.pad(2, mode='constant', constant_value=.5),
        transform.jitter(4),
        transform.jitter(4),
        transform.jitter(4),
        transform.jitter(4),
        transform.jitter(4),
        transform.jitter(4),
        transform.jitter(4),
        transform.jitter(4),
        transform.jitter(4),
        transform.jitter(4),
        transform.random_scale([0.995**n for n in range(-5,80)] + [0.998**n for n in 2*list(range(20,40))]),
        transform.random_rotate(list(range(-20,20))+list(range(-10,10))+list(range(-5,5))+5*[0]),
        transform.jitter(2),
        transform.crop_or_pad_to(resolution, resolution)
    ]

    if alpha:
        standard_transforms.append(transform.collapse_alpha_random())
        param_f = lambda: param.image(resolution, batch=9, alpha=True)
    else:
        param_f = lambda: param.image(resolution, batch=9)

    optimizer      = tf.train.AdamOptimizer(0.02)
    ultimate_layer = [n.name for n in model.graph_def.node if "image_block_4" in n.name][-1]
    obj            = vector(ultimate_layer, neuron_obj) 
    facetsp        = [(5/len(layers))*attr(obj, style, [layer], strength) for style, layer in list(zip(style_attrs, layers))]
    for facetp in facetsp:
        obj = obj + facetp
    obj = obj + l2_weight*l2()
    if alpha:
        obj -= mean_alpha()
        obj -=  1e2 * objectives.blur_alpha_each_step()
    data = render.render_vis(model, obj, param_f, transforms=standard_transforms, optimizer=optimizer, thresholds=(1024*4,))
    return data

def one_hot(ind):
    z = np.zeros(2560)
    z[ind] = 1
    return z.astype(np.float32)

facets = ["face", "text", "logo", "pose", "arch", "nature", "indoor"]
model  = CLIPImage()
d      = one_hot(100)

for facet in facets:
    layernames  = [n.name for n in model.graph_def.node if ("image_block_3" in n.name) and ("Relu_2" in n.name)][::2]
    def loadnpy(url):
        import blobfile
        from io import BytesIO
        fp = blobfile.BlobFile(url, "rb")
        x  = np.load(BytesIO(fp.read()))
        fp.close()
        return x

    style_attrs = [loadnpy(f"https://openaipublic.blob.core.windows.net/clip/facets/{model.name}/{layername}/{facet}_spatial.npy") for layername in layernames]
    for l2_weight in [10]:
        img = render_facet(model, 
                           d, 
                           layernames, 
                           style_attrs, 
                           l2_weight = l2_weight, 
                           strength = (0.1, 5.0), 
                           alpha = False, 
                           resolution = 256)
        save(img[0][-1], f"/root/{facet}.png")
