"""
@author: Sebastian Cajas
 
 Loss functions multi-class image segmentation 
"""

from keras import backend as K
import tensorflow as tf

def jaccard_index(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection  = K.sum(y_true*y_pred)
    return (intersection+1.0)/(K.sum(y_true)+K.sum(y_pred)-intersection+1)

def jaccard_coefficient(y_true, y_pred):
    return -jaccard_index(y_true, y_pred)

'''Focal losss. 
Ref: https://github.com/mkocabas/focal-loss-keras/blob/master/focal_loss.py
e.g.: model.compile(optimizer='adam', loss=[focal_loss(alpha=.25, gamma=2)], metrics=['accuracy']) 
'''

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed