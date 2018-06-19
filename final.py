import os
import math, random
import glob
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

############# Parameter #####################
max_epochs = 25
image_path = "where_are_you_color/"
image_types = ["1_of_2", "1_of_3", "2_of_2","2_of_3","3_of_3"]
input_img_x = 64
input_img_y = 64
split_ratio = 0.9
batch_size =100
is_training=tf.placeholder(tf.bool)
drop_out_ratio=0.7

############# Model #####################
X = tf.placeholder(tf.float32, shape=[None, input_img_x, input_img_y, 3])
Y = tf.placeholder(tf.float32, shape=[None, len(image_types)])

L1=tf.layers.conv2d(X,16,[3,3])
L1=tf.layers.max_pooling2d(L1,[2,2],[2,2])
L1=tf.layers.dropout(L1,drop_out_ratio,is_training)

L2=tf.layers.conv2d(L1,32,[3,3])
L2=tf.layers.max_pooling2d(L2,[2,2],[2,2])
L2=tf.layers.dropout(L2,drop_out_ratio,is_training)

L3 = tf.contrib.layers.flatten(L2)
L3=tf.layers.dense(L3,16,activation=tf.nn.relu)
L3=tf.layers.dropout(L3,drop_out_ratio,is_training)

W1=tf.Variable(tf.random_normal([16,5],stddev=0.01))
model=tf.matmul(L3,W1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


############# Image Load #####################
full_set = []

for im_type in image_types:
    for path in glob.glob(os.path.join(image_path, im_type, "*")):
        im = cv2.imread(path)
        if not im is None:
            im = cv2.resize(im, (64, 64))


            one_hot_array = [0] * len(image_types)

            one_hot_array[image_types.index(im_type)] = 1

            full_set.append((im, one_hot_array, path))

random.shuffle(full_set)

############# Separate full_set to train_set and test_set #####################
split_index = int(math.floor(len(full_set) * split_ratio))
train_set = full_set[:split_index]
test_set = full_set[split_index:]

train_set_offset = len(train_set) % batch_size
test_set_offset = len(test_set) % batch_size
train_set = train_set[: len(train_set) - train_set_offset]
test_set = test_set[: len(test_set) - test_set_offset]

train_x, train_y, train_z = zip(*train_set)
test_x, test_y, test_z = zip(*test_set)

############# Training #####################
print("Starting training... [{} training examples]".format(len(train_x)))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(0, max_epochs):
    for j in range(0, (len(train_x) // batch_size)):
        start = batch_size * j
        end = batch_size * (j + 1)
        _,t_loss=sess.run([optimizer,loss],feed_dict={X: train_x[start:end], Y: train_y[start:end],is_training:True})
    t_loss = loss.eval(feed_dict={X: train_x, Y: train_y})
    print("Epoch {:5} , loss: {:15.10f},".format(i + 1, t_loss))


############# Result #####################
is_correct=tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도 : ',sess.run(accuracy,feed_dict={X:test_x,Y:test_y,is_training:False}))

############# Testing and Plotting #####################
y_=sess.run(model,feed_dict={X:test_x,Y: test_y,is_training:False})
print(y_)
print(y_.shape)
print(type(y_))
fig=plt.figure()

for i in range(10):
    for j in range(len(image_types)):
        if np.argmax(y_[i])==j:
            lbl=image_types[j]
    print(test_z[i])
    plt.subplot(2,5,i+1)
    plt.imshow(cv2.imread(test_z[i]))
    plt.axis('off')
    plt.title(lbl)
plt.show()

