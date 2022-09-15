#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from PtrNet.dataset import DataGenerator
from PtrNet.actor import Actor


def PtrNet(data, config):
    # Get running configuration

    # Build tensorflow graph from config
    actor = Actor(config)

    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        # if config.restore_model is True:
        #     saver.restore(sess, config.restore_from)
        #     print("Model restored.")

        training_set = DataGenerator(config, data)

        # Get test data
        input_batch = training_set.test_batch()
        feed = {actor.input_: input_batch}

        # Sample solutions
        positions, _, _, _ = sess.run([actor.positions, actor.reward, actor.train_step1, actor.train_step2],
                feed_dict=feed)

        city = input_batch[0]
        position = positions[0]
        return position



