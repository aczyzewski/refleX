import tensorflow as tf

def model_fn(features, labels, mode): # TODO dropout, learning_rate, vector_length, num_classes and num_featuers hardcoded

    logits_train = conv_net(features, 2, 0.25, reuse=False, is_training=True, vector_length=240, num_features=7)
    logits_test = conv_net(features, 2, 0.25, reuse=True, is_training=False, vector_length=240, num_features=7)

    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    f1_op = tf.contrib.metrics.f1_score(labels=labels, predictions=pred_classes)
    precision_op = tf.metrics.precision(labels=labels, predictions=pred_classes)
    recall_op = tf.metrics.recall(labels=labels, predictions=pred_classes)
    roc_auc_op = tf.metrics.auc(labels=labels, predictions=pred_classes)
    
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={
            'accuracy': acc_op,
            'f1': f1_op,
            'precision': precision_op,
            'recall': recall_op,
            'roc_auc': roc_auc_op
        })

    return estim_specs


def conv_net(feature_dict, n_classes, dropout, reuse, is_training, vector_length, num_features):

    with tf.variable_scope('ConvNet', reuse=reuse):
        
        # 3-D tensor input: [Batch Size, Vector Length, Channel]
        x = tf.reshape(feature_dict['images'], shape=[-1, vector_length, num_features])
        
        conv1 = tf.layers.conv1d(x, 8, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling1d(conv1, 2, 2)

        conv2 = tf.layers.conv1d(conv1, 16, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling1d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)

        fc1 = tf.layers.dense(fc1, 512)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        out = tf.layers.dense(fc1, n_classes)

    return out