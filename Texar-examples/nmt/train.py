# -*- UTF-8 -*-
import importlib
import tensorflow as tf
import texar as tx

# 导入config
from nmt.seq2seq_attn import build_model

def train(config_model, config_data):
    # Data
    train_data = tx.data.PairedTextData(hparams=config_data.train)
    val_data = tx.data.PairedTextData(hparams=config_data.val)
    test_data = tx.data.PairedTextData(hparams=config_data.test)
    data_iterator = tx.data.TrainTestDataIterator(train=train_data, val=val_data, test=test_data)

    batch = data_iterator.get_next()  # mini_batch

    train_op, infer_outputs = build_model(batch, train_data, config_model)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        data_iterator.switch_to_train_data(sess)

        step = 0
        while True:
            try:
                loss = sess.run(train_op)
                if step % config_data.display == 0:
                    print("step={}, loss={:.4f}".format(step, loss))
                step += 1
            except tf.errors.OutOfRangeError: # TODO Why using OutOfRangeError
                break

        # store the model
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    flags = tf.flags

    flags.DEFINE_string("config_model", "config.config_model", "The model config.")
    flags.DEFINE_string("config_data", "config.config_iwslt14", "The dataset config.")

    FLAGS = flags.FLAGS

    config_model = importlib.import_module(FLAGS.config_model)
    config_data = importlib.import_module(FLAGS.config_data)

    train(config_model, config_data)