# -*- UTF-8 -*-
import importlib
import tensorflow as tf
import texar as tx

from nmt.seq2seq_attn import build_model


def eval_epoch(sess, mode, data_iterator, batch, infer_outputs, val_data):
    if mode == 'val':
        data_iterator.switch_to_val_data(sess)
    else:
        data_iterator.switch_to_test_data(sess)

    refs, hypos = [], []
    while True:
        try:
            fetches = [
                batch['target_text'][:, 1:],
                infer_outputs.predicted_ids[:, :, 0]
            ]
            feed_dict = {
                tx.global_mode(): tf.estimator.ModeKeys.EVAL
            }
            target_texts_ori, output_ids = sess.run(fetches, feed_dict=feed_dict)

            target_texts = tx.utils.strip_special_tokens(
                target_texts_ori, is_token_list=True)
            output_texts = tx.utils.map_ids_to_strs(
                ids=output_ids, vocab=val_data.target_vocab)

            for hypo, ref in zip(output_texts, target_texts):
                hypos.append(hypo)
                refs.append([ref])
        except tf.errors.OutOfRangeError:
            break

    print("---------------------------refs-------------------")
    print(refs)
    print("---------------------------hypos-------------------")
    print(hypos)

    return tx.evals.corpus_bleu_moses(list_of_references=refs, hypotheses=hypos)

def evaluate(config_model, config_data):
    # Data
    train_data = tx.data.PairedTextData(hparams=config_data.train)
    val_data = tx.data.PairedTextData(hparams=config_data.debug_val)
    test_data = tx.data.PairedTextData(hparams=config_data.debug_test)
    data_iterator = tx.data.TrainTestDataIterator(train=train_data, val=val_data, test=test_data)

    batch = data_iterator.get_next()  # mini_batch

    train_op, infer_outputs = build_model(batch, train_data, config_model)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        best_val_bleu = -1.
        # restore session
        print("Loading model...")
        saver.restore(sess, "/tmp/model.ckpt")
        print("Load model succeed!!!")

        # for i in range(config_data.num_epochs):
        for i in range(1):
            val_bleu = eval_epoch(sess, 'val', data_iterator, batch, infer_outputs, val_data)
            best_val_bleu = max(best_val_bleu, val_bleu)
            print('val epoch={}, BLEU={:.4f}; best-ever={:.4f}'.format(i, val_bleu, best_val_bleu))

            # test_bleu = eval_epoch(sess, 'test', data_iterator, batch, infer_outputs, val_data)
            # print('test epoch={}, BLEU={:.4f}'.format(i, test_bleu))

            print('=' * 50)

if __name__ == '__main__':
    flags = tf.flags

    flags.DEFINE_string("config_model", "config.config_model", "The model config.")
    flags.DEFINE_string("config_data", "config.config_iwslt14", "The dataset config.")

    FLAGS = flags.FLAGS

    config_model = importlib.import_module(FLAGS.config_model)
    config_data = importlib.import_module(FLAGS.config_data)

    evaluate(config_model, config_data)