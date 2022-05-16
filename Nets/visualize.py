import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Nets.metrics as metrics


def plot_training_results(res=None, losses=None, metrics=None, res_fine=None, epochs=None,
                          save=False, path=None, max_f1=None):
    dim = (int(len(losses + metrics) / 2), 2)

    fig = plt.figure(figsize=(dim[0] * 12, dim[1] * 6))

    if max_f1 is not None:
        fig.suptitle("Maximum F1 Score = {:.3f} at threshold = {:.3f}".format(max_f1[0], max_f1[1]))

    # Plot Losses
    for i in range(len(losses)):
        plt.subplot(dim[0], dim[1], i + 1)

        if res_fine is None:
            plt.plot(range(1, len(res[losses[i]]) + 1), res[losses[i]], label='Training ' + losses[i])
            plt.plot(range(1, len(res[losses[i]]) + 1), res["val_" + losses[i]], label='Validation ' + losses[i])
            plt.xticks(range(1, len(res[losses[i]]) + 1))
        else:
            plt.plot(range(1, len(res[losses[i]] + res_fine[losses[i]]) + 1), res[losses[i]] + res_fine[losses[i]],
                     label='Training ' + losses[i])
            plt.plot(range(1, len(res[losses[i]] + res_fine[losses[i]]) + 1),
                     res["val_" + losses[i]] + res_fine["val_" + losses[i]],
                     label='Validation ' + losses[i])
            plt.xticks(range(1, len(res[losses[i]] + res_fine[losses[i]]) + 1))

            plt.plot([epochs, epochs], plt.ylim(), label='Start Fine Tuning')

        plt.legend(loc='upper right')
        plt.ylabel(losses[i])
        plt.xlabel('epoch')

    # Plot Metrics
    for i in range(len(metrics)):
        plt.subplot(dim[0], dim[1], i + 1 + len(losses))

        if "accuracy" in metrics[i]:
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])
        if res_fine is None:
            plt.plot(range(1, len(res[metrics[i]]) + 1), res[metrics[i]], label='Training ' + metrics[i])
            plt.plot(range(1, len(res[metrics[i]]) + 1), res["val_" + metrics[i]], label='Validation ' + metrics[i])
            plt.xticks(range(1, len(res[metrics[i]]) + 1))
        else:
            plt.plot(range(1, len(res[metrics[i]] + res_fine[metrics[i]]) + 1), res[metrics[i]] + res_fine[metrics[i]],
                     label='Training ' + metrics[i])
            plt.plot(range(1, len(res[metrics[i]] + res_fine[metrics[i]]) + 1),
                     res["val_" + metrics[i]] + res_fine["val_" + metrics[i]],
                     label='Validation ' + metrics[i])

            plt.xticks(range(1, len(res[metrics[i]] + res_fine[metrics[i]]) + 1))
            plt.plot([epochs, epochs], plt.ylim(), label='Start Fine Tuning')

        plt.legend(loc='lower right')
        plt.ylabel(metrics[i])
        plt.xlabel('epoch')
        # plt.title('Training and Validation Accuracy')

        if save:
            plt.savefig(path, bbox_inches='tight')

    plt.draw()


def plot_threshold_metrics_evaluation(model, ds, threshold_array, threshold_edge_width, save, path, accuracy_y_lim_min):
    tf.config.run_functions_eagerly(True)

    f1_score = np.zeros(threshold_array.shape)
    precision_score = np.zeros(threshold_array.shape)
    recall_score = np.zeros(threshold_array.shape)
    accuracy_score = np.zeros(threshold_array.shape)

    for i in range(threshold_array.shape[0]):

        threshold_prediction = np.log(threshold_array[i]) - np.log(1 - threshold_array[i])

        f1_evaluation = metrics.F1Edges(threshold_prediction=threshold_prediction,
                                        threshold_edge_width=threshold_edge_width, )
        accuracy_evaluation = metrics.BinaryAccuracyEdges(threshold_prediction=threshold_prediction)

        for img, label in ds:
            img, label = img, label

            prediction = model.predict(img)
            f1_evaluation.update_state(label, prediction[0])
            accuracy_evaluation.update_state(label, prediction[0])

        f1_score[i] = f1_evaluation.result()["f1"]
        precision_score[i] = f1_evaluation.result()["precision"]
        recall_score[i] = f1_evaluation.result()["recall"]
        accuracy_score[i] = accuracy_evaluation.result()

    tf.config.run_functions_eagerly(False)

    max_f1_score_idx = np.argmax(f1_score)
    max_f1_score = f1_score[max_f1_score_idx]
    max_precision_score_idx = np.argmax(precision_score)
    max_precision_score = precision_score[max_precision_score_idx]
    max_recall_score_idx = np.argmax(recall_score)
    max_recall_score = recall_score[max_recall_score_idx]
    max_accuracy_score_idx = np.argmax(accuracy_score)
    max_accuracy_score = accuracy_score[max_accuracy_score_idx]

    print("Maximum F1 Score = {:.3f} at threshold = {:.3f}"
          .format(max_f1_score, threshold_array[max_f1_score_idx]))
    print("Maximum Precision Score = {:.3f} at threshold = {:.3f}"
          .format(max_precision_score, threshold_array[max_precision_score_idx]))
    print("Maximum Recall Score = {:.3f} at threshold = {:.3f}"
          .format(max_recall_score, threshold_array[max_recall_score_idx]))
    print("Maximum Accuracy Score = {:.3f} at threshold = {:.3f}"
          .format(max_accuracy_score, threshold_array[max_accuracy_score_idx]))

    # define figure structure
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        "Maximum F1 Score = {:.3f} at Threshold = {:.3f} \n"
        "Maximum Accuracy Score = {:.3f} at threshold = {:.3f}".format(max_f1_score, threshold_array[max_f1_score_idx],
                                                                       max_accuracy_score,
                                                                       threshold_array[max_accuracy_score_idx]))
    overall_plot = plt.subplot2grid(shape=(3, 2), loc=(0, 0), colspan=2)
    accuracy_plot = plt.subplot2grid(shape=(3, 2), loc=(1, 0))
    f1_plot = plt.subplot2grid(shape=(3, 2), loc=(1, 1))
    recall_plot = plt.subplot2grid(shape=(3, 2), loc=(2, 0))
    precision_plot = plt.subplot2grid(shape=(3, 2), loc=(2, 1))

    overall_plot.plot(threshold_array, accuracy_score, label="Accuracy")
    overall_plot.plot(threshold_array, f1_score, label="F1")
    overall_plot.plot(threshold_array, precision_score, label="Precision")
    overall_plot.plot(threshold_array, recall_score, label="Recall")
    overall_plot.legend(loc='lower right')
    overall_plot.set_xlabel("Threshold")
    overall_plot.set_ylim([0, 1])
    overall_plot.set_xlim([0, 1])

    accuracy_plot.plot(threshold_array, accuracy_score)
    accuracy_plot.set_xlabel("Threshold")
    accuracy_plot.set_ylabel("Accuracy")
    accuracy_plot.set_ylim([accuracy_y_lim_min, 1])
    accuracy_plot.set_xlim([0, 1])
    # accuracy_plot.set_title("Maximum Accuracy Score = {:.3f} at Threshold = {:.3f}"
    #      .format(max_accuracy_score, threshold_array[max_accuracy_score_idx]))

    f1_plot.plot(threshold_array, f1_score)
    f1_plot.set_xlabel("Threshold")
    f1_plot.set_ylabel("F1")
    f1_plot.set_ylim([0, 1])
    f1_plot.set_xlim([0, 1])
    # f1_plot.set_title("Maximum F1 Score = {:.3f} at Threshold = {:.3f}"
    #      .format(max_f1_score, threshold_array[max_f1_score_idx]))

    recall_plot.plot(threshold_array, recall_score)
    recall_plot.set_xlabel("Threshold")
    recall_plot.set_ylabel("Recall")
    recall_plot.set_ylim([0, 1])
    recall_plot.set_xlim([0, 1])
    # recall_plot.set_title("Maximum Recall Score = {:.3f} at Threshold = {:.3f}"
    #      .format(max_recall_score, threshold_array[max_recall_score_idx]))

    precision_plot.plot(threshold_array, precision_score)
    precision_plot.set_xlabel("Threshold")
    precision_plot.set_ylabel("Precision")
    precision_plot.set_ylim([0, 1])
    precision_plot.set_xlim([0, 1])
    # precision_plot.set_title("Maximum Precision Score = {:.3f} at Threshold = {:.3f}"
    #      .format(max_precision_score, threshold_array[max_precision_score_idx]))

    if save:
        plt.savefig(path, bbox_inches='tight')

    plt.draw()

    return threshold_array[max_f1_score_idx]


def plot_training_results_old(train_res):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_res["acc"], label='Training Accuracy')
    plt.plot(train_res["val acc"], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(train_res["loss"], label='Training Loss')
    plt.plot(train_res["val loss"], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Multi Label Binary Cross Entropy')
    # plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.draw()


def plot_images(images=None, labels=None, predictions=None, save=False, path=None, batch_size=5):
    if predictions is None:
        prediction_bool = False
    else:
        prediction_bool = True

    if labels is None:
        label_bool = False
    else:
        label_bool = True

    plt.figure(figsize=(batch_size * 5, 16 + prediction_bool * 8))
    for i in range(batch_size):
        plt.subplot(1 + label_bool + prediction_bool, batch_size, i + 1)
        plt.title("Images")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images[i, :, :, :]))
        plt.axis('off')
        if label_bool:
            plt.subplot(1 + label_bool + prediction_bool, batch_size, batch_size + i + 1)
            plt.title("Ground Truth")
            plt.imshow(labels[i, :, :, 0], cmap='gray', vmin=0, vmax=3)
            plt.axis('off')
        if prediction_bool:
            plt.subplot(1 + label_bool + prediction_bool, batch_size, (1 + label_bool) * batch_size + i + 1)
            plt.title("Estimation")
            plt.imshow(predictions[i, :, :, 0], cmap='gray', vmin=0, vmax=3)
            plt.axis('off')

    if save:
        plt.savefig(path + ".png", bbox_inches='tight')
        plt.savefig(path + ".svg", bbox_inches='tight')

    plt.draw()


def predict_class_postprocessing(prediction, threshold=0.5):
    predictions = tf.math.sigmoid(prediction)
    val_max = tf.reduce_max(predictions, axis=-1)
    idx_max = tf.argmax(predictions, axis=-1) + 1

    predictions = tf.where(val_max >= threshold, idx_max, 0)

    predictions = tf.expand_dims(predictions, axis=-1)

    return predictions
