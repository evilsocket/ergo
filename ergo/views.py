import os
import json
import itertools
import numpy as np
import logging as log

def model(prj, img_only):
    if prj.model is not None:
        prj.model.summary()

def roc(prj, img_only):
    if prj.dataset.has_test():
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        log.info("found %s, loading ...", prj.dataset.test_path)
        prj.dataset.load_test()
        log.info("computing ROC curve on %d samples ...", len(prj.dataset.X_test))

        y_pred = prj.model.predict(prj.dataset.X_test)
        fpr, tpr, thresholds = roc_curve(prj.dataset.Y_test.ravel(), y_pred.ravel())

        plt.figure("ROC Curve")
        plt.title("ROC Curve")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()

        plt.savefig( os.path.join(prj.path, 'roc.png') )

def stats(prj, img_only):
    if os.path.exists(prj.txt_stats_path):
        with open(prj.txt_stats_path, 'rt') as fp:
            print(fp.read().strip())

    if os.path.exists(prj.json_stats_path):
        import matplotlib.pyplot as plt

        with open(prj.json_stats_path, 'rt') as fp:
            stats = json.load(fp)
            for who, header in prj.what.items():
                orig = np.array(stats[who]['cm'])
                cm = np.array(stats[who]['cm'])
                tot = cm.sum()
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                name = header.strip(" -\n").lower()
                title = "%s confusion matrix (%d samples)" % (name, tot)
                filename = os.path.join(prj.path, "%s_cm.png" % name)

                plt.figure(title)
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
                plt.title(title)
                plt.colorbar()
                classes = range(0, cm.shape[0])
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)
                thresh = cm.max() / 2.
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, i, "%.1f%% (%d)" % (cm[i, j] * 100, orig[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('truth')
                plt.xlabel('prediction')
                plt.savefig(filename)

def history(prj, img_only):
    if prj.history is not None:
        import matplotlib.pyplot as plt

        plt.figure("training history")
        # Plot training & validation accuracy values
        plt.subplot(2,1,1)
        plt.plot(prj.history['acc'])
        plt.plot(prj.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='lower right')

        # Plot training & validation loss values
        plt.subplot(2,1,2)
        plt.plot(prj.history['loss'])
        plt.plot(prj.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.tight_layout()
        plt.savefig( os.path.join(prj.path, 'history.png') )

def show(img_only):
    if not img_only:
        import matplotlib.pyplot as plt
        plt.show()
