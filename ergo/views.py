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
        plt.plot(prj.history['accuracy'])
        plt.plot(prj.history['val_accuracy'])
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

def plot_clusters(prj, pca, X, y, ca, D3 = False):
    import matplotlib.pyplot as plt
    from numpy import argmax
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import axes3d

    Xt = pca.transform(X)
    Xt = Xt[:, :3]
    y = argmax(y, axis=1)

    clusters = ca.labels_
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(0, cmap.N, cmap.N//len(set(clusters)))]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    if D3:
        fig = plt.figure('Clustering Analisys - 3D projection')
        ax = fig.add_subplot(1, 1, 1, projection = '3d')
        ax.set_zlabel('Principal component 3')
    else:
        fig = plt.figure('Clustering Analisys')
        ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    markers = [m for m, func in Line2D.markers.items()
                        if func != 'nothing' and m in Line2D.filled_markers]
    _legends = []
    for cl in list(set(y)):
        idx =  np.where(y == cl)[0]
        if D3:
            scatter = ax.scatter(Xt[idx, 0], Xt[idx, 1], Xt[idx,2], cmap=cmap, s=40, alpha=0.5,
                                 c=clusters[idx],
                                 marker=markers[cl],
                                 label=y[idx],
                                 edgecolor='black'
                                 )
        else:
            scatter = ax.scatter(Xt[idx,0], Xt[idx,1], cmap = cmap, s = 40, alpha = 0.5,
                                 c = clusters[idx],
                                 marker = markers[cl],
                                 label=y[idx],
                                 edgecolor='black'
                                )
        l = Line2D([], [], color='black', marker=markers[cl], linestyle='None',
                          markersize=5, mfc = 'w', markeredgecolor = 'black', label='Class %d' % cl)
        _legends.append(l)

    for cl in list(set(clusters)):
        label = 'Cluster %d' % cl
        if cl == -1:
            label = 'Noisy samples'
        l = Line2D([], [], color= cmaplist[cl], marker='o', linestyle='None',
                          markersize=5,  label=label)
        _legends.append(l)

    legend1 = ax.legend(handles=_legends, loc='upper left')
    ax.add_artist(legend1)

    fig.tight_layout()
    if D3:
        name = 'clusters_projection_3D.png'
    else:
        name = 'clusters_projection.png'
    fig.savefig(os.path.join(prj.path, name))

def plot_intertia(prj, x, y):
    import matplotlib.pyplot as plt

    fig = plt.figure("Inertia of Kmeans clustering algorithm")
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('# of clusters')
    ax.set_ylabel('Intertia')
    ax.plot(x,y, '*-')
    fig.savefig(os.path.join(prj.path, 'kmeans_inertia.png'))


def pca_projection(prj, pca, X, y, D3):
    import matplotlib.pyplot as plt
    from numpy import argmax
    from mpl_toolkits.mplot3d import axes3d

    Xt = pca.transform(X)
    Xt = Xt[:,:3]
    y = argmax(y, axis=1)

    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    if D3:
        fig = plt.figure('PCA decomposition - 3D projection')
        ax = fig.add_subplot(1, 1, 1, projection = '3d')
        ax.set_zlabel('Principal component 3')
    else:
        fig = plt.figure('PCA decomposition')
        ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')

    if D3:
        scatter = ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:,2], c=y, cmap=cmap, label = y, s=5, alpha=0.5)
    else:
        scatter = ax.scatter(Xt[:, 0], Xt[:, 1], c=y, cmap=cmap, label=y, s=5, alpha=0.5)
    legend1 = ax.legend(*scatter.legend_elements(), loc = 'upper right', title = 'Class')
    ax.add_artist(legend1)
    fig.tight_layout()
    if D3:
        name = 'pca_projection_3d.png'
    else:
        name = 'pca_projection.png'
    fig.savefig( os.path.join(prj.path, name))


def pca_explained_variance(prj, pca, img_only):
    import matplotlib.pyplot as plt
    exp = pca.explained_variance_ratio_.cumsum()

    exp90, exp95, exp99 = -1, -1, -1
    for i,j in enumerate(exp):
        if j >= 0.9 and exp90 == -1:
            exp90 = i
        elif j >= 0.95 and exp95 == -1:
            exp95 = i
        elif j >= 0.99 and exp99 == -1:
            exp99 = i

    fig = plt.figure('PCA explained variance')
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal component number')
    ax.set_ylabel('Explained variance')
    ax.plot(exp, '-')

    # show 90, 95 and 99 % explanation
    ax.axvline(x=exp90, label='%d PC > 90%%' % exp90, linestyle = '--', c='k')
    ax.axvline(x=exp95, label='%d PC > 95%%' % exp95, linestyle = '--', c='b')
    ax.axvline(x=exp99, label='%d PC > 99%%' % exp99, linestyle = '--', c='r')
    ax.legend(title = 'Required components')

    fig.tight_layout()
    fig.savefig(os.path.join(prj.path, 'pca_explained_ratio.png'))


def correlation_matrix(prj, corr, img_only):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, square = True, mask=mask, cmap=cmap,cbar_kws={"shrink": .7})
    fig.savefig(os.path.join(prj.path, 'corr_matrix.png'))

def show(img_only):
    if not img_only:
        import matplotlib.pyplot as plt
        plt.show()

