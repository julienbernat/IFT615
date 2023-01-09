from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_decision_regions(x: np.ndarray, y: np.ndarray, clf):
    """
    Code inspiré de scikit-learn.

    Parameters
    ----------
    x: np.ndarray
        Les données

    y: np.ndarray
        Les étiquettes

    clf
        Le classifieur
    """
    resolution = 0.001

    # define a set of markers
    markers = ('o', 'x')
    # define available colors
    cmap = ListedColormap(('red', 'blue'))

    # select a range of x containing the scaled test set
    x1_min, x1_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    x2_min, x2_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

    # create a grid of values to test the classifier on
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # plot the decision region...
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # ...and the points from the test set
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y == c1, 0],
                    y=x[y == c1, 1],
                    alpha=0.8,
                    c=cmap(idx),
                    marker=markers[idx],
                    label=c1)
    plt.show()


def conf_matrix_from_pred(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Code inspiré de https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

    Parameters
    ----------
    y_true: np.ndarray
        Vecteur d'étiquettes

    y_pred: np.ndarray
        Vecteur de prédictions
    """
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.suptitle("Confusion Matrix")

    plt.show()


def visualize_samples(X: np.ndarray, y_pred: np.ndarray, y_true: np.array):
    """
    Code inspiré de https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

    Parameters
    ----------
    X: np.ndarray
        Les données

    y_pred: np.ndarray
        Les prédictions

    y_true: np.ndarray
        La vérité terrain
    """
    n_rows = min(36 // 4, (y_pred != y_true).sum() // 4)
    n_cols = 4
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 12))
    fig.tight_layout(pad=2.0)
    i = 0
    for row in axes:
        l, r = i * n_cols, n_cols * i + n_cols
        images = X[l:r]
        predictions = y_pred[l:r]
        ground_truth = y_true[l:r]
        for ax, image, pred, actual in zip(row, images, predictions, ground_truth):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            image = image.reshape(8, 8)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title(f"Prédiction: {pred}")
            ax.set_ylabel(f"y: {actual}")
        i += 1
    plt.show()
