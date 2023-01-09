from abc import abstractmethod, ABC
from typing import Tuple
import numpy as np


class SKModel(ABC):

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class BinaryPerceptron(SKModel):

    def __init__(self, n_features: int, n_iter: int = 200, alpha: float = 1e-4):
        """
        Initialisateur de classe

        Parameters
        ----------
        n_features: int
            Le nombre de dimensions des données à entraîner

        n_iter: int
            Le nombre d'itérations

        alpha: float
            Le taux d'apprentissage
        """
        self.n_dim = n_features
        self.n_iter = n_iter
        self.alpha = alpha
        self.w, self.b = self.init_params()

    def __str__(self) -> str:
        """
        Returns
        -------
        Retourne une représentation en chaîne de caractère de la classe
        """
        return 'Perceptron'

    def init_params(self) -> Tuple[np.ndarray, float]:
        """
        Initialise les paramètres `b` et `w` du modèle.

        Returns
        -------
        w: np.ndarray
            Le vecteur de poids de dimension (n_dim,) ou (1, n_dim)

        b: float
            Le biais
        """
        w, b = np.array([]), None
        # !!! VOTRE CODE IÇI !!!
        return w, b

    def get_bias(self) -> float:
        """
        Retourne le paramètre `b` du modèle

        Returns
        -------
        b: float
            Le biais du modèle
        """
        return self.b

    def get_weights(self) -> np.ndarray:
        """
        Retourne les poids du modèle.
        Fonction utilisée par l'autograder.

        Returns
        -------
        w: np.ndarray
            Un vecteur de dimension (1, n_features) ou (n_features,)
        """
        return self.w

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entraînement le perceptron sur les données `X` en fonction des étiquettes `y`

        Parameters
        ----------
        X: np.ndarray
            Les données d'entraînement
        y: np.ndarray
            Vecteur d'étiquettes de dimension (X.shape[0],)

        """
        print("ALLLLOOO \n \n \n ")
        for i in range(self.n_iter):
            for x_i, y_i in zip(X, y):
                print("I",i)
                print("XI",x_i)
                print("YI",y_i)
                # !!! VOTRE CODE IÇI !!!
                pass

    def threshold(self, X: np.ndarray) -> np.ndarray:
        # !!! VOTRE CODE IÇI !!!
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les classes y associées aux données X.
        Retourne le résultat de la fonction threshold appliquée au produit scalaire entre les poids `w` et les données
        `X`. Référez-vous au besoin à la formule `Threshold Function` de la page 724 du manuel de Norvig.

        Parameters
        ----------
        X: np.ndarray
            Les données à prédire dans une matrice de dimension (n_rows, n_features)

        Returns
        -------
        \hat{y}: np.ndarray
            Vecteur des prédictions de dimension (n_rows,)
        """
        # !!! VOTRE CODE IÇI !!!
        pass


class MulticlassPerceptron(BinaryPerceptron):

    def __init__(self, n_classes: int, **kwargs):
        """
        Initialisateur de classe
        Hérite de BinaryPerceptron car partage plusieurs paramètres

        Parameters
        ----------
        n_classes: int
            Nombre de classes à prédire

        kwargs
            Autres arguments pertinents
        """
        self.n_classes = n_classes
        super().__init__(**kwargs)
        self.w, self.b = self.init_params()

    def __str__(self) -> str:
        """
        Returns
        -------
        Retourne une représentation en chaîne de caractère de la classe
        """
        return 'MulticlassPerceptron'

    def get_bias(self) -> np.ndarray:
        """
        Retourne le paramètre `b` du modèle

        Returns
        -------
        b: float
            Le biais du modèle
        """
        return self.b

    def init_params(self) -> Tuple[np.ndarray, np.ndarray]:
        # !!! VOTRE CODE IÇI !!!
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Procédure d'entraînement du modèle.
        Le modèle s'entraîne à prédire le vecteur `y` associé à la matrice de données `X`.

        Parameters
        ----------
        X: np.ndarray
            La matrice de données de format (N, D)

        y: np.ndarray
            Le vecteur d'étiquette de format (N,) associé à chaque entrée de la matrice de données

        """
        for i in range(self.n_iter):
            for x_i, y_i in zip(X, y):
                # !!! VOTRE CODE IÇI !!!
                pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Le modèle prédit une ou plusieurs étiquettes associées.
        Conseil: regardez la fonction `argmax` de Numpy et portez attention au paramètres `axis`.

        Parameters
        ----------
        X: np.ndarray
            Les données à prédire dans une matrice de dimension (n_rows, n_features) ou un vecteur (n_features,)

        Returns
        -------
        pred: np.ndarray
            Vecteur des prédictions de dimension (n_rows,)
        """
        pass


class FeatureEngPerceptron(MulticlassPerceptron):

    def __str__(self) -> str:
        """
        Returns
        -------
        Retourne une représentation en chaîne de caractère de la classe
        """
        return 'FeatureEngPerceptron'

    def preprocess(self, X: np.ndarray, y: np.ndarray):
        """
        Transforme les données originales

        Parameters
        ----------
        X: np.ndarray
            La matrice de données de dimension (n_rows, n_features)
        y: np.ndarray
            Le vecteur d'étiquettes de dimension (n_rows,)

        Returns
        -------
        X_tilde: np.array
            La matrice de données modifiées respectant les mêmes dimensions (n_rows, n_features)
        """
        X_tilde = np.copy(X)
        return X_tilde

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Procédure d'entraînement du modèle.
        Le modèle s'entraîne à prédire le vecteur `y` associé à la matrice de données `X`.

        Parameters
        ----------
        X: np.ndarray
            La matrice de données de format (N, D)

        y: np.ndarray
            Le vecteur d'étiquette de format (N,) associé à chaque entrée de la matrice de données

        """
        X = self.preprocess(X, y)
        super().fit(X, y)
