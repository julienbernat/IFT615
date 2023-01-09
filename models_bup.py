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

    def __init__(self, n_features: int, n_iter: int = 5, alpha: float = 1e-4):
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
        w = np.zeros(self.n_dim)
        b = 0.5
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
        Entraîne le perceptron sur les données `X` en fonction des étiquettes `y`

        Parameters
        ----------
        X: np.ndarray
            Les données d'entraînement
        y: np.ndarray
            Vecteur d'étiquettes de dimension (X.shape[0],)

        """
        r = np.ndarray((X.shape[0],1), dtype=float)
        for i in range(self.n_iter):
            for x_i, y_i in zip(X, y):
                # !!! VOTRE CODE IÇI !!!
                r = self.predict(x_i)
                if y_i != r:
                    #Mise à jour nécessaire
                    self.w = self.w+self.alpha*(y_i-r)*x_i
                    self.b = self.b+self.alpha*(y_i-r)
        pass

    def threshold(self, X: np.ndarray) -> np.ndarray:
        # !!! VOTRE CODE IÇI !!!
        r = np.ndarray((X.shape[0],X.shape[1]), dtype=int)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i,j] >= 0:
                    r[i,j] = 1
                else:
                    r[i,j] = 0
        return r
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les classes y associées aux données X.
        Retourne le résultat de la fonction threshold appliquée au produit scalaire entre les poids `w` et les données
        `X`. Référez-vous au besoin à la formule `Threshold Function` de la page 724 du manuel de Norvig.

        Parameters
        ----------
        X: np.ndarray
            Les données à prédire

        Returns
        -------
        \hat{y}: np.ndarray
            Vecteur d'étiquettes correspondant à la prédiction du modèle pour chaque donnée x_i \in X
        """
        # !!! VOTRE CODE IÇI !!!
        if X.ndim == 1:
            r = np.ndarray((1,1), dtype=float)
            r[0,0] = np.dot(self.w,X)+self.b
        else:
            r = np.ndarray((X.shape[0],1), dtype=float)
            for j in range (X.shape[0]):
                r[j,0] = np.dot(self.w,X[j])+self.b
        return self.threshold(r).ravel()
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
        b = np.ndarray(shape=(self.n_dim,), dtype=float)
        b.fill(0)
        w = np.zeros((self.n_dim,self.n_dim))
        return w,b
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
        r = np.ndarray((X.shape[0],), dtype=float)
        for i in range(self.n_iter):
            for x_i, y_i in zip(X, y):
                # !!! VOTRE CODE IÇI !!!
                r = self.predict(x_i)
                if y_i != r:
                    #Mise à jour nécessaire
                    self.w = self.w+self.alpha*(y_i-r)*x_i
                    self.b = self.b+self.alpha*(y_i-r)
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Le modèle prédit un ou plusieurs étiquettes associées

        Parameters
        ----------
        X: np.ndarray
            Les données à prédire sous forme de matrice (n_rows, n_features) ou un vecteur (n_features,).

        Returns
        -------
        pred: np.ndarray
            Vecteur des prédictions de dimension (1, n_rows) ou (n_rows,)
        """
        if X.ndim == 1:
            r = np.array(X.shape[0])
            r = np.argmax(np.dot(X,self.w.T)+self.b)
        else:
            r = np.array((X.shape[0],))
            r = np.argmax(np.dot(X,(self.w).T)+self.b, axis = 1)
            print(f"produit matriciel = {np.dot(X,(self.w.T))+self.b}")
            print(f"r = {r}")
        return r
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
        # !!! VOTRE CODE IÇI !!!
        """!!! JUSTIFICATION DE VOS MODIFICATIONS. N'OUBLIEZ PAS VOS SOURCES !!!"""
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
