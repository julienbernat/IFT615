################################################################################
# VERSION MODIFIÉE DE L'AUTOGRADER OFFERT PAR L'UNIVERSITÉ BERKLEY (CS188)
################################################################################

import optparse
import sys
import traceback

import sklearn.metrics
from sklearn import metrics
from sklearn.linear_model import Perceptron
import dataset
import graphic


class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass


class Tracker:
    def __init__(self, questions, maxes, prereqs, mute_output, use_graphic):
        self.questions = questions
        self.maxes = maxes
        self.prereqs = prereqs
        self.use_graphic = use_graphic

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
*** because Question {} builds upon your answer for Question {}.
""".format(prereq, q, q, prereq))
                return False

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        print("\nNotes provisoires\n==================")
        for q in self.questions:
            print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (sum(self.points.values()),
                                sum([self.maxes[q] for q in self.questions])))

        print("""Veuillez noter que le résultat obtenu n'est pas final, mais constitue un excellent indicateur de la fonctionnalité attendue de votre code.""")

    def add_points(self, pts):
        self.points[self.current_question] += pts


TESTS = []
PREREQS = {}


def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)


def test(q, points):
    def deco(fn):
        TESTS.append((q, points, fn))
        return fn

    return deco


def parse_options(argv):
    parser = optparse.OptionParser(description='Auto-correcteur pour les étudiant.es')
    parser.set_defaults(
        no_graphics=False,
        check_dependencies=False,
    )
    parser.add_option('--question', '-q',
                      dest='grade_question',
                      default=None,
                      help='Valider une seule question (e.x. `-q q1`)')
    parser.add_option('--no-graphics',
                      dest='no_graphics',
                      action='store_true',
                      help='Do not display graphics (visualizing your implementation is highly recommended for debugging).')
    parser.add_option('--check-dependencies',
                      dest='check_dependencies',
                      action='store_true',
                      help='check that numpy and matplotlib are installed')
    (options, args) = parser.parse_args(argv)
    return options


def main():
    options = parse_options(sys.argv)
    if options.check_dependencies:
        check_dependencies()
        return

    questions = set()
    maxes = {}
    for q, points, fn in TESTS:
        questions.add(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if options.grade_question:
        if options.grade_question not in questions:
            print("ERREUR: La question {} n'existe pas".format(options.grade_question))
            sys.exit(1)
        else:
            questions = [options.grade_question]
            PREREQS[options.grade_question] = set()

    tracker = Tracker(questions, maxes, PREREQS, False, not options.no_graphics)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            try:
                fn(tracker)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nKeyboardInterrupt: sortie du autograder")
                tracker.finalize()
                print("\n[autograder a été interrompu]")
                sys.exit(1)
            except:
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()
    tracker.finalize()


################################################################################
# Tests begin here
################################################################################

import numpy as np
import matplotlib
import contextlib


def check_dependencies():
    import matplotlib.pyplot as plt
    import sklearn
    import time
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    line, = ax.plot([], [], color="black")
    plt.show(block=False)

    for t in range(400):
        angle = t * 0.05
        x = np.sin(angle)
        y = np.cos(angle)
        line.set_data([x, -x], [y, -y])
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(1e-3)


def verify_node(node, expected_type, expected_shape, method_name):
    if expected_type == 'parameter':
        assert node is not None, (
            "{} should return an instance of nn.ndarray, not None".format(method_name))
        assert isinstance(node, np.ndarray), (
            "{} should return an instance of np.ndarray, instead got type {!r}".format(method_name,
                                                                                       type(node).__name__))
    elif expected_type == 'type':
        assert node is not None, "{} devrait retourner un objet de type {} et non None".format(method_name,
                                                                                               expected_shape)
        assert isinstance(node, expected_shape), "{} devrait retourner un objet de type {} et non {}".format(
            method_name, expected_shape, type(node))
    elif expected_type == 'node':
        assert node is not None, (
            "{} should return a node object, not None".format(method_name))
        assert isinstance(node, np.ndarray), (
            "{} should return a node object, instead got type {!r}".format(
                method_name, type(node).__name__))
    else:
        assert False, "If you see this message, please report a bug in the autograder"

    if expected_type != 'type':
        assert any(
            [(expected == '?' or node.shape == expected) for expected in expected_shape]), \
            ("{} devrait retourner un objet de dimension {} et non {}".format(method_name, expected_shape, node.shape)
             )


def trace_node(node_to_trace):
    """
    Returns a set containing the node and all ancestors in the computation graph
    """
    nodes = set()
    tape = []

    def visit(node):
        if node not in nodes:
            for parent in node.parents:
                visit(parent)
            nodes.add(node)
            tape.append(node)

    visit(node_to_trace)

    return nodes


@test('q1', points=13 - 2)
def check_binary_perceptron(tracker):
    import models

    np_random = np.random.RandomState(42)

    # Vérification de la méthode d'initialisation
    print("Vérification de la méthode init_params")
    for dim in range(1, 10):
        model = models.BinaryPerceptron(dim, n_iter=3, alpha=1e-4)
        weights = model.get_weights()
        bias = model.get_bias()
        verify_node(weights, 'parameter', [(1, dim), (dim,)], "BinaryPerceptron.get_weights()")
        verify_node(bias, 'type', float, "BinaryPerceptron.get_bias()")
    tracker.add_points(2)

    # Vérification de la méthode threshold
    print("Vérification de la méthode threshold")
    expected = np.load("data/autograder/q1-c-thresh.npy", allow_pickle=True)
    for dim in range(1, 10):
        model = models.BinaryPerceptron(dim)
        data = np_random.uniform(-10, 10, (dim, dim))
        actual = model.threshold(data)
        cur = expected[dim - 1]
        assert np.array_equal(actual, cur), (
            "La sortie de BinaryPerceptron.threshold() {} est différente du résultat attendu {}".format(actual, cur)
        )
    tracker.add_points(1)

    # Vérification de la méthode predict
    print("Vérification de la méthode predict")
    expected = np.load("data/autograder/q1-c-pred.npy", allow_pickle=True)
    data = np.load("./data/autograder/q1-c-pred-data.npy", allow_pickle=True)
    for i, d in enumerate(data):
        dim = i + 1
        model = models.BinaryPerceptron(dim, n_iter=3)
        weights, bias = np_random.uniform(0, 1, dim), 0
        model.w = weights
        model.b = bias
        actual = model.predict(data[i])
        cur = expected[i]
        assert (actual.ndim == 1), (
            "La méthode BinaryPerceptron.predict() doit retourner un vecteur de dimension {}."
            " En ce moment, votre méthode retourne une matrice de taille {}".format(expected.shape, actual.shape)
        )
        assert all([x == 0 or x == 1 for x in actual]), (
            "La méthode BinaryPerceptron.predict() peut seulement retourner 1 ou 0."
        )
        assert np.array_equal(actual, cur), (
            "La sortie de BinaryPerceptron.predict() {} est différente du résultat attendu {}".format(actual, cur)
        )
    tracker.add_points(3)

    # Vérification de la méthode fit
    for i, dim in enumerate(range(5, 10, 20)):
        X, y = sklearn.datasets.make_classification(
            100, dim, class_sep=3., random_state=np_random
        )
        # Entraînement
        model = models.BinaryPerceptron(dim)
        sk_model = Perceptron(random_state=np_random, shuffle=False)
        model.fit(X, y)
        sk_model.fit(X, y)
        # Prédiction et scores
        pred = model.predict(X)
        sk_pred = sk_model.predict(X)
        score = metrics.accuracy_score(y, pred)
        sk_score = metrics.accuracy_score(y, sk_pred)
        assert np.isclose(sk_score, score, atol=5), (
            "Le score doit être proche de {:%} (plus ou moins 5%). Vous avez obtenu {:%}".format(sk_score, score)
        )
    # Dernier test sur dataset linéairement séparable
    X, y, _, _ = dataset.load_iris(return_train_test_split=False)
    # Entraînement
    model = models.BinaryPerceptron(X.shape[1])
    model.fit(X, y)
    # Prédiction et scores
    pred = model.predict(X)
    score = metrics.accuracy_score(y, pred)

    # Visualisation
    if tracker.use_graphic:
        graphic.plot_decision_regions(X, y, model)

    assert score == 1., (
        "Les données sont linéairement séparables."
        "Par conséquent, BinaryPerceptron.fit() devrait converger vers une solution optimale qui ne génère aucune erreur de classification."
        "Résultat attendu: {:%}"
        "Vous avez obtenu: {:%}".format(1., score)
    )
    tracker.add_points(5)


@test('q2', points=13 - 2)
def check_multiclass_digits(tracker):
    import models

    np_random = np.random.RandomState(42)

    # Vérification de la méthode d'initialisation
    print("Vérification de la méthode init_params")
    for dim in range(1, 10):
        model = models.MulticlassPerceptron(dim, n_features=dim)
        weights = model.get_weights()
        bias = model.get_bias()
        verify_node(weights, 'parameter', [(dim, dim)], "MulticlassPerceptron.get_weights()")
        verify_node(bias, 'parameter', [(1, dim), (dim,)], "MulticlassPerceptron.get_bias()")
    tracker.add_points(2)

    # Vérification de la méthode predict
    print("Vérification de la méthode predict")
    # Méthode doit fonctionner sur des matrices et des vecteurs
    model = models.MulticlassPerceptron(3, n_features=2)
    data = np_random.uniform(-10, 10, (1, 2))
    pred = model.predict(data)
    assert pred.shape == (1, 1) or pred.shape == (1,), (
        "MulticlassPerceptron.predict doit retourner une matrice de taille (n_rows,) ou (n_rows, 1). Vous avez {}".format(
            pred.shape)
    )
    data = np_random.uniform(-10, 10, (100, 2))
    pred = model.predict(data)
    assert pred.shape == (100, 1) or pred.shape == (100,), (
        "MulticlassPerceptron.predict doit retourner une matrice de taille (n_rows,) ou (n_rows, 1). Vous avez {}".format(
            pred.shape)
    )

    # Tests statiques
    expected = np.load("./data/autograder/q2-pred.npy", allow_pickle=True)
    for dim in range(1, 10):
        model = models.MulticlassPerceptron(dim, n_features=dim)
        model.w = np_random.uniform(0, 1, (dim, dim))
        data = np_random.uniform(-10, 10, (dim, dim))
        actual = model.predict(data)
        cur = expected[dim - 1]
        assert np.array_equal(actual, cur), (
            "La sortie de MulticlassPerceptron.predict() {} est différente du résultat attendu {}".format(actual, cur)
        )
    tracker.add_points(4)

    # Vérification de la méthode fit
    print("Vérification de la méthode fit")
    X_train, y_train, X_test, y_test = dataset.load_nist()
    dim = X_train.shape[1]
    # Entraînement
    model = models.MulticlassPerceptron(10, n_features=dim, alpha=1e-4)
    model.fit(X_train, y_train)
    sk_model = Perceptron(shuffle=False)
    sk_model.fit(X_train, y_train)
    # Prédiction et score
    pred = model.predict(X_test)
    sk_pred = sk_model.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    sk_score = metrics.accuracy_score(y_test, sk_pred)

    # Visualisation
    if tracker.use_graphic:
        graphic.conf_matrix_from_pred(y_test, pred)
        X_err, y_err = X_test[pred != y_test], pred[pred != y_test]
        graphic.visualize_samples(X_err, y_err, y_test[pred != y_test])

    assert np.isclose(score, sk_score, atol=.05), (
        "Le score doit être supérieur ou égal à {:%} plus ou moins 5%. Vous avez obtenu {:%}".format(sk_score, score)
    )
    tracker.add_points(5)


#@test('q3', points=13 - 6)
def check_feature_engineering(tracker):
    import models

    X_train, y_train, X_test, y_test = dataset.load_nist()
    basic_model = models.MulticlassPerceptron(n_classes=10, n_features=X_train.shape[1], n_iter=3)
    improved_model = models.FeatureEngPerceptron(n_classes=10, n_features=X_train.shape[1], n_iter=3)

    # Vérification qu'on modifie les données
    X_tilde = improved_model.preprocess(X_train, y_train)
    assert not np.allclose(X_train, X_tilde), (
        "Pour l'instant, les données de départ et les données modifiées sont pratiquement identiques."
        "FeatureEngPerceptron.preprocess() doit modifier les données reçues en entrée."
    )
    del X_tilde
    tracker.add_points(5)

    basic_model.fit(X_train, y_train)
    basic_pred = basic_model.predict(X_test)
    basic_accuracy = metrics.accuracy_score(y_test, basic_pred)

    X_test = improved_model.preprocess(X_test, y_test)
    improved_model.fit(X_train, y_train)
    improved_pred = improved_model.predict(X_test)
    improved_accuracy = metrics.accuracy_score(y_test, improved_pred)

    # Vérification que les modifications apportent une amélioration au modèle de départ
    if improved_accuracy >= basic_accuracy + .01:
        tracker.add_points(2)
        print("Bravo! Vous avez obtenu une amélioration de {:%}.\n"
              "Modèle de base: {:%}\n"
              "FeatureEngPerceptron: {:%}".format(improved_accuracy-basic_accuracy, basic_accuracy, improved_accuracy))
    else:
        print("Le score de FeatureEngPerceptron doit être supérieur ou égal au score du modèle de base ({:%}) + 1%."
              " Vous avez obtenu {:%}".format(basic_accuracy, improved_accuracy))


if __name__ == '__main__':
    main()
