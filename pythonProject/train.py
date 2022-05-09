import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.model_selection
from deap import base, creator, tools, algorithms
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.svm import OneClassSVM
from enum import Enum
import threading
import pickle


class ScoreMode(Enum):
    AUC_ROC_SCORE = 1
    F1_SCORE = 2
    ACCURACY_SCORE = 3

class ModelFit:
    def __init__(self, file=None, n_isolation_forest=20, n_oneclass_svm=5, per_cent_of_attributes=0.4, threshold=-0.9,
                 scoreMode = ScoreMode.AUC_ROC_SCORE):
        if not type(scoreMode) == ScoreMode:
            raise TypeError("type of argument scoreMode must be ScoreMode")

        if not file == None:
            self.load_data_from_csv(file)

        self.per_cent_of_attributes = per_cent_of_attributes
        self.threshold = threshold
        self.scoreMode = scoreMode
        self.n_isolation_forest = n_isolation_forest
        self.n_oneclass_svm = n_oneclass_svm

        self.best_score = 0

    def load_data_from_csv(self, file):
        df = pd.read_csv(file, index_col=0)

        # Список столбцов
        columns = list(df.columns)

        # Не основные столбцы в обучении, при IsolationForest
        columns.remove("label")
        columns.remove("pid")

        # Столбец проверки обученной модели или для учителя модели
        y = df["label"]

        # Исходные данные
        X = df[columns]

        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                                        train_size=0.5,
                                                                                                        random_state=42)

        self.y_test[self.y_test == 1] = -1
        self.y_test[self.y_test == 0] = 1

        self.X_train = self.X_train.values
        self.X_test = self.X_test.values


    def fit_svm(self, individual):
        # print(individual)
        self.oneclasssvm_models = [OneClassSVM(nu=ind) for ind in individual]

        # Обучение моделей на разных случайных подмножествах признаков
        self.oneclasssvm_subset_indices = [
            np.random.choice(self.X_train.shape[1], replace=False,
                             size=int(self.X_train.shape[1] * self.per_cent_of_attributes)) for _ in
            range(len(self.oneclasssvm_models))]
        self.n_oneclasssvm_subsets = len(self.oneclasssvm_subset_indices)

        for i, model in enumerate(self.oneclasssvm_models):
            X_subset = self.X_train[:, self.oneclasssvm_subset_indices[i]]
            model.fit(X_subset)

    def fit_isolation_forest(self, individual):
        # Создание списка моделей IsolationForest
        self.isolation_models = [IsolationForest(n_estimators=ind) for ind in individual]

        # Обучение моделей на разных случайных подмножествах признаков
        self.subset_indices = [np.random.choice(self.X_train.shape[1], replace=False,
                                                size=int(self.X_train.shape[1] * self.per_cent_of_attributes)) for _ in
                               range(len(self.isolation_models))]
        self.n_subsets = len(self.subset_indices)

        for i, model in enumerate(self.isolation_models):
            X_subset = self.X_train[:, self.subset_indices[i]]
            model.fit(X_subset)

    def evaluate(self, individual):
        ind_isolation_forest = individual[:self.n_isolation_forest]
        ind_svm = individual[self.n_isolation_forest:]
        self.fit_isolation_forest(ind_isolation_forest)
        self.fit_svm(ind_svm)
        # Оценка качества ансамбля
        return self.test()

    def f1_score_fitnesses(self, y_pred):
        y_pred = [-1 if element < self.threshold else 1 for element in y_pred]

        score = f1_score(self.y_test, y_pred, pos_label=-1)
        return score

    def roc_auc_fitnesses(self, y_pred):
        score = roc_auc_score(self.y_test, y_pred)
        return score

    def accuracy_fitnesses(self, y_pred):
        y_pred = [-1 if element < self.threshold else 1 for element in y_pred]

        score = accuracy_score(self.y_test, y_pred)
        return score

    def test(self):
        y_pred_subset = np.zeros((self.X_test.shape[0], self.n_subsets + self.n_oneclasssvm_subsets))

        # Получение оценок IsolationForest
        for i, model in enumerate(self.isolation_models):
            X_subset = self.X_test[:, self.subset_indices[i]]
            y_pred_subset[:, i] = model.predict(X_subset)

        # Получение оценок SVM
        for i, model in enumerate(self.oneclasssvm_models):
            X_subset = self.X_test[:, self.oneclasssvm_subset_indices[i]]
            y_pred_subset[:, self.n_subsets + i] = model.predict(X_subset)

        y_pred = np.mean(y_pred_subset, axis=1)

        if self.scoreMode == ScoreMode.AUC_ROC_SCORE:
            score = self.roc_auc_fitnesses(y_pred)
        elif self.scoreMode == ScoreMode.F1_SCORE:
            score = self.f1_score_fitnesses(y_pred)
        else:
            score = self.accuracy_fitnesses(y_pred)

        if self.best_score < score:
            self.best_score = score
            self.save("ensemble.pickle")

        return (score,)


    def init_individual(self):
        # n_isolation_forest моделей IsolationForest с количеством деревьев от 10 до 100
        ind_isolation_forest = [np.random.randint(10, 101) for _ in range(self.n_isolation_forest)]

        # n_oneclass_svm моделей SVM с параметром C от 0.1 до 10
        ind_svm = [np.random.uniform(0.01, 0.5) for _ in range(self.n_oneclass_svm)]

        return creator.Individual(tuple(ind_isolation_forest + ind_svm))

    def my_mutation(self, individual):
        for i in range(len(individual)):
            if random.random() < 0.5 and i > self.n_isolation_forest:
                individual[i] = np.random.uniform(0.01, 0.5)
            elif random.random() < 0.5 and i < self.n_isolation_forest:
                individual[i] = np.random.randint(10, 101)
        return (individual,)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

lock = threading.Lock()

def plot_graphic(x, y, xlabel, ylabel, global_label):
    with lock:
        # plt.hlines(MIN_FUNC, min(x), max(x), colors='r', linestyles='dashed')
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(global_label)
        plt.show()
        plt.clf()


def independence_of_time_from_pop_size():
    time_res = []
    population = []
    for size in range(10, 60, 10):
        pop = toolbox.population(n=size)

        print("\nPopulation: test #", size // 10, "of 5")
        start = time.time()
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof,
                                           verbose=True)
        end = time.time()
        time_res.append(end - start)
        population.append(size)

        # Вывод результатов
        print("Best individual:", hof[0])
        print("Best fitness:", hof[0].fitness.values[0])

    plot_graphic(population, time_res, "Популяция", "Время", "Время от популяции: ")

def independence_of_time_from_generations():
    time_res = []
    generations = []
    for ind in np.arange(10, 50, 10):
        pop = toolbox.population(n=10)
        print("\nGeneration: test #", ind // 10, "of 5")
        start = time.time()
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ind, stats=stats, halloffame=hof,
                                           verbose=True)
        end = time.time()
        time_res.append(end - start)
        generations.append(ind)

    plot_graphic(generations, time_res, "Количество поколений", "Время", "Время от поколения: ")

def independence_of_time_from_cross():
    time_res = []
    crosses = []
    for ind in np.arange(0.1, 1.0, 0.1):
        pop = toolbox.population(n=10)
        print("\nCrossover: test #", int(ind * 10 - 1), "of 11")
        start = time.time()
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=ind, mutpb=0.2, ngen=10, stats=stats, halloffame=hof,
                                           verbose=True)
        end = time.time()
        time_res.append(end - start)
        crosses.append(ind)

    plot_graphic(crosses, time_res, "Вероятность кроссинговера", "Время", "Время от вероятности кроссинговера: ")

def independence_of_time_from_mutate():
    time_res = []
    mutations = []
    for ind in np.arange(0.1, 1.0, 0.1):
        pop = toolbox.population(n=10)
        print("\nMutation: test #", int(ind * 10 - 1), "of 11")
        start = time.time()
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=ind, ngen=10, stats=stats, halloffame=hof,
                                           verbose=True)
        end = time.time()
        time_res.append(end - start)
        mutations.append(ind)

    plot_graphic(mutations, time_res, "Вероятность мутации", "Время", "Время от вероятности мутации: ")

def independence_of_b_fitnesses_from_pop_size():
    b_fitnesses = []
    population = []
    for size in range(10, 60, 10):
        pop = toolbox.population(n=size)

        print("\nBest fitnesses: test #", size // 10, "of 5")
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof,
                                           verbose=True)
        b_fitnesses.append(hof[0].fitness.values[0])
        population.append(size)

        # Вывод результатов
        print("Best individual:", hof[0])
        print("Best fitness:", hof[0].fitness.values[0])

    plot_graphic(population, b_fitnesses, "Наилучшие фитнес-функции", "Время",
                 "Наилучшие фитнес-функции от популяции: ")

def independence_of_b_fitnesses_from_cross():
    crosses = []
    b_fitnesses = []
    for ind in np.arange(0.1, 1.0, 0.1):
        pop = toolbox.population(n=10)
        print("\nCrossover: test #", int(ind * 10 - 1), "of 11")
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=ind, mutpb=0.2, ngen=10, stats=stats, halloffame=hof,
                                           verbose=True)
        crosses.append(ind)
        print("Best individual:", hof[0])
        print("Best fitness:", hof[0].fitness.values[0])
        b_fitnesses.append(hof[0].fitness.values[0])

    plot_graphic(crosses, b_fitnesses, "Кроссинговер", "Наилучшие фитнес-функции",
                 "Наилучшие фитнес-функции от вероятности кроссинговера: ")

def independence_of_b_fitnesses_from_mutate():
    mutations = []
    b_fitnesses = []
    for ind in np.arange(0.1, 1.0, 0.1):
        pop = toolbox.population(n=10)
        print("\nMutation: test #", int(ind * 10 - 1), "of 11")
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=ind, ngen=10, stats=stats, halloffame=hof,
                                           verbose=True)
        mutations.append(ind)
        # Вывод результатов
        print("Best individual:", hof[0])
        print("Best fitness:", hof[0].fitness.values[0])
        b_fitnesses.append(hof[0].fitness.values[0])

    plot_graphic(mutations, b_fitnesses, "Мутация", "Наилучшие фитнес-функции",
                 "Наилучшие фитнес-функции от вероятности мутации: ")

def independence_of_b_fitnesses_from_generation():
    generations = []
    b_fitnesses = []
    for ind in np.arange(10, 50, 10):
        pop = toolbox.population(n=10)
        print("\nGeneration: test #", ind // 10, "of 5")
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ind, stats=stats, halloffame=hof,
                                           verbose=True)
        generations.append(ind)
        # Вывод результатов
        print("Best individual:", hof[0])
        print("Best fitness:", hof[0].fitness.values[0])
        b_fitnesses.append(hof[0].fitness.values[0])

    plot_graphic(generations, b_fitnesses, "Поколения", "Наилучшие фитнес-функции",
                 "Наилучшие фитнес-функции от поколения: ")

modelFit = ModelFit("main11.csv", 10, 10, threshold=-0.6, scoreMode=ScoreMode.F1_SCORE)

# Создание класса для оптимизации
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", modelFit.init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", modelFit.evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", modelFit.my_mutation)
toolbox.register("select", tools.selTournament, tournsize=3)

# Запуск оптимизации
# pop = toolbox.population(n=10)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

obj = ModelFit.load("ensemble.pickle")
obj.load_data_from_csv("main11.csv")
print(obj.test())

pop = toolbox.population(n=10)
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof,
                                   verbose=True)

# independence_of_time_from_pop_size()
# independence_of_b_fitnesses_from_pop_size()
# independence_of_time_from_generations()
# independence_of_time_from_cross()
# independence_of_time_from_mutate()
#independence_of_b_fitnesses_from_cross()
#independence_of_b_fitnesses_from_mutate()
#independence_of_b_fitnesses_from_generation()

# Вывод результатов
print("Best individual:", hof[0])
print("Best fitness:", hof[0].fitness.values[0])

