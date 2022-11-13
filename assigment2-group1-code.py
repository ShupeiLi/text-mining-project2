# -*- coding: utf-8 -*-

"""
Group 1
Name and Student Number: 
Shupei Li, s3430863
Qinshan Sun, s3674320

Versions:
    python: 3.6.13
    spacy: 3.4.2, pipline: en_core_web_trf
    sklearn: 0.22
"""

import re
import warnings
import spacy
from tqdm import tqdm
import json
import pandas as pd
from functools import wraps
from time import time
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


warnings.filterwarnings("ignore")
plt.style.use("ggplot")

train_path= "./data/wnut17train.conll"
evaluate_path = "./data/emerging.dev.conll"
test_path = "./data/emerging.test.annotated"


def add_pos(file_path, info, cdict=True, ex_feature=False):
    """
    Task 2: Add POS tags.
    """
    print("Loading data...")
    with open(file_path, "r", encoding="utf-8") as f:
        words = list()
        biotags = list()

        lines = f.readlines()
        for line in lines:
            line_lst = line.split("\t")
            if line_lst[0] != "":
                words.append(line_lst[0])
            else:
                words.append("\n")
            if len(line_lst) > 1:
                if line_lst[1] != "\n":
                    biotags.append(line_lst[1].strip("\n"))
                else:
                    biotags.append(line_lst[1])
            else:
                if line_lst[0] != "\n":
                    words.remove(line_lst[0])
                    words.append("\n")
                biotags.append("\n")

    text = " ".join(words)
    text = text.split("\n")
    text = [string.strip(" ") for string in text if string.strip(" ") != ""]

    print("Loading model...")
    nlp = spacy.load("en_core_web_trf")

    if cdict:
        print("Creating a dictionary...")
        pos_tags_dict = dict()
        for index in tqdm(range(len(text))):
            sentence = text[index]
            doc = nlp(sentence)
            for token in doc:
                if token.text not in pos_tags_dict:
                    pos_tags_dict[token.text] = token.pos_
        with open(f"./data/{info}_pos_tag_dict.json", "w") as f:
            json.dump(pos_tags_dict, f)
    else:
        with open(f"./data/{info}_pos_tag_dict.json", "r") as f:
            pos_tags_dict = json.load(f)

    print("Matching...")
    pos_tags = list()
    if ex_feature:
        shapes = list()
        stopwords = list()
        lemmas = list()
    for index in tqdm(range(len(words))):
        word = words[index]
        doc = nlp(word)
        if word in pos_tags_dict:
            pos_tags.append(pos_tags_dict[word])
        else:
            pos_tags.append(doc[0].pos_)
        if ex_feature:
            shapes.append(doc[0].shape_)
            stopwords.append(doc[0].is_stop)
            lemmas.append(doc[0].lemma_)

    if ex_feature:
        df = pd.DataFrame({"word": words, "pos": pos_tags, "biotag": biotags, 
                           "shape": shapes, "stopword": stopwords, "lemma": lemmas})
    else:
        df = pd.DataFrame({"word": words, "pos": pos_tags, "biotag": biotags})
    print(df.info())
    if ex_feature:
        df.to_pickle(f"./data/{info}_ex_data.pkl")
    else:
        df.to_pickle(f"./data/{info}_data.pkl")
    print("Done.")


def cal_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Time: func {func.__name__} took {end_time - start_time:2.4f} s.")
        return result
    return wrapper


class CrfModel():
    """
    Implement sequence labelling.
    """
    def __init__(self, add=False, ex=0):
        self.ex = ex
        if add:
            self.train = pd.read_pickle(f"./data/train_ex_data.pkl")
            self.evaluate = pd.read_pickle(f"./data/evaluate_ex_data.pkl")
            self.test = pd.read_pickle(f"./data/test_ex_data.pkl")
        else:
            self.train = pd.read_pickle(f"./data/train_data.pkl")
            self.evaluate = pd.read_pickle(f"./data/evaluate_data.pkl")
            self.test = pd.read_pickle(f"./data/test_data.pkl")

    def _save_results(self, msg, info):
        print(msg)
        with open("./results/ex3.txt", "a") as f:
            f.write(f"{info}\n")
            f.write(msg)
            f.write("\n")

    def _word2features_base(self, sent, i):
        """
        Task 3: Baseline with features from the tutorial.
        """
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features

    def _sent2features(self, sent):
        if self.ex == 1:
            return [self._word2features_ex(sent, i) for i in range(len(sent))]
        elif self.ex == 2:
            return [self._word2features_ex2(sent, i) for i in range(len(sent))]
        elif self.ex == 3:
            return [self._word2features_ex3(sent, i) for i in range(len(sent))]
        else:
            return [self._word2features_base(sent, i) for i in range(len(sent))]

    def _sent2labels(self, sent):
        if self.ex == 3:
            return [label for token, postag, label, shape, stopword, lemma in sent]
        else:
            return [label for token, postag, label in sent]

    def _sent2tokens(self, sent):
        if self.ex == 3:
            return [token for token, postag, label, shape, stopword, lemma in sent]
        else:
            return [token for token, postag, label in sent]

    def _transformer(self, data, info):
        features = list()
        labels = list()
        tokens = list()

        one_sentence = list()
        for row in data.iterrows():
            if row[1].tolist()[0] == "\n":
                features.append(self._sent2features(one_sentence))
                labels.append(self._sent2labels(one_sentence))
                tokens.append(self._sent2tokens(one_sentence))
                one_sentence = list()
            else:
                one_sentence.append(tuple(row[1].tolist()))
        print(f"{info}: {len(features)} sentences.")
        return features, labels, tokens

    @cal_time
    def preprocessing(self):
        """
        Create train data, evaluation data, and test data.
        """
        self.X_train, self.y_train, _ = self._transformer(self.train, "train")
        self.X_evaluate, self.y_evaluate, _ = self._transformer(self.evaluate, "evaluate")
        self.X_test, self.y_test, _ = self._transformer(self.test, "test")

    @cal_time
    def train_model(self, c1=0.1, c2=0.1):
        self.model = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=c1,
                c2=c2,
                max_iterations=100,
                all_possible_transitions=True,
            )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self, info):
        self.labels = list(self.model.classes_)
        self.labels.remove("O")
        y_pred = self.model.predict(self.X_test)
        sorted_labels = sorted(
                self.labels,
                key=lambda name: (name[1:], name[0])
            )
        self._save_results((metrics.flat_classification_report(
            self.y_test, y_pred, labels=sorted_labels, digits=4
        )), info)

    @cal_time
    def hyper_tuning(self, plot=False):
        """
        Task 4: Hyperparameter optimization.
        """
        self.preprocessing()
        self.train_model()
        self.labels = list(self.model.classes_)
        self.labels.remove("O")
        model_s = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                max_iterations=100,
                all_possible_transitions=True,
            )
        if self.ex == 2 or self.ex == 3:
            params_space = {
                    "c1": stats.expon(scale=1),
                    "c2": stats.expon(scale=1),
                }
        else:
            params_space = {
                    "c1": stats.expon(scale=0.05),
                    "c2": stats.expon(scale=0.05),
                }
        f1_scorer = make_scorer(
                metrics.flat_f1_score,
                average='weighted', 
                labels=self.labels,
            )
        rs = RandomizedSearchCV(
                model_s, 
                params_space,
                cv=5,
                verbose=1,
                n_jobs=-1,
                n_iter=50,
                scoring=f1_scorer,
            )
        rs.fit(self.X_evaluate, self.y_evaluate)

        self._save_results(f"best params: {rs.best_params_}", "Best Params")
        self._save_results(f"best CV score: {rs.best_score_}", "Best Score")
        self._save_results(f"model size: {rs.best_estimator_.size_ / 1000000:0.2f}M", "Model Size")

        if plot:
            _x = [s["c1"] for s in rs.cv_results_["params"]]
            _y = [s["c2"] for s in rs.cv_results_["params"]]
            _c = [s for s in rs.cv_results_["mean_test_score"]]

            fig = plt.figure()
            fig.set_size_inches(12, 12)
            ax = plt.gca()
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel("C1")
            ax.set_ylabel("C2")
            ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
                min(_c), max(_c)
            ))
            ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])
            plt.show()

        self.train_model(rs.best_params_["c1"], rs.best_params_["c2"])
        self.evaluate_model("After tuning params")

    def _word2features_ex(self, sent, i):
        """
        Task 5: Feature engineering - context: -2/+2
        """
        features = self._word2features_base(sent, i)
        if i > 1:
            word2 = sent[i - 2][0]
            postag2 = sent[i - 2][1]
            features.update({
                "-2:word.lower()": word2.lower(),
                "-2:word.istitle()": word2.istitle(),
                "-2:word.isupper()": word2.isupper(),
                "-2:postag": postag2,
                "-2:postag[:2]": postag2[:2],
            })
        else:
            features["-2BOS"] = True

        if i < len(sent) - 2:
            word2 = sent[i + 2][0]
            postag2 = sent[i + 2][1]
            features.update({
                "+2:word.lower()": word2.lower(),
                "+2:word.istitle()": word2.istitle(),
                "+2:word.isupper()": word2.isupper(),
                "+2:postag": postag2,
                "+2:postag[:2]": postag2[:2],
            })
        else:
            features["+2EOS"] = True

        return features

    def _word2features_ex2(self, sent, i):
        """
        Task 5: Feature engineering - context: -3/+3
        """
        features = self._word2features_ex(sent, i)
        if i > 2:
            word3 = sent[i - 3][0]
            postag3 = sent[i - 3][1]
            features.update({
                "-3:word.lower()": word3.lower(),
                "-3:word.istitle()": word3.istitle(),
                "-3:word.isupper()": word3.isupper(),
                "-3:postag": postag3,
                "-3:postag[:2]": postag3[:2],
            })
        else:
            features["-3BOS"] = True

        if i < len(sent) - 3:
            word3 = sent[i + 3][0]
            postag3 = sent[i + 3][1]
            features.update({
                "+3:word.lower()": word3.lower(),
                "+3:word.istitle()": word3.istitle(),
                "+3:word.isupper()": word3.isupper(),
                "+3:postag": postag3,
                "+3:postag[:2]": postag3[:2],
            })
        else:
            features["+3EOS"] = True

        return features

    def _word2features_ex3(self, sent, i):
        """
        Task 5: Feature engineering - additional features
        """
        features = self._word2features_ex2(sent, i)
        features.update({
            "word.shape": sent[i][3],
            "word.stopword": sent[i][4],
            "word.lemma": sent[i][5],
        })

        return features

    def statistics(self):
        def report_stat(df, info):
            space = df[df["word"]=="\n"].shape[0]
            print(f"{info}: #words - {df.shape[0] - space}, #sentences - {space}")

        def plot_format(i):
            for p in axes[i].patches:
                axes[i].annotate("{}".format(p.get_height()), (p.get_x() + 0.05, p.get_height() + 8))
            axes[i].tick_params(axis="x", rotation=90)

        report_stat(self.train, "Train")
        report_stat(self.evaluate, "Evaluate")
        report_stat(self.test, "Test")
        
        fig, axes = plt.subplots(1, 3)
        sub_train = self.train.loc[(self.train["biotag"] != "\n") & (self.train["biotag"] != "O")]
        plot_order = list(sub_train["biotag"].unique())
        plot_order.sort(key=lambda x: re.sub("\w-", "", x))
        sns.countplot(sub_train["biotag"], ax=axes[0], order=plot_order).set_title("Training Set")
        plot_format(0)
        sub_evaluate = self.evaluate.loc[(self.evaluate["biotag"] != "\n") & (self.evaluate["biotag"] != "O")]
        sns.countplot(sub_evaluate["biotag"], ax=axes[1], order=plot_order).set_title("Evaluation Set")
        plot_format(1)
        sub_test = self.test.loc[(self.test["biotag"] != "\n") & (self.test["biotag"] != "O")]
        sns.countplot(sub_test["biotag"], ax=axes[2], order=plot_order).set_title("Test Set")
        plot_format(2)
        plt.show()

    def run(self, info):
        self.preprocessing()
        self.train_model()
        self.evaluate_model(info)


if __name__ == "__main__":
    # Task 2: Add POS tags
    add_pos(train_path, "train", cdict=True)
    add_pos(evaluate_path, "evaluate", cdict=True)
    add_pos(test_path, "test", cdict=True)

    # Task 3: Baseline run
    crf = CrfModel()
    ## Statistics
    crf.statistics()
    ## Run baseline
    crf.run("Baseline")
    
    # Task 4: Hyperparameter optimization
    crf.hyper_tuning(True)

    # Task 5 & 6: Feature engineering
    ## Context: -2/+2
    crf.ex = 1
    crf.run("Context: -2/+2")
    crf.hyper_tuning()

    ## Context: -3/+3
    crf.ex = 2
    crf.run("Context: -3/+3")
    crf.hyper_tuning()

    ## Add other features
    add_pos(train_path, "train", cdict=True, ex_feature=True)
    add_pos(evaluate_path, "evaluate", cdict=True, ex_feature=True)
    add_pos(test_path, "test", cdict=True, ex_feature=True)
    crf = CrfModel(add=True, ex=3)
    crf.run("Other features")
    crf.hyper_tuning()
