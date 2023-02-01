#!/usr/bin/env python
# coding: utf-8

# ### Машинное Обучения
# 
# ## Домашнее задание №2 - Дерево решений

# **Общая информация**
# 
# **Срок сдачи:** 1 февраля 2023, 08:30   
# **Штраф за опоздание:** -2 балла за каждые 2 дня опоздания
# 
# Решений залить в свой github репозиторий.
# 
# Используйте данный Ipython Notebook при оформлении домашнего задания.

# ##  Реализуем дерево решений (3 балла)

# Допишите недостающие части дерева решений. Ваша реализация дерева должна работать по точности не хуже DecisionTreeClassifier из sklearn.
# Внимание: если Вас не устраивает предложенная структура хранения дерева, Вы без потери баллов можете сделать свой класс MyDecisionTreeClassifier, в котором сами полностью воспроизведете алгоритм дерева решений. Обязательно в нем иметь только функции fit, predict . (Но название класса не менять)

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import tqdm

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


# In[2]:


class MyDecisionTreeClassifier:
    NON_LEAF_TYPE = 0
    LEAF_TYPE = 1

    def __init__(self, min_samples_split=2, max_depth=5, criterion="gini"):
        """
        criterion -- критерий расщепления. необходимо релизовать три:
        Ошибка классификации, Индекс Джини, Энтропийный критерий
        max_depth -- максимальная глубина дерева
        min_samples_split -- минимальное число объектов в листе, чтобы сделать новый сплит
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_class = -1
        # Для последнего задания
        self.feature_importances_ = dict()
        self.criterion = criterion
        # Структура, которая описывает дерево
        # Представляет словарь, где для  node_id (айдишник узла дерева) храним
        # (тип_узла, айдишник признака сплита, порог сплита) если тип NON_LEAF_TYPE
        # (тип_узла, предсказание класса, вероятность класса) если тип LEAF_TYPE
        # Подразумевается, что у каждого node_id в дереве слева 
        # узел с айди 2 * node_id + 1, а справа 2 * node_id + 2
        self.tree = dict()
        if criterion == "gini":
            self.quality_func = self.gini_
        elif criterion == "entropy":
            self.quality_func = self.entropy_
        else:
            print("Неверный критерий")
            
    
    def classify_(self, y, n):
        p = y / n[:, np.newaxis]
        return 1 - np.max(p, axis=1)

    def gini_(self, y, n):
        p = y / n[:, np.newaxis]
        p[np.isnan(p)] = 0
        return 1 - np.sum(p**2, axis=1)

    def entropy_(self, y, n):
        p = y / n[:, np.newaxis]
        p_log = p
        p_log[np.abs(p) < 1e-10] = 1
        p_log = np.log2(p)
        p_sum = p * p_log
        return (-1) * np.sum(p_sum, axis=1)

    def __div_samples(self, x, y, feature_id, threshold):
        """
        Разделяет объекты на 2 множества
        x -- матрица объектов
        y -- вектор ответов
        feature_id -- айдишник признака, по которому делаем сплит
        threshold -- порог, по которому делаем сплит
        """
        left_mask = x[:, feature_id] > threshold
        right_mask = ~left_mask
        return x[left_mask], x[right_mask], y[left_mask], y[right_mask]


    def __find_threshold(self, x, y):
        """
        Находим оптимальный признак и порог для сплита
        Здесь используемые разные impurity в зависимости от self.criterion
        """
        ind = np.argsort(x)
        x_sort, y_sort = x[ind], y[ind]
        thres, splits = np.unique(x_sort, return_index=True)
        counts = np.bincount(y_sort, minlength=self.num_class)
        left = np.array([np.bincount(y_sort[:k], minlength=self.num_class) for k in splits[:]])
        alles = np.tile(counts, thres.shape[0]).reshape(-1, counts.size)
        alles = alles[1:,:]
        left = left[1:, :]
        right = alles - left
        n_left = np.sum(left, axis=1)
        n_right = np.sum(right, axis=1)
        n = n_left + n_right
        qualities = self.quality_func(alles, n) - (n_left * self.quality_func(left, n_left) + n_right * self.quality_func(right, n_right)) / n
        if(np.isnan(qualities).all()):
            qualities =  np.array([0])
        qualities = qualities[~np.isnan(qualities)]
        max_idx = np.argmax(qualities)
        return qualities[max_idx], thres[max_idx]
    
    def __fit_node(self, x, y, node_id, depth):
        """
        Делаем новый узел в дереве
        Решаем, терминальный он или нет
        Если нет, то строим левый узел  с айди 2 * node_id + 1
        И правый узел с  айди 2 * node_id + 2
        """
        n = y.size
        values, counts = np.unique(y, return_counts=True)
        if depth >= self.max_depth or n < self.min_samples_split or values.shape[0] < 2:
            probability = counts / n
            ind = probability.argmax()
            self.tree[node_id] = (self.__class__.LEAF_TYPE, values[ind], probability[ind]) 
        else:
            thres_qualities = np.array([self.__find_threshold(i, y) for i in x.T])
            feature_id = np.argmax(thres_qualities[:, 0])
            threshold = thres_qualities[feature_id, 1]
            x_left, x_right, y_left, y_right = self.__div_samples(x, y, feature_id, threshold)
            if x_left.shape[0] == 0 or x_right.shape[0] == 0:
                probability = counts / n
                ind = probability.argmax()
                # создание текущего узла
                self.tree[node_id] = (self.__class__.LEAF_TYPE, values[ind], probability[ind]) 
            else:
                if feature_id not in self.feature_importances_:
                    self.feature_importances_[feature_id] = 0
                self.feature_importances_[feature_id] += thres_qualities[feature_id, 0]
                #создание текущего узла
                self.tree[node_id] = (self.__class__.NON_LEAF_TYPE, feature_id, threshold) 
                # создание левого узла
                self.__fit_node(x_left, y_left, 2 * node_id + 1, depth + 1)  
                #создание правого ущла
                self.__fit_node(x_right, y_right, 2 * node_id + 2, depth + 1) 

    def fit(self, x, y):
        """
        Рекурсивно строим дерево решений
        Начинаем с корня node_id 0
        """
        self.num_class = np.unique(y).size
        self.__fit_node(x, y, 0, 0) 

    def __predict_class(self, x, node_id):
        """
        Рекурсивно обходим дерево по всем узлам,
        пока не дойдем до терминального
        """
        node = self.tree[node_id]
        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_id, threshold = node
            if x[feature_id] > threshold:
                return self.__predict_class(x, 2 * node_id + 1)
            else:
                return self.__predict_class(x, 2 * node_id + 2)
        else:
            return node[1]
        
    def predict(self, X):
        """
        Вызывает predict для всех объектов из матрицы X
        """
        return np.array([self.__predict_class(x, 0) for x in X])
    
    def fit_predict(self, x_train, y_train, predicted_x):
        self.fit(x_train, y_train)
        return self.predict(predicted_x)
    
    def get_feature_importance(self):
        """
        Возвращает важность признаков
        """
        return self.feature_importances_


# In[3]:


my_clf = MyDecisionTreeClassifier(min_samples_split=2, criterion="entropy")
clf = DecisionTreeClassifier(min_samples_split=2, criterion="entropy")


# In[4]:


wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.1, stratify=wine.target)


# In[5]:


clf.fit(X_train, y_train)
my_clf.fit(X_train, y_train)


# In[6]:


sklearn_pred = clf.predict(X_test)
my_pred = my_clf.predict(X_test)
print("sklearn:", accuracy_score(y_pred=sklearn_pred, y_true=y_test))
print("My:", accuracy_score(y_pred=my_pred, y_true=y_test))


# Совет: Проверьте, что ваша реализация корректно работает с признаками в которых встречаются повторы. 
# И подумайте, какие еще граничные случаи могут быть.
# Например, проверьте, что на таком примере ваша модель корректно работает:

# In[7]:


X = np.array([[1] * 10, [0, 1, 2, 5, 6, 3, 4, 7, 8, 9]]).T
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
for depth in range(1, 5):
    my_clf = MyDecisionTreeClassifier(max_depth=depth)
    my_clf.fit(X, y)
    print("DEPTH:", depth, "\nTree:", my_clf.tree, my_clf.predict(X))


# ### Придумайте интересные примеры для отладки дерева решений (доп. задание)
# Это необязательный пункт. За него можно получить 1 доп балл. 
# Можете придумать примеры для отладки дерева, похожие на пример выше. 
# 
# Они должны быть не сложные, но в то же время информативные, чтобы можно было понять, что реализация содержит ошибки.
# Вместе с примером нужно указать ожидаемый выход модели. 

# In[8]:


# Примеры


# ## Ускоряем дерево решений (2 балла)
# Добиться скорости работы на fit не медленнее чем в 10 раз sklearn на данных wine. 
# Для этого используем numpy.

# In[9]:


get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# In[10]:


get_ipython().run_line_magic('time', 'my_clf.fit(X_train, y_train)')


# ## Боевое применение (3 балла)
# 
# На практике Вы познакомились с датасетом Speed Dating Data. В нем каждая пара в быстрых свиданиях характеризуется определенным набором признаков. Задача -- предсказать, произойдет ли матч пары (колонка match). 
# 
# Данные и описания колонок во вложениях.
# 
# Пример работы с датасетом можете найти в практике пункт 2
# https://github.com/VVVikulin/ml1.sphere/blob/master/2019-09/lecture_06/pract-trees.ipynb
# 
# Либо воспользоваться функцией:

# In[11]:


df = pd.read_csv("Speed_Dating_Data.csv", encoding="ISO 8859-1")
df = df.iloc[:, :97]


# In[12]:


df = df.drop(["id"], axis=1)
df = df.drop(["idg"], axis=1)
df = df.drop(["condtn"], axis=1)
df = df.drop(["round"], axis=1)
df = df.drop(["position", "positin1"], axis=1)
df = df.drop(["order"], axis=1)
df = df.drop(["partner"], axis=1)
df = df.drop(["age_o", "race_o", "like_o","pf_o_att", "pf_o_sin", "met_o", "pf_o_int", "pf_o_fun", 
              "pf_o_amb", "pf_o_sha", "dec_o", "attr_o", "sinc_o", "intel_o", "fun_o",
              "amb_o", "shar_o", "prob_o"], axis=1)
df = df.dropna(subset=["age"])
df.loc[:, "field_cd"] = df.loc[:, "field_cd"].fillna(19)
df = df.drop(["field"], axis=1)
df["field_cd"] = df["field_cd"].map(df.groupby("field_cd").match.sum() / df.groupby("field_cd").size()) 
df = df.drop(["undergra"], axis=1)
df.loc[:, "mn_sat"] = df.loc[:, "mn_sat"].str.replace(",", "").astype(np.float)
df = df.drop(["mn_sat"], axis=1)
df = df.drop(["tuition"], axis=1)
encoder = OneHotEncoder(sparse=False)
race_list = ["black", "europ", "latino", "asian","other"]
p = pd.DataFrame(encoder.fit_transform(df.race.values.reshape(-1, 1)), columns=["race_" + i for i in race_list])
df = pd.concat([df, p], axis=1)
df = df.drop(["race"], axis=1)
df = df.dropna(subset=["race_black"])
df = df.dropna(subset=["imprelig", "imprace"])
df = df.drop(["from", "zipcode"], axis=1)
df = df.drop(["income"], axis=1)
df = df.dropna(subset=["date"])
df = df.drop(["career"], axis=1)
df.loc[:, "career_c"] = df.loc[:, "career_c"].fillna(18)
df["career_c"] = df["career_c"].map(df.groupby("career_c").match.sum() / df.groupby("career_c").size())
df = df.drop(["sports","tvsports","exercise", "tv","dining","museums","art", "music","hiking","gaming",
       "clubbing","reading","theater","concerts","shopping","yoga", "movies"], axis=1)
df = df.drop(["expnum"], axis=1)
df.loc[:, "temp_totalsum"] = df.loc[:, ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]].sum(axis=1)
df.loc[:, ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]] = (df.loc[:, ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]].T/df.loc[:, "temp_totalsum"].T).T * 100
df.loc[:, "temp_totalsum"] = df.loc[:, ["attr2_1", "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]].sum(axis=1)
df.loc[:, ["attr2_1", "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]] = (df.loc[:, ["attr2_1", "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]].T/df.loc[:, "temp_totalsum"].T).T * 100
df = df.drop(["temp_totalsum"], axis=1)

for i in [4, 5]:
    feat = ["attr{}_1".format(i), "sinc{}_1".format(i), 
            "intel{}_1".format(i), "fun{}_1".format(i), 
            "amb{}_1".format(i), "shar{}_1".format(i)]
    
    if i != 4:
        feat.remove("shar{}_1".format(i))
    df = df.drop(feat, axis=1)

df = df.drop(["wave"], axis=1)
df_male = df.query("gender == 1").drop_duplicates(subset=["iid", "pid"])                                 .drop(["gender"], axis=1)                                 .dropna()
df_female = df.query("gender == 0").drop_duplicates(subset=["iid"])                                   .drop(["gender", "match", "int_corr", "samerace"], axis=1)                                   .dropna()
        
df_female.columns = df_female.columns + "_f"
df_female = df_female.drop(["pid_f"], axis=1)
df_female = df_female.rename(columns={"iid_f" : "pid"})


# In[13]:


df_both = pd.merge(df_male, df_female, on="pid")
df_both = df_both.drop(["iid", "pid"], axis=1)
X, y = df_both.iloc[:,1:].values, df_both.match.values.astype("int")


# Скачайте датасет, обработайте данные, как показано на семинаре или своим собственным способом. Обучите дерево классифкации. В качестве таргета возьмите колонку 'match'. Постарайтесь хорошо обработать признаки, чтобы выбить максимальную точность. Если точность будет близка к случайному гаданию, задание не будет защитано. В качестве метрики можно взять roc-auc. 
# 

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[15]:


my_clf = MyDecisionTreeClassifier(min_samples_split=2, criterion="gini")
clf = DecisionTreeClassifier(min_samples_split=2, criterion="gini")


# In[16]:


my_clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
my_predict = my_clf.predict(X_test)
sklearn_predict = clf.predict(X_test)


# In[17]:


my_res = (accuracy_score(y_test, my_predict), f1_score(y_test, my_predict, average="macro"))
sklearn_res = (accuracy_score(y_test, sklearn_predict), f1_score(y_test, sklearn_predict, average="macro"))
print("Мой результат:", my_res)
print("Результат sklearn", sklearn_res)


# Разбейте датасет на трейн и валидацию. Подберите на валидации оптимальный критерий  информативности. 
# Постройте графики зависимости точности на валидации и трейне от глубины дерева, от минимального числа объектов для сплита. (Т.е должно быть 2 графика, на каждой должны быть 2 кривые - для трейна и валидации)
# Какой максимальной точности удалось достигнуть?

# In[18]:


cv = StratifiedKFold(n_splits=5, shuffle=True)
depth_res = []

for depth in range(1, 70):
    my_clf = MyDecisionTreeClassifier(criterion="gini", max_depth=depth)
    p_res = []
    
    for train_ind, test_ind in cv.split(X, y):
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        my_clf.fit(X_train, y_train)
        my_predict = my_clf.predict(X_test)
        p_res.append((accuracy_score(y_test, my_predict), f1_score(y_test, my_predict, average="macro")))
    p_res = np.array(p_res).mean(axis=0)
    depth_res.append(p_res)


# In[19]:


plt.plot(range(1, 70), np.array(depth_res)[:, 1], c="m", lw=3)
plt.xlabel("Максимальная глубина")
plt.ylabel("F1-score")


# In[20]:


cv = StratifiedKFold(n_splits=5, shuffle=True)
sample_res = []

for sample in range(2, 70):
    my_clf = MyDecisionTreeClassifier(criterion="gini", max_depth=69, min_samples_split=sample)
    p_res = []
    
    for train_ind, test_ind in cv.split(X, y):
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        my_clf.fit(X_train, y_train)
        my_predict = my_clf.predict(X_test)
        p_res.append((accuracy_score(y_test, my_predict), f1_score(y_test, my_predict, average="macro")))
    p_res = np.array(p_res).mean(axis=0)
    sample_res.append(p_res)


# In[21]:


plt.plot(range(2, 70), np.array(sample_res)[:, 1], c="g", lw=3)
plt.xlabel("Максимальная глубина")
plt.ylabel("F1-score")


# Известным фактом является то, что деревья решений сильно переобучаются при увеличении глубины и просто запоминают трейн. 
# Замечаете ли вы такой эффект судя по графикам? Что при этом происходит с качеством на валидации? 

# В целом скачащий скор - не самый хороший признак. Присутствует элемент неточности.(переобучение?) На первом графике видно, что хоть выбросы и есть но тенденции на уменьшение скора не наблюдается.
# По второму графику видно, что скор постепенно падает с увеличением глубины дерева.

# ## Находим самые важные признаки (2 балла)
# 
# 

# По построенному дереву  легко понять, какие признаки лучше всего помогли решить задачу. Часто это бывает нужно  не только  для сокращения размерности в данных, но и для лучшего понимания прикладной задачи. Например, Вы хотите понять, какие признаки стоит еще конструировать -- для этого нужно понимать, какие из текущих лучше всего работают в дереве. 

# Самый простой метод -- посчитать число сплитов, где использовался данные признак. Это не лучший вариант, так как по признаку который принимает всего 2 значения, но который почти точно разделяет выборку, число сплитов будет очень 1, но при этом признак сам очень хороший. 
# В этом задании предлагается для каждого признака считать суммарный gain (в лекции обозначено как Q) при использовании этого признака в сплите. Тогда даже у очень хороших признаков с маленьким число сплитов это значение должно быть довольно высоким.  

# Реализовать это довольно просто: создаете словарь номер фичи : суммарный гейн и добавляете в нужную фичу каждый раз, когда используете ее при построении дерева. 

# Добавьте функционал, который определяет значения feature importance. Обучите дерево на датасете Speed Dating Data.
# Выведите 10 главных фичей по важности.

# In[22]:


my_clf = MyDecisionTreeClassifier(min_samples_split=14, max_depth=20, criterion="gini")
my_clf.fit(X, y)


# In[23]:


feature_importance = my_clf.get_feature_importance()
index = np.argsort(np.array(list(feature_importance.items()))[:,1])
feature_names = df_both.columns.values[1:][index][::-1]
feature_ids = np.array(list(feature_importance.keys()))[index][::-1]
feature_values = np.array(list(feature_importance.values()))[index][::-1]


# In[24]:


for k, i, j in zip(feature_names[:10], feature_ids[:10], feature_values[:10]):
    print("Фича: ", k, "\t\tScore is ", j)


# ## Фидбек (бесценно)

# * Какие аспекты обучения деревьев решений Вам показались непонятными? Какое место стоит дополнительно объяснить?

# ### Ваш ответ здесь

# * Здесь Вы можете оставить отзыв о этой домашней работе или о всем курсе.

# ### ВАШ ОТЗЫВ ЗДЕСЬ
# 
# 

# мне бы было намного легче, если бы где-то в углу слайда или после рассказа про критерии/методы где-то внизу была вставочка о том, как эти критерии/методы принято или чаще всего называют в коде. 

# Сложно, сложно. Комментарий из маркдауна выше в целом можно отнести ко всем темам. 
# Некоторые "лесенки и переходы" от чего-то простого к чему-то сложному часто пропускались и из-за этого на лекции много что было не понятно.  
# Предложение: на С++ у нас преподаватель после каждого занятия пускает анонимную гугл форму с обратной связью по занятию. Там есть какие-то дефолтные критерии, которые можно отметить звёздочками и есть место для комментариев и предложений. Преподаватель прислушивался к комментариям, которые считал полезными и качество проведения лекций росло. Это позволяет иметь постоянную обратную связь и курс был очень "гибким". Мне понравился такой подход)
