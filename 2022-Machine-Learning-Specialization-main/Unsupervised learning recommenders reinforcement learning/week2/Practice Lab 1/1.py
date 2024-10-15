import numpy as np
from numpy import loadtxt

import tensorflow as tf
from tensorflow import keras

import pandas as pd


#############################################################################
def normalizeRatings(Y, R):
    """
    对电影评分进行均值归一化。注意只使用 R(i,j)=1 的评分计算均值。
    Args:
        Y：二维矩阵，记录所有的原始评分。每一行表示一个电影、每一列表示一个用户。
        R：二维矩阵，表示当前位置的评分是否有效。
    returns:
        Ynorm：二维矩阵，均值归一化后的评分。
        Ymean：一维向量，记录每个电影的平均评分。
    """
    Ymean = (np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12)).reshape(-1, 1)
    Ynorm = Y - np.multiply(Ymean, R)
    return (Ynorm, Ymean)


def load_precalc_params_small():
    """
    加载初始化的用户参数和电影特征：W、b、X
    """
    floder_path = 'jetbrains://pycharm/navigate/reference?project=Practice Lab 1&path=data'
    # 加载W
    file = open(floder_path + '/data/small_movies_W.csv', 'rb')
    W = loadtxt(file, delimiter=",")
    # 加载b
    file = open(floder_path + '/data/small_movies_b.csv', 'rb')
    b = loadtxt(file, delimiter=",")
    b = b.reshape(1, -1)
    # 加载X
    file = open(floder_path + '/data/small_movies_X.csv', 'rb')
    X = loadtxt(file, delimiter=",")
    return (X, W, b)


def load_ratings_small():
    """
    加载评分相关的参数：Y、R
    """
    floder_path = jetbrains://pycharm/navigate/reference?project=Practice Lab 1&path=data
    # 加载Y
    file = open(floder_path + '/data/small_movies_Y.csv', 'rb')
    Y = loadtxt(file, delimiter=",")
    # 加载R
    file = open(floder_path + '/data/small_movies_R.csv', 'rb')
    R = loadtxt(file, delimiter=",")
    return (Y, R)


def load_Movie_List_pd():
    """
    加载记录了“电影名称”、“平均评分”、“评分数量”等信息的表格。
    函数名中的“pd”表示使用了pands中的方法。
    returns df with and index of movies in the order they are  in the Y matrix
    """
    floder_path = 'jetbrains://pycharm/navigate/reference?project=Practice Lab 1&path=data'
    df = pd.read_csv(floder_path + '/data/small_movie_list.csv', header=0, index_col=0, delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return (mlist, df)


#############################################################################
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    计算协同过滤算法的代价函数(有正则项)。
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    # # 方法一：使用numpy中的矩阵运算
    # yhat = np.matmul(X,W.T)+b
    # squ_error = (((yhat-Y)*R)**2).sum() / 2
    # regular_item = ((W**2).sum() + (X**2).sum()) / 2 * lambda_
    # return squ_error + regular_item

    # 方法二：使用TensorFlow中的矩阵运算
    yhat = tf.linalg.matmul(X, tf.transpose(W)) + b - Y
    J = 0.5 * tf.reduce_sum((yhat * R) ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
    return J


###################################主函数##################################
# 加载原始数据集
# X, W, b = load_precalc_params_small()
# Y, R = load_ratings_small()
# num_movies, num_features = X.shape
# num_users,_ = W.shape
# print("Y", Y.shape, "R", R.shape)
# print("X", X.shape)
# print("W", W.shape)
# print("b", b.shape)
# print("num_features", num_features)
# print("num_movies",   num_movies)
# print("num_users",    num_users)

# 加载原始数据集
Y, R = load_ratings_small()
movieList, movieList_df = load_Movie_List_pd()

# 设置新用户的偏好
my_ratings = np.zeros(Y.shape[0])  # Initialize my ratings
# Check the file small_movie_list.csv for id of each movie in our dataset
# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[2700] = 5  # Toy Story 3 (2010)
my_ratings[2609] = 2  # Persuasion (2007)
my_ratings[929] = 5  # Lord of the Rings: The Return of the King, The
my_ratings[246] = 5  # Shrek (2001)
my_ratings[2716] = 3  # Inception
my_ratings[1150] = 5  # Incredibles, The (2004)
my_ratings[382] = 2  # Amelie (Fabuleux destin d'Amélie Poulain, Le)
my_ratings[366] = 5  # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
my_ratings[622] = 5  # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988] = 3  # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925] = 1  # Louis Theroux: Law & Disorder (2008)
my_ratings[2937] = 1  # Nothing to Declare (Rien à déclarer)
my_ratings[793] = 5  # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
# 输出新用户的评分
print('新用户的评分有：')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i, "title"]}');

# 将新用户评分新增到数据集中
Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0).astype(int), R]
num_movies, num_users = Y.shape
num_features = 100

# 均值归一化
Ynorm, Ymean = normalizeRatings(Y, R)

# 使用tf.Variable初始化参数：W、X、b
tf.random.set_seed(1234)  # for consistent results
W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

# 使用“协同过滤算法”迭代训练参数：W、X、b
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
iterations = 200  # 迭代次数
lambda_ = 1  # 代价函数的正则化参数
print("\n开始进行训练：")
for iter in range(iterations):
    # 定义代价函数计算过程
    with tf.GradientTape() as tape:
        # 计算代价的大小
        cost_value = cofi_cost_func(X, W, b, Ynorm, R, lambda_)

    # 定义需要求解哪些参数的偏导
    grads = tape.gradient(cost_value, [X, W, b])

    # Adam算法更新参数
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    # 展示迭代进度及代价
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

# 使用上述训练好的参数进行预测
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
pm = p + Ymean
my_predictions = pm[:, 0]
# 预测评分和原始评分的对比
print('\n新用户的原始评分 vs 预测评分:')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')
# 预测用户最喜欢的电影
ix = tf.argsort(my_predictions, direction='DESCENDING')
print("\n下面是预测评分Top17中没看过的电影：")
for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')
