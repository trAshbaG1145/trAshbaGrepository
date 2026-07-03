import numpy as np
from tqdm import tqdm
import pandas as pd

def changeID(data):
    changeSet = {}
    IDlist = data.drop_duplicates().to_list()
    newID = 0
    for i in IDlist:
        changeSet[i] = newID
        newID += 1
    return changeSet

def loadData(dataset_path):
    # 一条数据项(data_item)内所包含的信息按顺序展示为:
    # userId::movieId::rating::timestamp (ml-1m)
    data = pd.read_csv(dataset_path, sep="::", engine='python', header=None)
    mapping = changeID(data[0])
    data[0] = data[0].map(mapping)
    mapping = changeID(data[1])
    data[1] = data[1].map(mapping)
    userIDlist = data[0].drop_duplicates().to_list()
    itemIDlist = data[1].drop_duplicates().to_list()
    num_users = len(userIDlist)
    num_items = len(itemIDlist)
    print('***************************')
    print('Numbers of users:', num_users)  # 19445
    print('Numbers of items:', num_items)  # 7050
    print('Numbers of inters:', data.shape[0])  # 160792
    print('***************************')
    inters = np.array(data[[0, 1]])
    return num_users, num_items, inters

def getUIMat(num_users, num_items, data):
    # 构造U-I评分矩阵
    UI_matrix = np.zeros((num_users, num_items))
    # 遍历历史交互数据，令uimat[u][i] = 1
    UI_matrix[data[:, 0], data[:, 1]] = 1
    print("UI_matrix.shape:", UI_matrix.shape)
    return UI_matrix

class MF():
    def __init__(self, R, K, alpha, beta, epochs):
        """
        执行矩阵分解，预测矩阵中的0项。
        参数
        - R (ndarray)   : user-item 评分矩阵
        - K (int)       : 隐特征维度
        - alpha (float) : 学习率
        - beta (float)  : 正则化参数
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs

    def train(self):
        # 初始化用户和项目隐特征矩阵
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # 初始化 biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # 构建训练样本
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # 迭代进行随机梯度下降
        training_process = []
        for i in tqdm(range(self.epochs), total=self.epochs):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            # 每完成10%的训练迭代，就输出一次损失
            if (i == 0) or ((i+1) % (self.epochs / 10) == 0):
                print("Epoch: %d ; loss: %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        均方误差损失
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        for i, j, r in self.samples:
            # 计算预测值和error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            # 更新 biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            # 更新 user 和 item 隐特征矩阵
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        获取预测评分 r_ij，其中i是用户id，j是项目id
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        获取完整的预测矩阵
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

if __name__ == "__main__":
    num_users, num_items, obs_dataset = loadData('./ratings.dat')  # 读取数据 ratings.dat
    R = getUIMat(num_users, num_items, obs_dataset)  # 获取交互矩阵

    # alpha是学习率，不宜过大；beta是正则化系数，不宜过小
    mf = MF(R, K=2, alpha=0.1, beta=0.3, epochs=50)
    mf.train()

    # ------ 进行推荐 ------ #
    # 给用户1推荐top10
    each_user = 1
    user_ratings = mf.full_matrix()[each_user].tolist()
    topK = [(i, user_ratings.index(i)) for i in user_ratings]  # 关联项目id及其评分
    topK = [i[1] for i in sorted(topK, key=lambda x:x[0], reverse=True)][:10]

    print("------ user ------")
    print(each_user)
    print("------ topK ------")
    print(topK)

    # # 给所有用户推荐Top10
    # user_list = [i[0] for i in obs_dataset]
    # for each_user in tqdm(list(set(user_list)), total=len(list(set(user_list)))):
    #     user_ratings = mf.full_matrix()[each_user].tolist()
    #     topK = [(i, user_ratings.index(i)) for i in user_ratings]  # 关联项目id及其评分
    #     # 对TopN列表排序，取出index，即项目id
    #     topN = [i[1] for i in sorted(topK, key=lambda x:x[0], reverse=True)][:10]
    #     print("------ each_user ------")
    #     print(each_user)
    #     print("------ topK ------")
    #     print(topK)

