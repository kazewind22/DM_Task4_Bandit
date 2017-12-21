import numpy as np

class LinUCB():
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        print("LinUCB with alpha "+str(alpha))

    def init(self):
        self.A = {}
        self.b = {}
        self.iter = {} # save for lazy update
        self.invA = {}
        for a in self.articles:
            self.A[a] = np.eye(6)
            self.b[a] = np.zeros(6)
            self.iter[a] = 0
            self.invA[a] = np.eye(6)

    def UCB(self, user_features, aid):
        # x = np.hstack((user_features, self.articles[aid]))
        x = np.array(user_features)
        theta = self.invA[aid].dot(self.b[aid])
        return theta.dot(x) + self.alpha * np.sqrt(x.dot(self.invA[aid].dot(x)))

    def recommend(self, time, user_features, choices):
        a = choices[np.argmax([self.UCB(user_features, c) for c in choices])]
        # self.x_t = np.hstack((user_features, self.articles[a]))
        self.x_t = np.array(user_features)
        self.a_t = a
        return a

    def update(self, reward):
        # if(reward==0):
        #     return
        a_t = self.a_t
        x_t = self.x_t
        self.A[a_t] += np.outer(x_t, x_t)
        # lazy update
        self.iter[a_t] += 1
        if self.iter[a_t] < 100000 or (self.iter[a_t]%10000==0):
            self.invA[a_t] = np.linalg.inv(self.A[a_t])
        # self.invA[a_t] = np.linalg.inv(self.A[a_t])
        if(reward==1):
            reward = 200
        elif(reward==0):
            reward = -1
        else:
            reward = 0
        self.b[a_t] += reward * x_t

class HybridLinUCB():
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        print("HybridLinUCB with alpha "+str(alpha))

    def init(self):
        self.A0 = np.eye(36)
        self.invA0 = np.eye(36)
        self.b0 = np.zeros(36)
        self.Beta = np.zeros(36) # invA0 * b0
        self.iter0 = 0 # save for lazy update
        self.A = {}
        self.invA = {}
        self.B = {}
        self.b = {}
        self.iter = {} # save for lazy update
        for a in self.articles:
            self.A[a] = np.eye(12)
            self.invA[a] = np.eye(12)
            self.B[a] = np.zeros((12,36))
            self.b[a] = np.zeros(12)
            self.iter[a] = 0

    def UCB(self, user_features, aid):
        invA0 = self.invA0
        invA_a = self.invA[aid]
        B_a = self.B[aid]

        x = np.hstack((user_features, self.articles[aid]))
        z = np.outer(np.array(user_features), self.articles[aid]).reshape(-1,)
        assert len(x) == 12 and len(z) == 36
        theta = invA_a.dot(self.b[aid] - B_a.dot(self.Beta))
        s = z.dot(invA0.dot(z)) - 2 * z.dot(invA0.dot(B_a.T.dot(invA_a.dot(x)))) + \
            x.dot(invA_a.dot(x)) + x.dot(invA_a.dot(B_a.dot(invA0.dot(B_a.T.dot(invA_a.dot(x))))))
        return z.dot(self.Beta) + x.dot(theta) + self.alpha * np.sqrt(s)

    def recommend(self, time, user_features, choices):
        a = choices[np.argmax([self.UCB(user_features, c) for c in choices])]
        self.x_t = np.hstack((user_features, self.articles[a]))
        self.z_t = np.outer(np.array(user_features), self.articles[a]).reshape(-1,)
        self.a_t = a
        return a

    def update(self, reward):
        a_t = self.a_t
        x_t = self.x_t
        z_t = self.z_t

        self.iter0 += 1
        self.iter[a_t] += 1

        self.A0 += self.B[a_t].T.dot(self.invA[a_t].dot(self.B[a_t]))
        self.b0 += self.B[a_t].T.dot(self.invA[a_t].dot(self.b[a_t]))
        self.invA0 = np.linalg.inv(self.A0)
        self.Beta = self.invA0.dot(self.b0)
        self.A[a_t] += np.outer(x_t, x_t)
        self.invA[a_t] = np.linalg.inv(self.A[a_t])
        self.B[a_t] += np.outer(x_t, z_t)
        self.b[a_t] += reward * x_t
        self.A0 += np.outer(z_t, z_t) - self.B[a_t].T.dot(self.invA[a_t].dot(self.B[a_t]))
        self.b0 += reward * z_t - self.B[a_t].T.dot(self.invA[a_t].dot(self.b[a_t]))


model = LinUCB(10)

def set_articles(articles):
    model.articles = articles
    model.init()

def update(reward):
    model.update(reward)

def recommend(time, user_features, choices):
    return model.recommend(time, user_features, choices)
