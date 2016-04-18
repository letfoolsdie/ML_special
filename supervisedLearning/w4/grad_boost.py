from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression

#def gbm_predict(X):
#    return [sum([coeff * algo.predict([x])[0] \
#    for algo, coeff in zip(trees, coefs)]) \
#        for x in X]
#N = 50
#trees = []
#coefs = [0.9/(i+1) for i in range(N)]
#df = load_boston()
#X = df.data
#y = df.target
#
#X_train, y_train = X[:380], y[:380]
#X_test, y_test = X[380:], y[380:]
#
#base = y_train.mean()
#svect =  -1 * (base - y_train)
#
#clf = DecisionTreeRegressor(max_depth=5, random_state=42)
#clf.fit(X_train, svect)
#trees.append(clf)
#
#for i in range(N-1):
#    svect = -1 * (gbm_predict(X_train) - y_train)
#    clf = DecisionTreeRegressor(max_depth=5, random_state=42)
#    clf.fit(X_train, svect)
#    trees.append(clf)
#print(mean_squared_error(y_test, gbm_predict(X_test)))
#
#clflin = LinearRegression()
#clflin.fit(X_train, y_train)
#print(mean_squared_error(y_test, clflin.predict(X_test)))

df = load_boston()
X = df.data
y = df.target


base_algorithms_list = [DecisionTreeRegressor(max_depth=5, random_state=42) for _ in range(50)]
coefficients_list = [0.9] * 50

def gbm_predict(X, i):
   return [sum([coeff * algo.predict([x])[0] for algo, coeff in zip(base_algorithms_list[0:i], coefficients_list[0:i])]) for x in X]

s = y

for inx, (algorithm, coefficient) in enumerate(zip(base_algorithms_list, coefficients_list)):
   algorithm.fit(X, s)
#   s = (gbm_predict(X, inx + 1) - y)
   s = -1 * (gbm_predict(X, inx+1) - y)
   print(mean_squared_error(gbm_predict(X, inx + 1), y))













