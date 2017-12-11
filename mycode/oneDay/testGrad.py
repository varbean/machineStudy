# 数据校验
def validate(X, Y):
    if len(X) != len(Y):
        raise Exception("参数异常")
    else:
        m = len(X[0])
        for l in X:
            if len(l) != m:
                raise Exception("参数异常")
        if len(Y[0]) != 1:
            raise Exception("参数异常")


# 计算差异值
def calcDiffe(x, y, a):
    lx = len(x)
    la = len(a)
    if lx == la:
        result = 0
        for i in range(lx):
            result += x[i] * a[i]
        return y - result
    elif lx + 1 == la:
        result = 0
        for i in range(lx):
            result += x[i] * a[i]
        result += 1 * a[lx]  # 加上常数项
        return y - result
    else:
        raise Exception("参数异常")


## 要求X必须是List集合，Y也必须是List集合
def fit(X, Y, alphas, threshold=1e-6, maxIter=200, addConstantItem=True):
    import math
    import numpy as np
    ## 校验
    validate(X, Y)
    ## 开始模型构建
    l = len(alphas)
    m = len(Y)
    n = len(X[0]) + 1 if addConstantItem else len(X[0])
    print(l)
    print(m)
    print(n)
    B = [True for i in range(l)]
    ## 差异性(损失值)
    J = [np.nan for i in range(l)]
    # 1. 随机初始化0值(全部为0), a的最后一列为常数项
    a = [[0 for j in range(n)] for i in range(l)]
    # 2. 开始计算
    for times in range(maxIter):
        for i in range(l):
            if not B[i]:
                # 如果当前alpha的值已经计算到最优解了，那么不进行继续计算
                continue

            ta = a[i]
            for j in range(n):
                alpha = alphas[i]
                ts = 0
                for k in range(m):
                    if j == n - 1 and addConstantItem:
                        ts += alpha * calcDiffe(X[k], Y[k][0], a[i]) * 1
                    else:
                        ts += alpha * calcDiffe(X[k], Y[k][0], a[i]) * X[k][j]
                t = ta[j] + ts
                ta[j] = t
            ## 计算完一个alpha值的0的损失函数
            flag = True
            js = 0
            for k in range(m):
                js += math.pow(calcDiffe(X[k], Y[k][0], a[i]), 2)
                if js > J[i]:
                    flag = False
                    break;
            if flag:
                J[i] = js
                for j in range(n):
                    a[i][j] = ta[j]
            else:
                # 标记当前alpha的值不需要再计算了
                B[i] = False
                ## 计算完一个迭代，当目标函数/损失函数值有一个小于threshold的结束循环
        r = [0 for j in J if j <= threshold]
        if len(r) > 0:
            break
        # 如果全部alphas的值都结算到最后解了，那么不进行继续计算
        r = [0 for b in B if not b]
        if len(r) > 0:
            break;
    # 3. 获取最优的alphas的值以及对应的0值
    min_a = a[0]
    min_j = J[0]
    min_alpha = alphas[0]
    for i in range(l):
        if J[i] < min_j:
            min_j = J[i]
            min_a = a[i]
            min_alpha = alphas[i]

    print("最优的alpha值为:", min_alpha)

    # 4. 返回最终的0值
    return min_a


# 预测结果
def predict(X, a):
    Y = []
    n = len(a) - 1
    for x in X:
        result = 0
        for i in range(n):
            result += x[i] * a[i]
        result += a[n]
        Y.append(result)
    return Y


# 计算实际值和预测值之间的相关性
def calcRScore(y, py):
    if len(y) != len(py):
        raise Exception("参数异常")
    import math
    import numpy as np
    avgy = np.average(y)
    m = len(y)
    rss = 0.0
    tss = 0
    for i in range(m):
        rss += math.pow(y[i] - py[i], 2)
        tss += math.pow(y[i] - avgy, 2)
    r = 1.0 - 1.0 * rss / tss
    return r