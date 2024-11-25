## 现有模型的不足

现有模型在进行Granger因果构建时，很少涉及imperfect intervention with unknown target的情况

## 模型框架

> <img width="458" alt="image" src="https://github.com/user-attachments/assets/7eba1199-5965-4547-8d37-e1fc19a3a608">

1.包含三个网络：

> 预测网络、介入网络和因果网络

## VAR模型

向量自回归模型把每个内生变量作为系统中所有内生变量滞后值的函数来构造模型，从而避开了结构建模方法中需要对系统每个内生变量关于所有内生变量滞后值的建模问题。

### 模型结构

<img width="763" alt="image" src="https://github.com/user-attachments/assets/90701a46-3d66-49c6-9adb-e71989e8c4d8">

<img width="963" alt="image" src="https://github.com/user-attachments/assets/942ee306-251c-4fb4-a20c-555a03d5da34">

### 模型平稳性

VAR模型的平稳性是一个重要的假设。只有在模型是平稳的情况下，VAR模型的估计结果才是可靠的。平稳性意味着时间序列的均值、方差和协方差在时间上不随时间变化。对于VAR模型来说，平稳性可以通过检查特征根来判断。一个VAR模型是平稳的，当且仅当模型的特征根的模小于1。

## 疑问

### 1.Paper中的Heterogeneous体现在哪里？

### 2.Paper中的不同的env是如何生成的？

### 3.Intervention体现在哪？

> <img width="725" alt="image" src="https://github.com/user-attachments/assets/47c77199-b37e-4aee-9250-6ea798b367f2">

### 4.分析Non-Linear代码部分

def nonlinear_ts_data(T, lag, beta, GC, seed=0, sd=0.1, interv=False, anomaly=200, strength=0.1): # 产生Interventional data  # T=1000,strength=0.05，anomaly=500
    np.random.seed(seed)

    p = np.shape(GC)[0] # p=5
    beta = make_var_stationary(beta)    # beta作为VAR模型的系数矩阵

    interv_target = np.random.randint(0, 2, (p, 1))     # 随机生成被介入的目标节点
    interv_matrix = np.tile(interv_target, (1, p)) * GC * strength      # tile函数将target进行复制拓展
    interv_beta = beta + interv_matrix  # 维度5*5

    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))     # 生成标准差为0.1，均值为0的正态分布，其维度为5*1100
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        if interv and t > anomaly:  # anomaly：干预开始的节点
            X[:, t] = np.dot(interv_beta, X[:, (t - lag):t].flatten(order='F'))
            alpha = 0.1  # Leaky ReLU parameter
            X[:, t] = np.where(X[:, t] > 0, X[:, t], alpha * X[:, t])
        else:
            X[:, t] = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))    # 利用AR模型计算出下一步X
            alpha = 0.1  # Leaky ReLU parameter
            X[:, t] = np.where(X[:, t] > 0, X[:, t], alpha * X[:, t])   # 如果X第t列的元素大于0，则保持不变，否则乘以0.1
        X[:, t] += + errors[:, t - 1]

    return X.T[burn_in:], interv_target, interv_beta    # burn_in指预热阶段的持续时间，预热阶段的值需要舍弃，以保证模型的稳定性

### 5.VAR模型的作用

> VAR模型是为了仿真节点之间的联系
