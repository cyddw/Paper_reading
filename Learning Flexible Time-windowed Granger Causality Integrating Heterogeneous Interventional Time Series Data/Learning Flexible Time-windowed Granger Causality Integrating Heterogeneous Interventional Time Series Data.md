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

## [近端梯度下降法](https://www.bilibili.com/video/BV1AS4y1q76x/?spm_id_from=333.337.search-card.all.click&vd_source=887e79a2964e5ce84cbcf68e50febd27)

### 为什么提出近端梯度下降？

> 对于凸优化问题，当其目标函数存在不可微部分(例如目标函数中有1-范数或迹范数)时，近端梯度下降法才会派上用场。

## [KKT条件](https://zhuanlan.zhihu.com/p/38163970)

## 疑问

### 1.Paper中的Heterogeneous体现在哪里？

### 2.Paper中的不同的env是如何生成的？

> 循环调用5次以生成5个不同的Non_Linear data

### 3.Intervention体现在哪？

> <img width="725" alt="image" src="https://github.com/user-attachments/assets/47c77199-b37e-4aee-9250-6ea798b367f2">

### 4.分析Non-Linear代码部分

> 确认模型以及模型参数：

> <img width="166" alt="image" src="https://github.com/user-attachments/assets/4015082f-5bac-49ae-84f9-cd3b2efe649f">

> 其中lag=1，表示t时刻的节点只取决于t-1时刻，fi(X)=ReLU(beta*X)，噪声为均值为0，标准差为0.1的正态分布

> 当T>500时，开始介入，即beta变为beta_interv(interv_beta的生成：随机生成interv_target，对interv_target施加strength，得到的结果和原来的beta相加)


### 5.VAR模型的作用

> VAR模型是为了仿真节点之间的联系

### 6.因果图是如何获取的？

> 通过计算model的ref_layer的权重的二范数得到

### 7.IF_est代表什么？

### 8.为什么通过计算ref_layer和inv_layer的权重的二范数可以得到GC和IF？

### 9."elam=0.0, glam=0.0, lam_ridge=0.0, lookback=5, check_every=20, verbose=1"参数说明

### 10.为什么模型预测部分输入(1,999,5)，输出(1,999,1)?


