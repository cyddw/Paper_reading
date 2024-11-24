## 现有模型的不足

现有模型在进行Granger因果构建时，很少涉及imperfect intervention with unknown target的情况

## VAR模型

向量自回归模型把每个内生变量作为系统中所有内生变量滞后值的函数来构造模型，从而避开了结构建模方法中需要对系统每个内生变量关于所有内生变量滞后值的建模问题。

### 模型结构

<img width="763" alt="image" src="https://github.com/user-attachments/assets/90701a46-3d66-49c6-9adb-e71989e8c4d8">

<img width="963" alt="image" src="https://github.com/user-attachments/assets/942ee306-251c-4fb4-a20c-555a03d5da34">

### 模型平稳性

VAR模型的平稳性是一个重要的假设。只有在模型是平稳的情况下，VAR模型的估计结果才是可靠的。平稳性意味着时间序列的均值、方差和协方差在时间上不随时间变化。对于VAR模型来说，平稳性可以通过检查特征根来判断。一个VAR模型是平稳的，当且仅当模型的特征根的模小于1。

## 疑问

### Paper中的Heterogeneous体现在哪里？
