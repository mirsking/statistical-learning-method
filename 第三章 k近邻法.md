# 统计学习方法-第三章 k近邻法

标签（空格分隔）： 学习笔记 统计学习方法

---

## [在线链接](https://www.zybuluo.com/mirsking/note/197452)

[toc]

**k近邻法（k-nearest neighbor, kNN）**是一种基本的分类和回归算法。本章只涉及kNN在分类上的应用。
## 模型
特征空间中，每个训练实例点$x_i$，距离该点比其它点更近的所有点组成**单元（cell）**。
所有训练实例点的单元构成对特征空间的一个划分。
最近邻法将实例$x_i$的类$y_i$作为其单元中所有点的类标记(class label)。
![kNN 模型][1]

-----

## 策略
### 分类决策规则
#### 多数表决
误分类概率，预测类别是$c_j$
$$P(Y\neq f(X)) = 1- P(Y=f(X))$$ $$\frac1k\sum_{x_i\in N_k(x)}I(y_i\neq c_j) = 1 - \frac1k\sum_{x_i\in N_k(x)}I(y_i=c_j)$$
多数表决对应于$\frac1k\sum_{x_i\in N_k(x)}I(y_i=c_j)$最大，也就意味着$\frac1k\sum_{x_i\in N_k(x)}I(y_i\neq c_j)$最小，也即经验风险最小化。故kNN常用的多数表决分类决策规则，即是经验风险最小化。

-----

## 算法
> 输入： 训练数据集$T=\{(x_1, y_1), (x_2,y_2),...(x_N,y_N)\}$，实例特征向量$x$
> 输出： 实例$x$所属的类
> 算法：
> > Step 1. 根据**距离度量**，在$T$中找到与$x$距离最近的$k$个点，$(x_j,y_j)\in N_k(x), j=1,...,k$。
> > Step 2. 根据**分类决策规则**，决定$x$的类别$y$。$$y=argmax_{c_j} \sum_{x_i\in N_k(x)}I(y_i=c_j), i=1,2,...,N;j=1,2,...,k$$

$k=1$时，作为kNN法的特例，称为**最近邻算法**。

### 距离度量
#### $L_p$距离（也叫闵可夫斯基(Minkowski)距离）
$$L_p(x_i,x_j)=(\sum_{l=1}^{n}|x_i^{(l)}-x_j^{(l)}|^p)^{\frac1p},p\geq 1$$
* $p=2$， 欧式距离$$L_p(x_i,x_j)=(\sum_{l=1}^{n}|x_i^{(l)}-x_j^{(l)}|^2)^{\frac12}$$
* $p=1$， 曼哈顿距离$$L_p(x_i,x_j)=\sum_{l=1}^{n}|x_i^{(l)}-x_j^{(l)}|$$
* $p=\infty$， 切比雪夫距离（$L_\infty$距离，各坐标距离的最大值）$$L_p(x_i,x_j)=max_l|x_i^{(l)}-x_j^{(l)}|$$
$L_p$距离之间的关系图示
![L_p距离][2]
### k值选择
* k较小
    * 学习的近似误差（Approximation error）小：与输入实例较近（相似）的训练实例才会对预测结果起作用
    * 学习的估计误差（Estimation error）大：预测结果对近邻的实例点敏感，若近邻实例点刚好是噪声，则预测即会出错。
    * k值得减小意味着整体模型变得复杂，容易放生过拟合
* k较大
    * 学习的近似误差大
    * 学习的估计误差小
    * k值得增大意味着模型的简单

应用中，k值一般取一个较小的数值，通过**交叉验证法**取最优的k值。

### kd树
#### 构建kd树
> 输入： exm_set样本集
> 输出：kd树
> 这是一个递归的算法
> 算法：
> > 1. 如果exm_set为空，则返回空的kd树
> > 2. 对exm_set进行结点分裂，得到 1) dom_elt : exm_set的一个样本点； 2) split: 分裂维的序号
> > 3. exm_set_left = {exm_set - dom_elt && exm_set[split] <= dom_elt[split]}
exm_set_right = {exm_set - dom_elt && exm_set[split] > dom_elt[split]}
> > 4. left = createKDTree(exm_set_left)
right = createKDTree(exm_set_right)

##### 细节
1. 分裂结点的选择：一般对所有样本点，统计他们在每个维上的方差，挑选出方差中的最大值，对应的维是split的维度。

#### 搜索kd树
> 输入： kd树
> 输出： 最近邻
> 算法：
> > 1. 如果kd树是空的，则设dist为无穷大返回
> > 2. 向下搜索知道叶子结点
```
pSearch = &kd
while(pSearch!=NULL)
{
    save pSearch into search_path
    if(target[pSearch->split] <= pSearch->dom_elt[pSearch->split])
        pSearch = pSearch->left;
    else
        pSearch = pSearch->right;
}

nearest = search_path[last];
dist = Distance(nearest, target);
```
> > 3. 回溯搜索路径
```
while(search_path非空)
{
    pBack = search_path[last];
    if(pBack->left == NULL && pBack->right == NULL)//叶子结点
    {
        if(dist > Distance(pBack->dom_elt, target))
        {
            nearest = pBack->dom_elt;
            dist = Distance(pBack->dom_elt, target);
        }
    }
    else
    {
        s = pBack->split;
        if(abs(pBack->dom_elt[s]-target[s]) < dist)//如果以target为中心的圆（球、超球），半径为dist的圆与分割超平面相交，则要调到另一边的子空间去搜索
        {
            nearest = pBack->dom_elt;
            dist = Distance(pBack->dom_elt, target);
            if(target[s] <= pBack->dom_elt, target)// 如果target位于pBack的左子空间，则要跳到右子空间去搜索
                pSearch = pBack->right;
            else
                pSearch = pBack->left;
                
            if(pSearch != NULL)
                save pSearch into search_path
        }
    }
}
```


  [1]: http://7xnluw.com1.z0.glb.clouddn.com/statistical_learning_method/kNN_model.png "kNN 模型"
  [2]: http://7xnluw.com1.z0.glb.clouddn.com/statistical_learning_method/L_p%20distance.png "L_p distance"