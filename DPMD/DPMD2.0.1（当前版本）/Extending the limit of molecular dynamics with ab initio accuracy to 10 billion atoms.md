# Extending the limit of molecular dynamics with ab initio accuracy to 10 billion atoms

### Deep Potential Model

#### 模型定义

![DPMD-1](.\图片\DPMD-1.png)

在N个原子的物理系统中，每个原子使用**原子位置**$r_i=(x_i,y_i,z_i)\in \R^3,\ i=1,2,\dots,N$表示。

DP模型假设原子$i$的势能$E_i$仅取决于它的邻居集合$\mathcal{R}_i=\{r_{ij}|j\in L_{R_c}(i)\}$，其中$r_{ij}=r_j-r_i$并且$L_{R_c}(i)$表示距离原子$i$截止半径$R_c$范围内相邻原子的索引集。如图1（a）所示。

DP模型的执行流程：

1. **环境矩阵（environment matrix）**$\tilde{\mathcal{R}_i}\in \R^{N_m\times4}$从原子$i$的邻居列表中生成，记录本地原子与邻居原子之间的相对位置，$N_m$是所有邻居列表的最大值
   $$
   \tilde{\mathcal{R}_i}=s(r_{ij})\times(1,x_{ij}/\lvert r_{ij}\rvert,y_{ij}/\lvert r_{ij}\rvert,z_{ij}/\lvert r_{ij}\rvert),
   $$
   其中$s(r_{ij})=\omega(\lvert r_{ij}\rvert)/\lvert r_{ij}\rvert$，$\omega(\lvert r_{ij}\rvert)$是一个当$\lvert r_{ij}\rvert\le R_c$时从1平滑衰减至0的门函数

2. 环境矩阵$\tilde{\mathcal{R}}_i$的第一列$s(r_{ij})$通过三层隐藏层embedding net生成**嵌入矩阵**$\mathcal{G}_i \in \R^{N_m\times M}$，记录原子间的距离，如图1（e）所示
   $$
   \mathcal{G}_i=\mathcal{L}^e_m\circ\cdots\circ\mathcal{L}^e_1\circ\mathcal{L}^e_0(s(r_{ij}))
   $$
   第一层是一个使用tanh为激活函数的标准全连接层：
   $$
   \mathcal{L}^e_0(x)=tanh(x\cdot W^e_0+b^e_0),\ \ \ W^e_0\in\R^{d_1},b^e_0\in\R^{d_1}
   $$
   其余各层为全连接层，采用快捷连接和tanh激活函数：
   $$
   \mathcal{L}^e_k(x)=(x,x)+tanh(x\cdot W^e_k+b^e_k)
   $$
   最终的输出大小为$M$，$M=4d_1$

3. 构造**满足三层不变性的描述符**$\mathcal{D}(\tilde{\mathcal{R}}_i)\in \R^{M^<\times M}$，它保留了物理对称性，如图1（f）所示的平移、旋转和置换不变性：
   $$
   \mathcal{D}(\tilde{\mathcal{R}}_i)=(\mathcal{G}_i^<)^T\tilde{\mathcal{R}}_i(\tilde{\mathcal{R}}_i)^T\mathcal{G}_i
   \\
   $$
   $\mathcal{G}_i^< \in \R^{N_m\times M^<}$是截取了$\mathcal{G}_i$矩阵的前$M^<$列的子矩阵。

   对称不变性描述符$\mathcal{D}$通过三层隐藏层fitting net $\mathcal{N}$生成**原子$i$的势能**$E_i$：$E_i=\mathcal{N}(\mathcal{D}(\tilde{\mathcal{R}}_i))$

4. 将所有单个势能相加得到系统的总势能$E = \sum_iE_i$，原子$i$的力由总势能的负梯度导出：$F_i=-\nabla_{r_i}E=-\sum_j\nabla_{r_i}E_j$

#### 计算密集部分与内存占用

DP模型中计算量最大的部分在于图1（e）中所示的$s(r_{ij})$通过嵌入网络计算得到嵌入矩阵$\mathcal{G}_i$。

嵌入网络的FLOPs为**$N_a\times(N_m\times d_1+10\times N_m\times d^2_1)$**，$N_a$为单个MPI任务上的原子数（对于铜原子$N_a$最大为4600，$N_m$为512；水的$N_m$为128），$d_1$为32。

因而，$\mathcal{G}_i$（维度为：$N_a\times N_m\times128$）是DPMD中最耗费内存的变量（4600个铜原子系统中，$\mathcal{G}_i$占用了2.4GB的GPU全局内存），与$\mathcal{G}_i$有关的内存占用占总占用的95%以上。

### 创新

使用上述公式（3）和（4）作为激活函数，嵌入网络就是一个高纬且异常复杂的**线性函数**。

#### 算法创新

![DPMD-2](.\图片\DPMD-2.png)

使用上述定理来近似嵌入网络。输入域平均分为n个区间，用节点$x_0\lt x_1\lt \cdots \lt x_n$表示。在第$\theta$个区间$(x_{\theta-1},x_{\theta}], \theta=1,2,\dots,n$，我们使用M个五阶多项式近似嵌入网络。
$$
f^{\eta}_{\theta}=a^{\eta}_{\theta0}+a^{\eta}_{\theta1}x+a^{\eta}_{\theta2}x^2+a^{\eta}_{\theta3}x^3+a^{\eta}_{\theta4}x^4+a^{\eta}_{\theta5}x^5
\\
a^{\eta}_{\theta\xi},\ \ \  \xi=0,1,2,3,4,5,\ \ \ \eta=1,2,\dots,M
$$
**tabulation操作**：将得到的多项式系数$a^{\eta}_{\theta\xi}$收集起来以表格的形式存储，使用在多项式近似来替代嵌入式网络计算。

查表操作的精确性由区间$(x_{\theta-1},x_{\theta}]$的大小来决定。当区间大小为0.001时达到了双精度限制。选择0.01为默认区间大小。

查表的计算量为$N_a\times 56\times N_m\times d_1$

#### 优化策略

![DPMD-3](.\图片\DPMD-3.png)

1. 通过收缩变量和合并计算来减少内存占用。

   通过将查表和矩阵乘操作合并，即$\tilde{\mathcal{R}}_i^Tf_\theta(s(r_{ij}))$，来优化$\mathcal{G}_i$的访问。（由图4可以看出是外积）

2. 消除冗余。

   ![DPMD-4](.\图片\DPMD-4.png)

   $N_m$在**水分子**系统中被设置为**128**，在**铜原子**系统中被设置为**512**，这就导致了可能会出现很多多余的0，如图4（a）中的$\mathcal{R}_i$

3. 通过提高每个MPI进程的计算粒度提高并行效率。

   当使用更多的MPI任务时，与ghost区域的通信量会随之增加。理论上，最好在每个节点上只启动一个MPI进程，并通过OpenMP、CUDA等将工作负载分配到更细的粒度。

#### 在GPU上实现优化策略

##### Kernel fusion

如图4所示，$\mathcal{G}_i$的每一行在一个线程块中计算并存储在寄存器中（没有存储回全局存储），然后环境矩阵$\tilde{\mathcal{R}}_i^T$的一列被加载到寄存器中以执行外积。这样$\mathcal{G}_i$没有在全局内存和寄存器之间进行分配和移动。

外积矩阵的大小是$4\times M$，并且可以在V100 GPU的共享内存中高效的累加。

#### 在ARM CPU上实现优化策略

##### 嵌入网络的查表

在之前的实现中，嵌入网络的模型以结构体的数组（AoS）的方式存储，即多项式的6个参数按行连续存储。但是由于内存访问不连续，所以AoS不能充分利用A64FX CPU的1024GB/s的带宽。

我们通过将每16个结构体转至以优化制表模块的数据分布，以便在访问表格时调用512位可扩展矢量扩展指令。

##### MPI+OPenMP

![DPMD-6](.\图片\DPMD-6.png)

图6（a）中的结构中每个MPI任务只能分配得到0.67GB大小的内存，因此会使得驻留在单个MPI任务上的子区域大小受到可用内存的极大限制。

图6（b）中运算符内部的MPI+OpenMP并行化并不高效，因为不同运算符之间频繁的forking和joining。

在图6（c）的方案里，每个OpenMP线程通过持有部分子区域来模拟MPI任务的行为。在每个MD步中只发生一次线程的forking和joining。在每个MPI任务中，只保留一份TensorFlow图副本并在OpenMP线程之间共享。上述优化使得计算粒度增加，MPI间的通信显著减少。

### 评测

#### 单个A64FX CPU

在单个Fugaku节点（A64FX）上测试18432个原子的水系统和2592个原子的铜系统。

![DPMD-Table2](.\图片\DPMD-Table2.png)

**如表2所示，对于水系统，单个A64FX的TtS为4.47 us/step/atom，单个V100的TtS为2.58 us/step/atom**

#### 强可扩展

![DPMD-9](.\图片\DPMD-9.png)

水系统（在Fugaku上有8294400个原子，在Summit上有4147200个原子）。

在Summit上4560节点的并行效率为46.99%，相应的TtS为6.0 ns/day；

在Fugaku上4560节点的并行效率为41.20%，相应的TtS为2.1 ns/day。

![DPMD-10](.\图片\DPMD-10.png)

铜系统（在Fugaku上有2177280个原子，在Summit上有13500000个原子。

在Summit上4560个节点的扩展效率为35.96%，TtS为11.2 ns/day；

在Fugaku上4560个节点的扩展效率为32.76%，TtS为4.7 ns/day；

#### 弱可扩展

![DPMD-11](.\图片\DPMD-11.png)

弱可扩展性时通过水系统和铜系统99MD步的系统大小和FLOPS来衡量的。

在Fugaku上，9936个节点上的水和铜系统的最大规模可以达到15.6亿和10.8亿。铜原子的TtS可以达到$4.1\times 10^{-11} second/step/atom$，相应的峰值性能可以达到119 PFLOPS（理论峰值的22.17%）；

在Summit上，水和铜系统的最大规模可以达到39亿和34亿。铜原子的TtS可以达到$1.1\times10^{-10} second/step/atom$，相应的峰值性能可以达到 43.7 PFLOPS（理论峰值的22.8%）。
