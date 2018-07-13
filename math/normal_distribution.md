>> 原文发布在个人博客上  http://www.soaringroad.com/?p=1070

### 正态分布

*   正态分布，也被称作高斯分布，日常生活中很多现象的概率分布符合正态分布。
*   X服从数学期望即均值为μ，标注差为σ的正态概率分布，则记为x∈N(μ,σ^2)
*   μ决定了分布的位置，σ决定了分布的幅度
*   概率密度函数 f(x) = exp(-(x-μ)^2/2/σ^2)/sqrt(2*pi)/σ
*   通过python将函数曲线图画出来
    
    x = np.linspace(-5, 5, 512, True) mu_sigma_set = [(0, 1), (1, 1), (2, 1), (0, 2), (0, 3)] for mu, sigma in mu_sigma_set: y = np.exp(-np.square(x - mu)/2/np.square(sigma))/np.sqrt(2*np.pi)/sigma plt.plot(x,y) plt.legend(["0-1","1-1","2-1","0-2","0-3"]) plt.show()

*   ![image][1]

### 正态分布的性质

*   距离μ一个标准差的概率分布面积为68.268949%
*   距离μ两个标准差的概率分布面积为95.449974%
*   距离μ三个标准差的概率分布面积为99.730020%
*   ![image][2]

### 标准正态分布

*   当μ=0,σ=1的时候，称为标准正态分布，即f(x) = exp(-x^2/2)/sqrt(2*pi)

 [1]: https://github.com/wangzhenhui1992/StudyProject/blob/master/math/normal_distribution.png?raw=true
 [2]: https://github.com/wangzhenhui1992/StudyProject/blob/master/math/normal_distribution2.png?raw=true
