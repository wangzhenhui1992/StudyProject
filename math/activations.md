# 激活函数

人工神经网络中，主要用来将线性输入实现非线性映射的函数。

### Sigmoid

S型生长曲线,范围在(0,1)之间

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def show_sigmoid():
        x = np.linspace(start=-10, stop=10, num=256)
        y = sigmoid(x)
        plt.title("sigmoid")
        plt.plot(x, y)
        plt.show()
    
    show_sigmoid()
    

![sigmoid][1]

### Tanh

双曲正切函数，范围在(-1,1)之间

    def tanh(x):
        p1 = np.exp(x)
        p2 = np.exp(-x)
        return (p1 - p2) / (p1 + p2)
    
    def show_tanh():
        x = np.linspace(start=-5, stop=5, num=256)
        y = tanh(x)
        plt.title("tanh")
        plt.plot(x, y)
        plt.show()
    
    show_tanh()
    

![tanh][2]

### ReLu

修正线性单元,小于0时等于0，大于0时等于X

    def relu(x):
        return np.where(x > 0, x, 0)
    
    def show_relu():
        x = np.linspace(start=-5, stop=5, num=256)
        y = relu(x)
        plt.title("relu")
        plt.plot(x, y)
        plt.show()
    
    show_relu()
    

![relu][3]

### Leaky ReLu

ReLu的变种，小于0时也会有负数值，但是幅度特别小，大于0时等于自身

    def leakyrelu(x, rate):
        return np.where(x > 0, x, rate * x)
    
    def show_leakyrelu():
        x = np.linspace(start=-10, stop=10, num=256)
        y = leakyrelu(x, 1e-1)
        plt.title("leakyrelu")
        plt.plot(x, y)
        plt.show()
    
    show_leakyrelu()
    

![leakyrelu][4]

 [1]: https://github.com/wangzhenhui1992/StudyProject/blob/master/math/sigmoid.png?raw=true
 [2]: https://github.com/wangzhenhui1992/StudyProject/blob/master/math/tanh.png?raw=true
 [3]: https://github.com/wangzhenhui1992/StudyProject/blob/master/math/relu.png?raw=true
 [4]: https://github.com/wangzhenhui1992/StudyProject/blob/master/math/leakyrelu.png?raw=true
