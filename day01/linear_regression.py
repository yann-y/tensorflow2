#conding:utf-8
import  numpy as  np
import matplotlib.pyplot as plt
'''
线性回归（linear  regression）
'''
def compute_error_for_line_given_points(b,w,points):
    totalError = 0
    # 循环叠加每一个点的误差。
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        #computer mean-squared-error
        # ** 2表示平方
        totalError += (y-(w*x+b)) ** 2
    return  totalError / float(len(points))
def step_gradient(b_current,w_current,points,learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        # grad_b = 2(wx+b-y)
        b_gradient += (2/N) * ((w_current * x + b_current) - y )
        # grad_W = w(wx+b-y)*x
        w_gradient += (2/N) * x * ((w_current * x + b_current) - y)
    # update w'
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b,new_w]
def gradient_descent_runner(points,starting_b,starting_w,learning_rate,num_iterations):
    b = starting_b
    w = starting_w
    # update for serveral times
    # num_iterations 循环次数。
    for i in range(num_iterations):
        b,w = step_gradient(b,w,np.array(points),learning_rate)
    return [b,w]
def  run():
    # 生成随机训练集
    x = np.random.randint(-50, 160, size=130)
    y = 2.5 * x + 3.2
    points = np.column_stack((x, y))
    # 初始参数
    learning_rate = 0.0001 #学习率
    initial_b = 0
    initial_w = 0
    num_iterations = 1000 # 迭代次数。
    print("Starting gradient descent descrnt at b = {0}, w= {1},error = {2}"
            .format(initial_b,initial_w,compute_error_for_line_given_points(initial_b,initial_w,points)))
    print("runing...")
    [b,w] = gradient_descent_runner(points,initial_b,initial_w,learning_rate,num_iterations)
    print("after {0} iterations b = {1},w={2},error ={3}"
        .format(num_iterations,b,w,compute_error_for_line_given_points(b,w,points)))

    # 绘制散点图
    Scatter_plot(x, y, w, b)

def Scatter_plot(x,y,w,b):
    plt.scatter(x, y)
    plt.title('figure')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.plot(x, x * w + b, 'm', linewidth=2)
    plt.show()

if __name__ == '__main__':
    run()