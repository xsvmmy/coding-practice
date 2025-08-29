import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

training_set = pd.read_csv('data.csv')
x_train = training_set["YearsExperience"].values
y_train = training_set["Salary"].values

def cost_function(x, y, w, b):
    m = len(x)
    cost_sum=0
    for i in range(m):
        f = w * x[i] + b
        cost = (f-y[i])**2
        cost_sum += cost
    total_cost = (1/(2+m))*cost_sum
    return total_cost

# GRADIENT FUNCTION: compute derivatives of the cost function
def gradient_function(x,y,w,b):
    m = len(x)
    dc_dw, dc_db=0,0
    for i in range(m):
        f=w*x[i]+b
        dc_dw += (f-y[i])*x[i]
        dc_db += (f-y[i])
    dc_dw = (1/m)*dc_dw
    dc_db = (1/m)*dc_db
    return dc_dw,dc_db

def gradient_descent(x,y,alpha,iterations):
    w,b = 0,0
    for i in range(iterations):
        dc_dw, dc_db = gradient_function(x,y,w,b)
        w = w - alpha * dc_dw
        b = b - alpha * dc_db
        print(f"Iteration {i}: Cost {cost_function(x,y,w,b)}")
    return w,b

learning_rate = 0.01
iterations = 10000

final_w, final_b = gradient_descent(x_train,y_train,learning_rate,iterations)
print(f"w:{final_w:.4f},b: {final_b:.4f}")

plt.scatter(x_train,y_train,label='Data Points')
x_vals= np.linspace(min(x_train),max(x_train),100)
y_vals=final_w*x_vals+final_b
plt.plot(x_vals,y_vals,color='red',label='Regression Line')
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()