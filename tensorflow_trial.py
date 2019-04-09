# coding: utf-8

import tensorflow as tf
x = tf.Variable(10.0, trainable = True)
f_x = 2 * x * x - 5 * x + 4
f_x
loss = f_x
opt = tf.train.GradientDescentOptimizer(0.1).minimize(f_x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        print(sess.run([x,loss]))
        sess.run(opt)
          
import sympy
x = sym.symbols('x')
import sympy as sym
x = sym.symbols('x')
x
f_x = 2 * x**2 - 5 * x + 4
f_x
f_x_tf = sym.lambdify(x, f_x, modules='tensorflow')
f_x_tf
f_x_tf.x
help(f_x_tf)
get_ipython().run_line_magic('pinfo', 'f_x_tf')
f_x_tf(tf.constant(1.0))
var = tf.Variable(1.0, trainable=True)
loss = f_x_tf(var)
loss
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        print(sess.run([x,loss]))
        sess.run(opt)
          
loss
get_ipython().run_line_magic('pinfo', 'loss')
with tf.Session() as sess:
    sess.run(loss)
    
var = tf.Variable(1.0, trainable=True)
var
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(f_x_tf(var))
        
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(f_x_tf(var)))
    
        
    
var
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([var, f_x_tf]))
    sess.run(tf.train.GradientDescentOptimizer(0.1).minimize(f_x_tf))     
    
    
f_x_tf(tf.constant(1.0))
f_x_t
f_x_tf
tf.convert_to_tensor(f_x_tf)
f_x_tf
f_x_tf(var)
f_x
y = tf.Variable(10.0, trainable=True)
y
f2 = 2 * y * y - 5 * y + 4
f2
f_x_tf(var)
loss = f_x_tf(var)
loss
tf.train.GradientDescentOptimizer(0.1).minimize(loss)
opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([var, loss]))
    sess.run(tf.train.GradientDescentOptimizer(0.1).minimize(f_x_tf))     
        
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([var, loss]))
    sess.run(opt)
        
            
    
f_x_tf
f_x
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        print(sess.run([var, loss]))
        sess.run(opt)
            
            
    
f_x_tf
f_x
a, b = sym.symbols('a b')
a
b
sym.lambdify([a,b], a + b + a * b**2)
f0 = sym.lambdify([a,b], a + b + a * b**2)
f0(1,2)
f0 = sym.lambdify([a,b], a + b + a * b**2 - a**2)
f0(1,2)
f0 = sym.lambdify([a,b], a + b + a * b**2 - a**2, module='tensorflow')
f0 = sym.lambdify([a,b], a + b + a * b**2 - a**2, 'tensorflow')
f0
var1 = tf.Variable(1.0)
var2 = tf.Variable(1.0)
loss = f0(var1, var2)
loss
var2 = tf.Variable(1.0, trainable=True)
var1 = tf.Variable(1.0, trainable=True)
loss = f0(var1, var2)
opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
opt
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        print(sess.run([var, loss]))
        sess.run(opt)
           
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        print(sess.run([var1,var2, loss]))
        sess.run(opt)
                 
loss
f0
f0 = sym.lambdify([a,b], a + b - a * b**2 - a**2, module='tensorflow')
f0 = sym.lambdify([a,b], a + b - a * b**2 - a**2, 'tensorflow')
loss = f0(var1, var2)
opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        print(sess.run([var1,var2, loss]))
        sess.run(opt)
                 
f0 = sym.lambdify([a,b], - a - b + a * b**2 +  a**2, 'tensorflow')
loss = f0(var1, var2)
opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        print(sess.run([var1,var2, loss]))
        sess.run(opt)
                 
f0 = sym.lambdify([a,b], 1 - a - b + b**2 +  a**2, 'tensorflow')
loss = f0(var1, var2)
opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        print(sess.run([var1,var2, loss]))
        sess.run(opt)
                       
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre, post = np.nan
    while abs(pre - post) > 1e-3:
        val  = sess.run([var1,var2, loss])
        print(val)
        pre = val[-1]
        sess.run(opt)
        post = val[-1]
         
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre, post = np.nan, np.nan
    while abs(pre - post) > 1e-3:
        val  = sess.run([var1,var2, loss])
        print(val)
        pre = val[-1]
        sess.run(opt)
        post = val[-1]
        
         
abs(np.nan - np.nan)
nan > 1e-3
np.nan > 1e-4
np.inf - np.inf
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre, post = 10, -10
    while abs(pre - post) > 1e-3:
        val  = sess.run([var1,var2, loss])
        print(val)
        pre = val[-1]
        sess.run(opt)
        post = val[-1]
                
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre, post = 10, -10
    while abs(pre - post) > 1e-3:
        val1  = sess.run([var1,var2, loss])
        print(val1)
        pre = val1[-1]
        val2 = sess.run([var1, var2, loss])
        sess.run(opt)
        post = val2[-1]
        
                
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre, post = 10, -10
    while abs(pre - post) > 1e-3:
        sess.run(opt)
        val1  = sess.run([var1,var2, loss])
        print(val1)
        pre = post
        post = val1[-1]        
                
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre, post = 10, -10
    while abs(pre - post) > 1e-5:
        sess.run(opt)
        val1  = sess.run([var1,var2, loss])
        print(val1)
        pre = post
        post = val1[-1]        
        
                
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pre, post = 10, -10
    while abs(pre - post) > 1e-10:
        sess.run(opt)
        val1  = sess.run([var1,var2, loss])
        print(val1)
        pre = post
        post = val1[-1]        
                        
get_ipython().run_line_magic('save', 'tensorflow_trial.py')
