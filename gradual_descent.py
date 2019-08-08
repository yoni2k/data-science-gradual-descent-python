import numpy as np
import pandas as pd

# how many observations prepare in the file
num_obs_file = 1000000
inputs_file_name = 'inputs.csv'
# weights_array = [-0.1, -0.01, -0.001, -0.0001, 0.0001, 0.001, 0.01, 0.1]
weights_array = [-0.1, -0.001, 0.001, 0.1]
funcs_params_array = [[2, -3, 5], [13, 7, -12]]
# num_obs_array = [1000, 10000, 100000]
num_obs_array = [1000, 10000]
# learn_rate_array = [0.0001, 0.001, 0.1, 1]
learn_rate_array = [0.001, 0.01, 0.1] # 0.1 and higher (possibly lower also doesn't succeed)
# num_iter_array = [1000, 10000, 1000000]
num_iter_array = [1000, 10000]


def prepare_data():
    xs = np.random.uniform(low=-10, high=10, size=(num_obs_file, 1))
    zs = np.random.uniform(-10, 10, (num_obs_file, 1))
    noise = np.random.uniform(-1, 1, (num_obs_file, 1))

    data = np.column_stack((xs, zs, noise))

    df = pd.DataFrame(data,columns=['xs', 'zs', 'noise'])

    print(f"xs: {xs[0:5].T}")
    print(f"zs: {zs[0:5].T}")
    print(f"noise: {noise[0:5].T}")
    print(f'Before writing to csv file:\n{df.head()}')

    df.to_csv(inputs_file_name)


def get_data():
    df = pd.read_csv(inputs_file_name)
    print(f'After reading from csv file:\n{df.head()}')

    xs = np.array(df['xs'])
    print(f'Read xs: {xs[0:5].T}')

    zs = np.array(df['zs'])
    print(f'Read zs: {zs[0:5].T}')

    noise = np.array(df['noise'])
    print(f'Read noise: {noise[0:5].T}')

    return xs, zs, noise


def prepare_targets(xs, zs, noise, func_params):
    targets = func_params[0] * xs + func_params[1] * zs + func_params[2] + noise
    return targets


def do_single_descent(inputs, targets, num_obs, num_iter, init_weights, init_bias, learn_rate, i):
    # print(f'{i}: Doing single descent: num_obs: {num_obs}, num_iter: {num_iter}, '
    #          f'init_w1: {init_w1}, init_w2: {init_w2}, bias: {init_bias}, learn_rate: {learn_rate}')

    weights, bias = init_weights, init_bias

    for i in range(num_iter):
        # This is the linear model: y = xw + b equation
        outputs = np.dot(inputs, weights) + bias

        # The deltas are the differences between the outputs and the targets
        # Note that deltas here is a vector <num_obs> x 1
        deltas = outputs - targets

        # We are considering the L2-norm loss, but divided by 2, so it is consistent with the lectures.
        # Moreover, we further divide it by the number of observations.
        # This is simple rescaling by a constant. We explained that this doesn't change the optimization logic,
        # as any function holding the basic property of being lower for better results, and higher for worse results
        # can be a loss function.
        loss = np.sum(deltas ** 2) / 2 / num_obs

        # Another small trick is to scale the deltas the same way as the loss function
        # In this way our learning rate is independent of the number of samples (observations).
        # Again, this doesn't change anything in principle, it simply makes it easier to pick a single learning rate
        # that can remain the same if we change the number of training samples (observations).
        # You can try solving the problem without rescaling to see how that works for you.
        deltas_scaled = deltas / num_obs

        # Finally, we must apply the gradient descent update rules from the relevant lecture.
        # The weights are 2x1, learning rate is 1x1 (scalar), inputs are 1000x2, and deltas_scaled are 1000x1
        # We must transpose the inputs so that we get an allowed operation.
        weights = weights - learn_rate * np.dot(inputs.T, deltas_scaled)
        bias = bias - learn_rate * np.sum(deltas_scaled)
    return weights[0][0].round(3), weights[1][0].round(3), bias.round(3), loss.round(3)


def do_multiple_parametrized_descents():
    xs, zs, noise = get_data()
    i = 0

    results_list = []

    for funcs_params in funcs_params_array:
        targets = prepare_targets(xs, zs, noise, funcs_params)
        print(f'targets: {targets[0:5].T}')
        for num_obs in num_obs_array:
            inputs = np.column_stack((xs[0:num_obs], zs[0:num_obs]))
            targets_num_obs = targets[0:num_obs].reshape(num_obs,1)
            for learn_rate in learn_rate_array:
                for init_w1 in weights_array:
                    for init_w2 in weights_array:
                        init_weights = np.array([[init_w1], [init_w2]])
                        for init_bias in weights_array:
                            for num_iter in num_iter_array:
                                i += 1
                                w1, w2, bias, loss = do_single_descent(
                                    inputs, targets_num_obs,
                                    num_obs, num_iter, init_weights, init_bias, learn_rate, i)
                                print(f'{i} after : num_obs: {num_obs}, num_iter: {num_iter}, learn_rate: {learn_rate}, '
                                      f'init_w1: {init_weights[0][0]}, init_w2: {init_weights[1][0]}, init_bias: {init_bias}, '
                                      f'Dw1: {(funcs_params[0] - w1).round(3)}, '
                                      f'Dw2: {(funcs_params[1] - w2).round(3)}, '
                                      f'Dbias: {(funcs_params[2] - bias).round(3)}, '
                                      f'loss: {loss}, w1: {w1}, w2: {w2}, bias: {bias}')
                                dic_result = {'Func params': funcs_params,
                                              'Num observations': num_obs,
                                              'Learn rate': learn_rate,
                                              'Init w1': init_w1,
                                              'Init w2': init_w2,
                                              'Init bias': init_bias,
                                              'Num iterations': num_iter,
                                              'w1 delta': (funcs_params[0] - w1).round(3),
                                              'w2 delta': (funcs_params[1] - w2).round(3),
                                              'bias delta': (funcs_params[2] - bias).round(3),
                                              'loss': loss,
                                              'w1': w1,
                                              'w2': w2,
                                              'bias': bias,
                                              }
                                results_list.append(dic_result)
    results = pd.DataFrame(results_list)
    results.to_csv('output.csv')
    print(results.head())




prepare_data()
do_multiple_parametrized_descents()