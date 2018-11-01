# Alnahhas, Faisal
# Date of submission (2018-10-29)
# Assignment-04-01
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

delay = 10
stride = 1
num_iterations = 10
split_size = 0.8

def calculate_activation_function(net_value, type):
    if type == "Linear":
        activation = net_value
    elif type == "Tanh":
        activation = np.tanh(net_value)
    elif type == "Hardlims":
        activation = net_value
        activation[activation<0] = 0
        activation[activation >= 0] = 1
    return activation

def read_csv_matrix(file_name):
    data = np.loadtxt(file_name, skiprows = 1, delimiter= ',', dtype=np.float32)
    return data
data = read_csv_matrix('data_set_2.csv')

def normalize(data_in):
    for x in range(len(data_in)):
    # for x in range(len(data_in)):
        data_in[x] = (2 * ((data_in[x] - np.min(data_in))/(np.max(data_in)-np.min(data_in))))-1
    return data_in

price = data[:, :1]
volume = data[:,1:]

normalized_price = normalize(price)
normalized_volume = normalize(volume)

normalized_data = np.hstack([normalized_price, normalized_volume])

def split():
    idx = int(split_size * normalized_data.shape[0])
    train, test = np.split(normalized_data, [idx])
    return train, test
train, test = split()

normalized_price_train = train[:, 0]
normalized_volume_train = train[:,1]
normalized_price_test = test[:, 0]
normalized_volume_test = test[:,1]

def make_input():
    stride = 1
    current = delay + 1
    current2 = delay + 1
    input_array_price = []
    input_array_vol = []
    input_array = []
    target_train = []

    output_array_price = []
    output_array_vol = []
    output_array = []
    target_test = []

    # for i in range(10):
    while current <= len(normalized_price_train) - 2:
        input_array_price = normalized_price_train[current - delay - 1:current]
        #     print(input_array_price)
        input_array_price = input_array_price.tolist()
        input_array_price.reverse()
        #     print(type(input_array_price))

        #     output_array_price = normalized_price_test[current-delay-1:current]
        # #     print(output_array_price)
        #     output_array_price = output_array_price.tolist()
        #     output_array_price.reverse()
        # #     print(type(output_array_price))

        input_array_vol = normalized_volume_train[current - delay - 1:current]
        input_array_vol = input_array_vol.tolist()
        input_array_vol.reverse()

        #     output_array_vol = normalized_volume_test[current-delay-1:current]
        #     output_array_vol = output_array_vol.tolist()
        #     output_array_vol.reverse()

        #     print("\n")
        combined = input_array_price + input_array_vol
        #     print("combined: ", combined)

        #     print("\n")
        input_array.append(combined)
        target_train.append(normalized_price_train[current + 1])

        #     print("\n")
        #     combined_test = output_array_price + output_array_vol
        # #     print("combined_test: ", combined_test)

        # #     print("\n")
        #     output_array.append(combined_test)

        #     target_test.append(normalized_price_test[current+1])

        current = current + stride

    print(len(combined))
    while current2 < len(normalized_price_test) - 2:
        output_array_price = normalized_price_test[current2 - delay - 1:current2]
        #     print(output_array_price)
        output_array_price = output_array_price.tolist()
        output_array_price.reverse()
        #     print(type(output_array_price))
        output_array_vol = normalized_volume_test[current2 - delay - 1:current2]
        output_array_vol = output_array_vol.tolist()
        output_array_vol.reverse()

        combined_test = output_array_price + output_array_vol
        #     print("combined_test: ", combined_test)

        #     print("\n")
        output_array.append(combined_test)

        target_test.append(normalized_price_test[current2 + 1])
        current2 = current2 + stride

    target_train = np.asarray(target_train)
    input_array = np.asanyarray(input_array)

    target_test = np.asarray(target_test)
    output_array = np.asanyarray(output_array)
    print(target_train.shape)
    print(input_array.shape)
    print(output_array.shape)
    return input_array, output_array, target_train, target_test
input_array, output_array, target_train, target_test = make_input()


def make_weight(in_array):
    W = np.zeros([in_array.shape[0], in_array.shape[1]])
    print("W.shape: ")
    print(W.shape)
    return W
W = make_weight(input_array)


def direct(input_array, target_train):
    R = (np.dot(input_array.T, input_array)) * (1 / len(input_array))
    h = (np.dot(target_train.T, input_array)) / len(input_array)
    x_star = np.dot(inv(R), h)
    return x_star
x_star = direct(input_array, target_train)

def direct_error():
    direct_MSE_MAE = []
    net_value = np.dot(x_star, output_array.T)
    direct_error = target_test - net_value
    MSE_direct = (np.sum(direct_error ** 2)) / len(direct_error)
    MAE_direct = (np.sum(np.abs(direct_error))) / len(direct_error)
    direct_MSE_MAE.append([MSE_direct, MAE_direct])
    return MSE_direct, MAE_direct
MSE_direct, MAE_direct = direct_error()


def LMS(W):
    MSE_LMS_list = []
    MAE_LMS_list = []
    for i in range(num_iterations):
        LMS_net_value_in = np.dot(W, input_array.T)
        LMS_in_error = target_train - LMS_net_value_in
        W = W + 2 * 0.1 * np.dot(LMS_in_error, input_array)
        LMS_net = np.dot(W, output_array.T)
        LMS_err = target_test - LMS_net
        MSE_LMS = (np.sum(LMS_err ** 2)) / len(LMS_err)
        MAE_LMS = (np.sum(np.abs(LMS_err))) / len(LMS_err)
        MSE_LMS_list.append(MSE_LMS)
        MAE_LMS_list.append(MAE_LMS)


        # LMS_net_value_in = np.dot(W, input_array.T)
        # LMS_in_error = target_train - LMS_net_value_in
        # W = W + 2 * 0.1 * np.dot(LMS_in_error, input_array)
        # LMS_net = np.dot(W, output_array.T)
        # LMS_err = target_test - LMS_net
        # MSE_LMS = (np.sum(LMS_err ** 2)) / len(LMS_err)
        # MAE_LMS = (np.sum(np.abs(LMS_err))) / len(LMS_err)
        # MSE_LMS_list.append(MSE_LMS)
        # MAE_LMS_list.append(MAE_LMS)
        # LMS_net_value_in = np.dot(W, input_array.T)
        # LMS_in_error = target_train - LMS_net_value_in
        # W = W + 2 * 0.1 * np.dot(LMS_in_error, input_array)
        # LMS_net = np.dot(W, output_array.T)
        # LMS_err = target_test - LMS_net
        # MSE_LMS = (np.sum(LMS_err ** 2)) / len(LMS_err)
        # MAE_LMS = (np.sum(np.abs(LMS_err))) / len(LMS_err)
        # MSE_LMS_list.append(MSE_LMS)
        # MAE_LMS_list.append(MAE_LMS)
        # print(MAE_LMS_list)
        print("LMS: ")
        print(MAE_LMS_list)
        return MSE_LMS_list, MAE_LMS_list
LMS_MSE, LMS_MAE = LMS(W)


##Make plot
fig = plt.figure(figsize=(7, 6))
ax = fig.gca()
plt.subplots_adjust(left=0.3, right=0.9, bottom = 0.3, top = 0.9, wspace = 0.2, hspace = 0.2)
plt.axis([0, 1000, 0, np.max(LMS_MSE)])
plt.xlabel("epoch")
plt.ylabel("error")
range = np.arange(100)
fig.canvas.set_window_title('Alnahhas_04')

axcolor = 'lightgoldenrodyellow'

##buttons
weights = plt.axes([0.01, 0.35, 0.20, 0.04])
weight_button = Button(weights, 'Make W', color=axcolor, hovercolor='0.975')

LMS = plt.axes([0.01, 0.45, 0.20, 0.04])
LMS_button = Button(LMS, 'LMS', color='lightblue', hovercolor='0.975')

Direct = plt.axes([0.01, 0.50, 0.20, 0.04])
Direct_button = Button(Direct, 'Direct', color='lightblue', hovercolor='0.975')

clear = plt.axes([0.01, 0.90, 0.20, 0.04])
clear_button = Button(clear, 'Clear', color='lightgreen', hovercolor='0.975')

##slider
axalpha = plt.axes([0.2, 0.05, 0.50, 0.03], facecolor=axcolor)
alpha_slider = Slider(axalpha, 'alpha', 0.001, 1.0, valinit=0.1)

axdelay = plt.axes([0.2, 0.1, 0.50, 0.03], facecolor=axcolor)
delay_slider = Slider(axdelay, 'delay', 0, 100, valinit=10)

axtrain = plt.axes([0.2, 0.15, 0.50, 0.03], facecolor=axcolor)
train_slider = Slider(axtrain, 'training %', 0, 100, valinit=80)

axstride = plt.axes([0.2, 0.2, 0.50, 0.03], facecolor=axcolor)
stride_slider = Slider(axstride, 'stride', 1, 100, valinit=1)

axiteration = plt.axes([0.2, 0.25, 0.50, 0.03], facecolor=axcolor)
iteration_slider = Slider(axiteration, 'num iterations', 1, 100, valinit=10)

# ##radio
# rax1 = plt.axes([0.01, 0.75, 0.2, 0.15], facecolor=axcolor)
# radio_activation = RadioButtons(rax1, ('Linear', 'Hardlims', 'Tanh'), active=0)
#
# rax2 = plt.axes([0.01, 0.55, 0.2, 0.15], facecolor=axcolor)
# radio_hebb = RadioButtons(rax2, ('Smoothing', 'Delta', 'Unsupervised'), active=0)

def weight_caller(event):
    make_weight(input_array)
weight_button.on_clicked(weight_caller)


def direct_caller(event):
    err_copy = MSE_direct
    err_copy2 = MAE_direct
    ax.scatter(num_iterations, err_copy, marker=">")
    ax.scatter(num_iterations, err_copy2, marker="x")
Direct_button.on_clicked(direct_caller)

def LMS_caller(event):
    LMS_MSE_copy = LMS_MSE
    LMS_MAE_copy = LMS_MAE
    print("in LMS")
    print(LMS_MSE_copy)
    print(LMS_MAE_copy)
    ax.scatter(num_iterations, LMS_MSE_copy)
    ax.scatter(num_iterations, LMS_MAE_copy)
LMS_button.on_clicked(LMS_caller)
#
# def clear_caller(event):
#     ax.clear()
#     print("axes cleared")
# clear_button.on_clicked(clear_caller)
#
def update_alpha(val):
    alpha = alpha_slider.val
    fig.canvas.draw_idle()
    print(alpha)
alpha_slider.on_changed(update_alpha)

def update_delay(val):
    delay = delay_slider.val
    fig.canvas.draw_idle()
    print(delay)
delay_slider.on_changed(update_delay)

def update_sample(val):
    sample = train_slider.val
    fig.canvas.draw_idle()
    print(sample)
train_slider.on_changed(update_sample)

def update_stride(val):
    stride = stride_slider.val
    fig.canvas.draw_idle()
    print(stride)
stride_slider.on_changed(update_stride)

def update_iterations(val):
    iterations = stride_slider.val
    fig.canvas.draw_idle()
    print(iterations)
iteration_slider.on_changed(update_iterations)
#
#
# def hebb_rule_caller(label):
#     global hebb_rule
#     hebb_rule = label
# radio_hebb.on_clicked(hebb_rule_caller)
#
# def activation_function_caller(label):
#     global activation_function
#     activation_function = label
# radio_activation.on_clicked(activation_function_caller)
#

plt.show()


