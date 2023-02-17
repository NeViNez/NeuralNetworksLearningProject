import os

import PythonBasics
import binaryClassNN
import numpyBasics
import ipfinding
import discretteClassifyingNeural as DCN
import neuralRegresion as regression
import convolutional
import orcestralNN

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # print("Task 1")
    # print("Enter the string. The program will check if any symbol in it presents two times or more.")
    # while True:
    #     x = input()
    #     h = PythonBasics.pBasics(x)
    #     if not h:
    #         print("There are unique symbols in the string")
    #     else:
    #         print("There are dublicating symbols in the string")
    # ipfinding.ipdron()

    # print("Task 2")
    # binaryClassNN.neuralNetwork()

    # print("Task 3")
    # canvas = numpyBasics.circle(4, 3, 4, 7)
    # with open('out_2.3.6.txt', 'w') as f:
    #     for x in canvas:
    #         for y in x:
    #             print(y, end='', file=f)
    #         print("\n", file=f)
    # f.close()

    # print("Task 4")
    # DCN.discretteNeural()

    print("Task 5")
    sine_dataset_path = os.getcwd() + "/5_1_dataset_sine.csv"
    linear_dataset_path = os.getcwd() + "/5_1_dataset_linear.csv"
    repeats = 10000
    # Для генерации датасета переключить в еrue
    generate_dataset = False
    if generate_dataset:
        regression.dataset_generation(sine_dataset_path, regression.sine_function, repeats=repeats)
        regression.dataset_generation(linear_dataset_path, regression.linear_function, repeats=repeats)
    regression.linear_regression_fit(linear_dataset_path, repeats)
    regression.linear_regression_fit(sine_dataset_path, repeats)

    # print("Task 6")
    # convolutional.doConv()

    # print("Task 8")
    # orcestralNN.doOrcestral()
