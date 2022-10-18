import PythonBasics
import binaryClassNN
import numpyBasics

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # print("Task 1.")
    # print("Enter the string. The program will check if any symbol in it presents two times or more.")
    # while True:
    #     x = input()
    #     h = PythonBasics.pBasics(x)
    #     if not h:
    #         print("There are unique symbols in the string")
    #     else:
    #         print("There are dublicating symbols in the string")

    print("Task 2.")
    binaryClassNN.neuralNetwork()

    print("Task 3.")
    canvas = numpyBasics.circle(4, 3, 4, 7)
    with open('out_2.3.6.txt', 'w') as f:
        for x in canvas:
            for y in x:
                print(y, end='', file=f)
            print("\n", file=f)
    f.close()