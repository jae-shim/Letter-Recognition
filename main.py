from Neural_Network import NN
import numpy as np

def main() -> None:
    data = open("Dataset/letter-recognition.data", "r")
    inputs = np.empty([16000, 16], 'double')
    outputs = np.empty([16000, 1], 'int')
    for x in range(16000):
        letter = data.readline().split(',')
        inputs[x] = letter[1:]
        outputs[x] = ord(letter[:1][0].lower()) - 96
    yeet = NN()
    yeet.train(inputs, outputs, 1000)
    tinputs = np.empty([4000, 16], 'double')
    toutputs = np.empty([4000, 1], 'int')
    for x in range(4000):
        letter = data.readline().split(',')
        tinputs[x] = letter[1:]
        toutputs[x] = ord(letter[:1][0].lower()) - 96
    print(yeet.think(tinputs))

if __name__ == "__main__":
    main()
