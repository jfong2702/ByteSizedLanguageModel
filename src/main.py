import os
from data import Data


curr_path = os.getcwd()
file_path = os.path.join(curr_path, "dataset\\Tweets.csv")
dataset = Data(file_path).data_file


def main():
    print(dataset[0])

if __name__ == "__main__":
    main()