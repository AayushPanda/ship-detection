import pandas as pd

from sklearn.model_selection import train_test_split

data = pd.DataFrame(pd.read_csv("data\\data_segmentations.csv", nrows=1000))


def classifyType(type):
    if type == "present":
        data['EncodedPixels'] = data['EncodedPixels'].apply(lambda input: not (isinstance(input, float)))
