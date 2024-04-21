# This wil server as the dataset class to easily acess data
import re
import pandas as pd

class Data():

    data_file = None

    def __init__(self, path):
        raw_data = pd.read_csv(path, usecols=['text'])
        self.sanitize_entries(raw_data['text'])
        self.data_file = raw_data['text']

    def sanitize_entries(self, entries):
        for i in range(len(entries)):
            entry = entries[i]
            if(type(entry) != str):
                entries.pop(i)
            else:
                entries[i] = re.sub(r"[^a-zA-Z\s]", '', entries[i])





