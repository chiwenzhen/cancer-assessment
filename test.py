import numpy as np

class App:
    name = None
    age = None
    def __init__(self):
        self.name = "james"
        self.age = 12

    def name(self):
        print(self.name)
    def age(self):
        print(self.age)