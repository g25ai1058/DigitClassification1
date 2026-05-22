from sklearn import datasets
from utils import load_data
# content of test_sample.py
def inc(x):
        return x+1
def test_answer():
        assert inc(4) == 5
def test_answer2():
        assert inc(4) == 5
class Test1:
    def test_something(self):
         assert 1==1
def test_getdata():
        X,_, y,_ = load_data()
        assert len(X)>0
        assert len(X)==len(y)