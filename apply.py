import pandas as pd

def function(x):
    # function returns square of number
    return x**2

df=pd.DataFrame({"Number":range(10, 100)})
df['Square'] = df['Number'].apply(lambda x: function(x))
