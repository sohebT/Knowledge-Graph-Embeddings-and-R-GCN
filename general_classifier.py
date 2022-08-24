import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_pickle('dataset')
df['state'] = df['state'].replace({1.0: 'staticConditionsNotReached', 0.0: 'stableConditions'})

train, test = train_test_split(df, test_size=0.2)

y_state = train['state'].values
yt_state = test['state'].values

y_cooler = train['cooler'].values
yt_cooler = test['cooler'].values

y_pump = train['pump'].values
yt_pump = test['pump'].values

y_valve = train['valve'].values
yt_valve = test['valve'].values

y_accumulator = train['accumulator'].values
yt_accumulator = test['accumulator'].values

X_train = train.drop(columns=['cooler', 'pump', 'valve', 'accumulator', 'state'])
X_test = test.drop(columns=['cooler', 'pump', 'valve', 'accumulator', 'state'])

model = XGBClassifier()
model.fit(X_train, y_state)
p = model.predict(X_test)
op = metrics.accuracy_score(yt_state, p)
top = {
    "true": yt_state,
    "predict": p
}
op_df = pd.DataFrame(top)
print("accuracy= ", op)
print(op_df.head(10))
