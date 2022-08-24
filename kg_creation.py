import numpy as np

from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.latent_features import ComplEx
import tensorflow as tf
from ampligraph.evaluation import evaluate_performance
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from ampligraph.utils import create_tensorboard_visualizations, restore_model
from ampligraph.discovery import find_clusters, query_topn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text

'''
ps1 = np.loadtxt('data/PS1.txt', dtype=float)
ps1 = np.mean(ps1, axis=1)
ps2 = np.loadtxt('data/PS2.txt', dtype=float)
ps2 = np.mean(ps2, axis=1)
ps3 = np.loadtxt('data/PS3.txt', dtype=float)
ps3 = np.mean(ps3, axis=1)
ps4 = np.loadtxt('data/PS4.txt', dtype=float)
ps4 = np.mean(ps4, axis=1)
ps5 = np.loadtxt('data/PS5.txt', dtype=float)
ps5 = np.mean(ps5, axis=1)
ps6 = np.loadtxt('data/PS6.txt', dtype=float)
ps6 = np.mean(ps6, axis=1)
eps1 = np.loadtxt('data/EPS1.txt', dtype=float)
eps1 = np.mean(eps1, axis=1)
fs1 = np.loadtxt('data/FS1.txt', dtype=float)
fs1 = np.mean(fs1, axis=1)
fs2 = np.loadtxt('data/FS2.txt', dtype=float)
fs2 = np.mean(fs2, axis=1)
ts1 = np.loadtxt('data/TS1.txt', dtype=float)
ts1 = np.mean(ts1, axis=1)
ts2 = np.loadtxt('data/TS2.txt', dtype=float)
ts2 = np.mean(ts2, axis=1)
ts3 = np.loadtxt('data/TS3.txt', dtype=float)
ts3 = np.mean(ts3, axis=1)
ts4 = np.loadtxt('data/TS4.txt', dtype=float)
ts4 = np.mean(ts4, axis=1)
vs1 = np.loadtxt('data/VS1.txt', dtype=float)
vs1 = np.mean(vs1, axis=1)
ce = np.loadtxt('data/CE.txt', dtype=float)
ce = np.mean(ce, axis=1)
cp = np.loadtxt('data/CP.txt', dtype=float)
cp = np.mean(cp, axis=1)
se = np.loadtxt('data/SE.txt', dtype=float)
se = np.mean(se, axis=1)
cooler = np.loadtxt('data/profile.txt', usecols=0, dtype=int)
valve = np.loadtxt('data/profile.txt', usecols=1, dtype=int)
pump = np.loadtxt('data/profile.txt', usecols=2, dtype=int)
accumulator = np.loadtxt('data/profile.txt', usecols=3, dtype=int)
state = np.loadtxt('data/profile.txt', usecols=4, dtype=int)
data = np.stack((ps1, ps2, ps3, ps4, ps5, ps6, eps1, fs1, fs2,
                 ts1, ts2, ts3, ts4, vs1, ce, cp, se, cooler,
                 valve, pump, accumulator, state), axis=1)
column = ['ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'eps1', 'fs1', 'fs2',
          'ts1', 'ts2', 'ts3', 'ts4', 'vs1', 'ce', 'cp', 'se', 'cooler',
          'valve', 'pump', 'accumulator', 'state']
df = pd.DataFrame(data=data,
                  columns=column)
df.to_pickle('dataset')
'''
df = pd.read_pickle('dataset')
# print(df[:10])
m = ''
triples = []
triples_sensors = []
triples_conditions = []
l = []
for _, row in df.iterrows():
    m = ''
    if 155.0 < row["ps1"] <= 159.0:
        m = m + '1'
    elif 159.0 < row["ps1"] <= 161.0:
        m = m + '2'
    else:
        m = m + '3'

    if 104.0 < row["ps2"] <= 108.0:
        m = m + '1'
    elif 108.0 < row["ps2"] <= 110.0:
        m = m + '2'
    else:
        m = m + '3'

    if 0.5 < row["ps3"] <= 1.5:
        m = m + '1'
    elif 1.5 < row["ps3"] <= 1.8:
        m = m + '2'
    else:
        m = m + '3'

    if 0.0 < row["ps4"] <= 2.3:
        m = m + '1'
    elif 2.3 < row["ps4"] <= 2.6:
        m = m + '2'
    elif 2.6 < row["ps4"] <= 2.9:
        m = m + '4'
    else:
        m = m + '5'

    if 8.0 < row["ps5"] <= 8.8:
        m = m + '1'
    elif 8.8 < row["ps5"] <= 9.3:
        m = m + '2'
    else:
        m = m + '3'

    if 8.0 < row["ps6"] <= 8.8:
        m = m + '1'
    elif 8.8 < row["ps6"] <= 9.3:
        m = m + '2'
    else:
        m = m + '3'

    if 2361.0 < row["eps1"] <= 2400.0:
        m = m + '1'
    elif 2400.0 < row["eps1"] <= 2600.0:
        m = m + '2'
    else:
        m = m + '3'

    if 2.0 < row["fs1"] <= 4.0:
        m = m + '1'
    elif 4.0 < row["fs1"] <= 5.0:
        m = m + '2'
    elif 5.0 < row["fs1"] <= 5.9:
        m = m + '3'
    elif 5.9 < row["fs1"] <= 6.0:
        m = m + '4'
    elif 6.0 < row["fs1"] <= 6.1:
        m = m + '5'
    elif 6.1 < row["fs1"] <= 6.2:
        m = m + '6'
    elif 6.2 < row["fs1"] <= 6.4:
        m = m + '7'
    else:
        m = m + '8'

    if 8.8 < row["fs2"] <= 9.3:
        m = m + '1'
    elif 9.3 < row["fs2"] <= 9.7:
        m = m + '2'
    else:
        m = m + '3'

    if 35.0 < row["ts1"] <= 44.0:
        m = m + '1'
    elif 44.0 < row["ts1"] <= 51.0:
        m = m + '2'
    else:
        m = m + '3'

    if 40.0 < row["ts2"] <= 47.0:
        m = m + '1'
    elif 47.0 < row["ts2"] <= 54.0:
        m = m + '2'
    else:
        m = m + '3'

    if 38.0 < row["ts3"] <= 46.0:
        m = m + '1'
    elif 46.0 < row["ts3"] <= 50.0:
        m = m + '2'
    else:
        m = m + '3'

    if 30.0 < row["ts4"] <= 37.0:
        m = m + '1'
    elif 37.0 < row["ts4"] <= 43.0:
        m = m + '2'
    elif 43.0 < row["ts4"] <= 49.0:
        m = m + '3'
    else:
        m = m + '4'

    if 0.5 < row["vs1"] <= 0.58:
        m = m + '1'
    elif 0.58 < row["vs1"] <= 0.63:
        m = m + '2'
    elif 0.63 < row["vs1"] <= 0.73:
        m = m + '3'
    else:
        m = m + '4'

    if 17.0 < row["ce"] <= 27.0:
        m = m + '1'
    elif 27.0 < row["ce"] <= 34.0:
        m = m + '2'
    elif 34.0 < row["ce"] <= 40.0:
        m = m + '3'
    else:
        m = m + '4'

    if 1.0 < row["cp"] <= 1.5:
        m = m + '1'
    elif 1.5 < row["cp"] <= 2.2:
        m = m + '2'
    else:
        m = m + '3'

    if 18.0 < row["se"] <= 40.0:
        m = m + '1'
    elif 40.0 < row["se"] <= 45.0:
        m = m + '2'
    elif 45.0 < row["se"] <= 50.0:
        m = m + '3'
    elif 50.0 < row["se"] <= 55.0:
        m = m + '4'
    else:
        m = m + '5'

    p1 = (str(row['ps1']), 'PS1valueFor', m)
    p2 = (str(row['ps2']), 'PS2valueFor', m)
    p3 = (str(row['ps3']), 'PS3valueFor', m)
    p4 = (str(row['ps4']), 'PS4valueFor', m)
    p5 = (str(row['ps5']), 'PS5valueFor', m)
    p6 = (str(row['ps6']), 'PS6valueFor', m)
    ep = (str(row['eps1']), 'EPS1valueFor', m)
    f1 = (str(row['fs1']), 'FS1valueFor', m)
    f2 = (str(row['fs2']), 'FS2valueFor', m)
    t1 = (str(row['ts1']), 'TS1valueFor', m)
    t2 = (str(row['ts2']), 'TS2valueFor', m)
    t3 = (str(row['ts3']), 'TS3valueFor', m)
    t4 = (str(row['ts4']), 'TS4valueFor', m)
    vs = (str(row['vs1']), 'VS1valueFor', m)
    ce = (str(row['ce']), 'CEvalueFor', m)
    cp = (str(row['cp']), 'CPvalueFor', m)
    se = (str(row['se']), 'SEvalueFor', m)

    if row['cooler'] == 3:
        cool = (m, 'efficiencyOfCooler', 'coolerCloseToFailure')
    elif row['cooler'] == 20:
        cool = (m, 'efficiencyOfCooler', 'reduced')
    elif row['cooler'] == 100:
        cool = (m, 'efficiencyOfCooler', 'full')

    if row['valve'] == 100:
        val = (m, 'conditionOfValve', 'optimal')
    elif row['valve'] == 90:
        val = (m, 'conditionOfValve', 'smallLag')
    elif row['valve'] == 80:
        val = (m, 'conditionOfValve', 'severeLag')
    elif row['valve'] == 73:
        val = (m, 'conditionOfValve', 'valveCloseToFailure')

    if row['pump'] == 0:
        pl = (m, 'leakageOfPump', 'noLeakage')
    elif row['pump'] == 1:
        pl = (m, 'leakageOfPump', 'weakLeakage')
    elif row['pump'] == 2:
        pl = (m, 'leakageOfPump', 'severeLeakage')

    if row['accumulator'] == 130:
        acc = (m, 'pressureByHydraulicPump', 'inOptimalRange')
    elif row['accumulator'] == 115:
        acc = (m, 'pressureByHydraulicPump', 'slightlyReduced')
    elif row['accumulator'] == 100:
        acc = (m, 'pressureByHydraulicPump', 'severelyReduced')
    elif row['accumulator'] == 90:
        acc = (m, 'pressureByHydraulicPump', 'hydraulicCloseToFailure')

    if row['state'] == 0.0:
        label = (m, 'stableCondition')
    elif row['state'] == 1.0:
        label = (m, 'staticConditionNotReached')

    l.append(list(label))
    triples.extend((p1, p2, p3, p4, p5, p6, ep, f1, f2, t1, t2, t3, t4, vs, ce, cp, se,
                    cool, val, pl, acc))
    triples_sensors.extend((p1, p2, p3, p4, p5, p6, ep, f1, f2, t1, t2, t3, t4, vs, ce, cp, se))
    triples_conditions.extend((cool, val, pl, acc))

# print(np.array(triples))
triples_df = pd.DataFrame(triples, columns=['subject', 'predicate', 'object'])
label_df = pd.DataFrame(l, columns=['id', 'state'])
label_df = label_df.drop_duplicates('id', keep='first')

# print(triples_df[(triples_df['object'] == 'hydraulicCloseToFailure')])
# print(triples_df[(triples_df.subject == "11251118133343115") | (triples_df.object == "11251118133343115")])

X_train, X_valid = train_test_split_no_unseen(np.array(triples), test_size=6305)
'''
model = ComplEx(batches_count=50,
                epochs=300,
                k=100,
                eta=20,
                optimizer='adam',
                optimizer_params={'lr': 1e-4},
                loss='multiclass_nll',
                regularizer='LP',
                regularizer_params={'p': 3, 'lambda': 1e-5},
                seed=0,
                verbose=True)
tf.logging.set_verbosity(tf.logging.ERROR)
model.fit(X_train)
filter_triples = np.concatenate((X_train, X_valid))
ranks = evaluate_performance(X_valid,
                             model=model,
                             filter_triples=filter_triples,
                             use_default_protocol=True,
                             verbose=True)
'''

triples_valid = pd.DataFrame(X_valid, columns=['subject', 'predicate', 'object'])
X_test = []
for _, row in triples_valid.iterrows():
    if row['predicate'] in ['efficiencyOfCooler', 'conditionOfValve', 'leakageOfPump', 'pressureByHydraulicPump']:
        i = row.tolist()
        X_test.append(i)
model = restore_model(model_name_path="sensor_complEx.pkl")
'''
filter_triples = np.concatenate((X_train, X_valid))
ranks = evaluate_performance(X_valid,
                             model=model,
                             filter_triples=filter_triples,
                             use_default_protocol=True,
                             verbose=True)
'''
filter_triples = np.concatenate((X_train, X_valid))
'''
ranks = evaluate_performance(np.array(X_test),
                             model=model,
                             filter_triples=filter_triples,
                             corrupt_side='o',
                             use_default_protocol=False,
                             verbose=True)

mr = mr_score(ranks)
mrr = mrr_score(ranks)
print("MRR: %.2f" % mrr)
print("MR: %.2f" % mr)
hits_10 = hits_at_n_score(ranks, n=10)
print("Hits@10: %.2f" % hits_10)
hits_3 = hits_at_n_score(ranks, n=3)
print("Hits@3: %.2f" % hits_3)
hits_1 = hits_at_n_score(ranks, n=1)
print("Hits@1: %.2f" % hits_1)
'''
'''
train, test = train_test_split(label_df, test_size=0.2)

m_id_train = train['id']
Xm_train = model.get_embeddings(m_id_train)
y_train = train['state'].values

m_id_test = test['id']
Xm_test = model.get_embeddings(m_id_test)
y_test = test['state'].values

# print(Xm_train.shape, Xm_test.shape, y_train.shape, y_test.shape)
# create_tensorboard_visualizations(model, 'Hydraulic_embeddings')
clf_model = XGBClassifier()
clf_model.fit(Xm_train, y_train)
p = clf_model.predict(Xm_test)
op = metrics.accuracy_score(y_test, p)
top = {
    "true": y_test,
    "predict": p
}
op_df = pd.DataFrame(top)
print("accuracy= ", op)
print(op_df)
'''
t = np.array(triples)
all_entities = np.array(list(set(t[:, 0]).union(t[:, 2])))
print(model.get_embeddings(['11251118133343115', 'hydraulicCloseToFailure']))
'''
t_embeddings = dict(zip(all_entities, model.get_embeddings(all_entities)))
embeddings_2d = PCA(n_components=2).fit_transform(np.array([i for i in t_embeddings.values()]))

kmeans = KMeans(n_clusters=3, n_init=100, max_iter=500)
clusters = find_clusters(all_entities, model, kmeans, mode='entity')
# Create a dataframe to plot the embeddings using scatterplot
df = pd.DataFrame({"entities": all_entities, "clusters": "cluster" + pd.Series(clusters).astype(str),
                   "embedding1": embeddings_2d[:, 0], "embedding2": embeddings_2d[:, 1]})

plt.figure(figsize=(20, 20))
plt.title("Cluster embeddings")

ax = sns.scatterplot(data=df, x="embedding1", y="embedding2", hue="clusters")
texts = []
for i, point in df.iterrows():
    # randomly choose a few labels to be printed
    if np.random.uniform() < 0.003:
        texts.append(plt.text(point['embedding1'] + .1, point['embedding2'], str(point['entities'])))

adjust_text(texts)
plt.savefig('industrial_data.png')
'''
