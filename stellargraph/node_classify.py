import numpy as np
import pandas as pd

import stellargraph as sg
from stellargraph import StellarDiGraph
from stellargraph.mapper import RelationalFullBatchNodeGenerator
from stellargraph.layer import RGCN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt

dataset = pd.read_pickle('dataset')
# graph = pd.read_pickle('dataset_graph')
# graph = graph.rename(columns={'subject': 'source', 'object': 'target', 'predicate': 'relation'})
# col = list(graph)
# col[0], col[1], col[2] = col[0], col[2], col[1]
# graph = graph[col]
# sensor_values = []
graph = []
measurement = []
stats = []
m = 0
for _, row in dataset.iterrows():
    if row['cooler'] == 3:
        cool = (m, 'coolerCloseToFailure', 'efficiencyOfCooler')
        cr3 = row
    elif row['cooler'] == 20:
        cool = (m, 'reduced', 'efficiencyOfCooler')
        cr20 = row
    elif row['cooler'] == 100:
        cool = (m, 'full', 'efficiencyOfCooler')
        cr100 = row

    if row['valve'] == 100:
        val = (m, 'optimal', 'conditionOfValve')
        val100 = row
    elif row['valve'] == 90:
        val = (m, 'smallLag', 'conditionOfValve')
        val90 = row
    elif row['valve'] == 80:
        val = (m, 'severeLag', 'conditionOfValve')
        val80 = row
    elif row['valve'] == 73:
        val = (m, 'valveCloseToFailure', 'conditionOfValve')
        val73 = row

    if row['pump'] == 0:
        pl = (m, 'noLeakage', 'leakageOfPump')
        p0 = row
    elif row['pump'] == 1:
        pl = (m, 'weakLeakage', 'leakageOfPump')
        p1 = row
    elif row['pump'] == 2:
        pl = (m, 'severeLeakage', 'leakageOfPump')
        p2 = row

    if row['accumulator'] == 130:
        acc = (m, 'inOptimalRange', 'pressureByHydraulicPump')
        a130 = row
    elif row['accumulator'] == 115:
        acc = (m, 'slightlyReduced', 'pressureByHydraulicPump')
        a115 = row
    elif row['accumulator'] == 100:
        acc = (m, 'severelyReduced', 'pressureByHydraulicPump')
        a100 = row
    elif row['accumulator'] == 90:
        acc = (m, 'hydraulicCloseToFailure', 'pressureByHydraulicPump')
        a90 = row

    if row['state'] == 0.0:
        st = 'stableConditions'
    elif row['state'] == 1.0:
        st = 'staticConditionsNotReached'
    m = m + 1
    graph.extend((cool, val, pl, acc))
    measurement.append(st)
stats.extend((cr3, cr20, cr100, val100, val90, val80, val73, p0, p1, p2, a130, a115, a100, a90))
stats_pd = pd.DataFrame(stats, columns=['ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'eps1', 'fs1', 'fs2', 'ts1', 'ts2',
                                        'ts3', 'ts4', 'vs1', 'ce', 'cp', 'se', 'cooler', 'valve', 'pump', 'accumulator',
                                        'state'],
                        index=['coolerCloseToFailure', 'reduced', 'full', 'optimal', 'smallLag',
                               'severeLag', 'valveCloseToFailure', 'noLeakage', 'weakLeakage',
                               'severeLeakage', 'inOptimalRange', 'slightlyReduced',
                               'severelyReduced', 'hydraulicCloseToFailure']
                        )
graph_pd = pd.DataFrame(graph, columns=['source', 'target', 'relation'])
measurement_pd = pd.DataFrame(measurement, columns=['class'])
# print(graph_pd)
nodes = dataset.drop(columns=['state', 'accumulator', 'pump', 'valve', 'cooler'])
# print(nodes)
status_nodes = {
    "c": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "v": [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "p": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    "h": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
}
status_nodes_pd = pd.DataFrame(status_nodes, index=['coolerCloseToFailure', 'reduced', 'full', 'optimal', 'smallLag',
                                                    'severeLag', 'valveCloseToFailure', 'noLeakage', 'weakLeakage',
                                                    'severeLeakage', 'inOptimalRange', 'slightlyReduced',
                                                    'severelyReduced', 'hydraulicCloseToFailure'])
# print(status_nodes_pd)
status = {
    "class": ['cooler', 'cooler', 'cooler', 'valve', 'valve', 'valve', 'valve', 'pump', 'pump', 'pump', 'HydraulicPump',
              'HydraulicPump', 'HydraulicPump', 'HydraulicPump']
}

status_pd = pd.DataFrame(status, index=['coolerCloseToFailure', 'reduced', 'full', 'optimal', 'smallLag', 'severeLag',
                                        'valveCloseToFailure', 'noLeakage', 'weakLeakage', 'severeLeakage',
                                        'inOptimalRange', 'slightlyReduced', 'severelyReduced',
                                        'hydraulicCloseToFailure'])
status_nodes_pd = pd.concat([stats_pd, status_nodes_pd], axis=1)
t = pd.concat([nodes, status_nodes_pd])
t = t.fillna(0)
# print(t)
tm = measurement_pd['class']
ts = status_pd['class']
dt = pd.concat([tm, ts], axis=0)
# print(dt)
print(dt.value_counts().to_frame())

kg = StellarDiGraph(
    {'measurement_nodes': t},
    graph_pd,
    edge_type_column='relation'
)

n_train_subjects, n_test_subjects = model_selection.train_test_split(
    tm, train_size=0.8, test_size=None, stratify=tm
)

val_subjects, v_test_subjects = model_selection.train_test_split(
    n_test_subjects, train_size=0.8, test_size=None, stratify=n_test_subjects
)

s_train_subjects, s_test_subjects = model_selection.train_test_split(
    ts, train_size=10, test_size=None, stratify=ts
)

train_subjects = pd.concat([n_train_subjects, s_train_subjects])
test_subjects = pd.concat([v_test_subjects, s_test_subjects])

print(train_subjects.value_counts().to_frame())
print(test_subjects.value_counts().to_frame())
print(val_subjects.value_counts().to_frame())
print(kg.info())

target_encoding = preprocessing.LabelBinarizer()
train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.fit_transform(test_subjects)
val_targets = target_encoding.transform(val_subjects)

generator = RelationalFullBatchNodeGenerator(kg, sparse=True)

train_gen = generator.flow(train_subjects.index, targets=train_targets)
test_gen = generator.flow(test_subjects.index, targets=test_targets)
val_gen = generator.flow(val_subjects.index, targets=val_targets)

rgcn = RGCN(
    layer_sizes=[32, 32],
    activations=["relu", "relu"],
    generator=generator,
    bias=True,
    num_bases=20,
    dropout=0.5,
)

x_in, x_out = rgcn.in_out_tensors()
predictions = Dense(train_targets.shape[-1], activation="softmax")(x_out)
model = Model(inputs=x_in, outputs=predictions)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(lr=0.01),
    metrics=["acc"],
)
history = model.fit(train_gen, validation_data=val_gen, epochs=20)
sg.utils.plot_history(history)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

all_nodes = dt.index
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen)
node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
df = pd.DataFrame({"Predicted": node_predictions, "True": dt})
print(df.loc[df['True'] == 'stableConditions'])
