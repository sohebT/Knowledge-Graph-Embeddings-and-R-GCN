from pykeen.models import RGCN
from pykeen.datasets import FB15k237
from pykeen.evaluation import RankBasedEvaluator
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
from pykeen.sampling import BasicNegativeSampler
from pykeen.models.predict import get_head_prediction_df
from pykeen.models.predict import get_tail_prediction_df
from pykeen.models.predict import get_relation_prediction_df
from pykeen.models.predict import predict_triples_df
import torch

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = FB15k237()
import pandas as pd

model = RGCN(triples_factory=dataset.training,
             interaction='DistMult')
model = model.to(device)
'''
optimizer = Adam(params=model.get_grad_params())
training_loop = SLCWATrainingLoop(model=model,
                                  optimizer=optimizer,
                                  negative_sampler=BasicNegativeSampler,
                                  triples_factory=dataset.training)

training_loop.train(num_epochs=50,
                    batch_size=100,
                    triples_factory=dataset.training,
                    sampler='schlichtkrull')
'''
# torch.save(model, 'RGCN_model.pkl')
model = torch.load('RGCN_model.pkl')
'''
evaluator = RankBasedEvaluator(filtered=True)
results = evaluator.evaluate(model,
                             mapped_triples=dataset.testing.mapped_triples,
                             additional_filter_triples=[
                                 dataset.training.mapped_triples,
                                 dataset.validation.mapped_triples
                             ])

res = results.to_df()

print("Mean Rank:", results.get_metric('both.realistic.amr'))
print("Mean Reciprocal Rank:", results.get_metric('both.realistic.amrr'))
print("Hits@1:", results.get_metric('both.realistic.hits@1'))
print("Hits@3:", results.get_metric('both.realistic.hits@3'))
print("Hits@10:", results.get_metric('both.realistic.adjustedhitsatk'))

print(res)

df = predict_triples_df(
    model=model,
    triples=("/m/027rn", "/location/country/form_of_government", "/m/06cx9"),
    triples_factory=dataset.training,
)
'''

relation = get_relation_prediction_df(model, '/m/027rn', '/m/06cx9', triples_factory=dataset.training)
tail = get_tail_prediction_df(model, '/m/027rn', '/location/country/form_of_government',
                              triples_factory=dataset.training)

head = get_head_prediction_df(model, '/location/country/form_of_government', '/m/06cx9',
                              triples_factory=dataset.training)
