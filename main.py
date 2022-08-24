from ampligraph.datasets import load_fb15k
import numpy as np
from ampligraph.latent_features import TransE, ComplEx
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score, mr_score
from ampligraph.utils import save_model, restore_model, create_tensorboard_visualizations
from ampligraph.discovery import find_clusters, query_topn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text


def display_aggregate_metrics(ranks):
    print("Mean Rank:", mr_score(ranks))
    print("Mean Reciprocal Rank:", mrr_score(ranks))
    print("Hits@1:", hits_at_n_score(ranks, 1))
    print("Hits@3:", hits_at_n_score(ranks, 3))
    print("Hits@10:", hits_at_n_score(ranks, 10))


def display_embeddings_details(test_model):
    print("The number of unique entities:", len(test_model.ent_to_idx))
    print("The number of unique relations:", len(test_model.rel_to_idx))
    print("Size of entity embeddings:", test_model.ent_emb.shape)
    print("Size of relation embeddings:", test_model.rel_emb.shape)


def model():
    x = load_fb15k()
    # print(x['train'][:5])

    # trans_model = TransE(k=150, epochs=4000, eta=10,
    #                     loss='multiclass_nll', optimizer='adam', optimizer_params={'lr': 5e-5},
    #                     initializer='xavier', initializer_params={'uniform': False},
    #                     regularizer='LP', regularizer_params={'lambda': 0.0001, 'p': 3},
    #                     embedding_model_params={'norm': 1, 'normalize_ent_emb': False},
    #                     seed=0, batches_count=100, verbose=True)

    # Complex_model = ComplEx(k=200, epochs=4000, eta=20,
    #                         loss='self_adversarial', loss_params={'margin': 1},
    #                         regularizer='LP', regularizer_params={'lambda': 0.0001, 'p': 3},
    #                         optimizer='adam', optimizer_params={'lr': 0.0005},
    #                         seed=0, batches_count=100, verbose=True)

    # trans_model.fit(x['train'])
    # Complex_model.fit(x['train'])
    # save_model(trans_model, model_name_path="transE_best_params.pkl")
    # save_model(Complex_model, model_name_path="complEx_best_params.pkl")

    test_model_transe = restore_model(model_name_path="transE_best_params.pkl")
    test_model_complex = restore_model(model_name_path="complEx_best_params.pkl")

    filter_triplets = np.concatenate([x['train'], x['valid'], x['test']], 0)

    triples, scores = query_topn(test_model_transe, top_n=10,
                                 head=None,
                                 relation='/location/country/form_of_government',
                                 tail='/m/06cx9',
                                 ents_to_consider=None,
                                 rels_to_consider=None)
    print("________________________TransE________________________")
    for triple, score in zip(triples, scores):
        print('Score: {} \t {} '.format(score, triple))

    sc = test_model_transe.predict(['/m/027rn', '/location/country/form_of_government', '/m/06cx9'])
    t = ['/m/027rn', '/location/country/form_of_government', '/m/06cx9']
    print('Score: {} \t {} '.format(sc, t))
    '''
    triples, scores = query_topn(test_model_complex, top_n=10,
                                 head=None,
                                 relation='/location/country/form_of_government',
                                 tail='/m/06cx9',
                                 ents_to_consider=None,
                                 rels_to_consider=None)
    print("________________________ComplEx________________________")
    for triple, score in zip(triples, scores):
        print('Score: {} \t {} '.format(score, triple))

    sc = test_model_complex.predict(['/m/027rn', '/location/country/form_of_government', '/m/06cx9'])
    t = ['/m/027rn', '/location/country/form_of_government', '/m/06cx9']
    print('Score: {} \t {} '.format(sc, t))
    '''
    ranks_transE = evaluate_performance(x['test'], model=test_model_transe,
                                        filter_triples=filter_triplets,
                                        corrupt_side='s,o',
                                        ranking_strategy='worst')

    print("________________________TransE________________________")
    display_aggregate_metrics(ranks_transE)

    ranks_ComplEx = evaluate_performance(x['test'], model=test_model_complex,
                                         filter_triples=filter_triplets,
                                         corrupt_side='s,o',
                                         ranking_strategy='worst')

    print("________________________ComplEx________________________")
    display_aggregate_metrics(ranks_ComplEx)
    
    # create_tensorboard_visualizations(test_model, 'embeddings_transE')
    # create_tensorboard_visualizations(test_model, 'embeddings_complEx')
    all_entities = np.array(list(set(filter_triplets[:, 0]).union(filter_triplets[:, 2])))
    kmeans = KMeans(n_clusters=4, n_init=100, max_iter=500)
    clusters = find_clusters(all_entities, test_model, kmeans, mode='entity')
    # Get the embeddings (150 dims) for all the entities of interest
    jobs_embeddings = test_model.get_embeddings(all_entities, embedding_type='entity')

    # Perform PCA and reduce the dims to 2
    embeddings_2d = PCA(n_components=2).fit_transform(np.array([emb for emb in jobs_embeddings]))

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
    plt.savefig('complEx4_embeddings_fbk15.png')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model()
