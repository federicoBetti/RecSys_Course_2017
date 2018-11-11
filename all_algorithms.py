import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from KNN.item_knn_CF import ItemKNNCFRecommender
from KNN.user_knn_CF import UserKNNCFRecommender
from MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE
from data.Movielens10MReader import Movielens10MReader


def create_URM_matrix(train):
    URM = np.zeros((50446, 20635))
    tracks_per_play = train.groupby(train.playlist_id).track_id.apply(list)
    for i in range(len(URM)):
        try:
            URM[i][tracks_per_play[i]] = 1
        except KeyError:
            continue
    URM_sparse = csr_matrix(URM)
    del URM
    return URM_sparse


def divide_train_test(train_old):
    msk = np.random.rand(len(train_old)) < 0.8
    train = train_old[msk]
    test = train_old[~msk]
    return train, test


def make_series(test):
    dict_train = test.groupby(test.playlist_id).track_id
    test_songs_per_playlist = pd.Series(dict_train.apply(list))
    return test_songs_per_playlist

def get_data():
    train = pd.read_csv("data\\train.csv")
    all_playlist_to_predict = pd.DataFrame(index=train.playlist_id.unique())
    s = train.playlist_id.unique()
    tracks = pd.read_csv("data\\tracks.csv")
    target_playlist = pd.read_csv("data\\target_playlists.csv")
    all_train = train.copy()
    train, test = divide_train_test(train)
    test_songs_per_playlist = make_series(test)
    all_playlist_to_predict['playlist_id'] = pd.Series(range(len(all_playlist_to_predict.index)),
                                                       index=all_playlist_to_predict.index)
    return all_train, train, test, tracks, target_playlist, all_playlist_to_predict, test_songs_per_playlist

if __name__ == '__main__':

    all_train, train, test, tracks, target_playlist, all_playlist_to_predict, test_songs_per_playlist = get_data()
    #dataReader = Movielens10MReader()

    URM_train = create_URM_matrix(train)
    #URM_validation = dataReader.get_URM_validation()
    URM_test = create_URM_matrix(test)

    recommender_list = []
    recommender_list.append(ItemKNNCFRecommender(URM_train))
    recommender_list.append(UserKNNCFRecommender(URM_train))
    recommender_list.append(MF_BPR_Cython(URM_train))
    #recommender_list.append(FunkSVD(URM_train))
    recommender_list.append(SLIM_BPR_Cython(URM_train, sparse_weights=False))
    #recommender_list.append(SLIM_R  MSE(URM_train))



    for recommender in recommender_list:

        print("Algorithm: {}".format(recommender.__class__))

        recommender.fit()

        results_run = recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
        print("Algorithm: {}, results: {}".format(recommender.__class__, results_run))

