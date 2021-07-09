import heapq
import math

import numpy as np
import pandas as pd
from scipy import spatial
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds


def create_matrix_a():
    a = pd.read_csv("rate.csv")
    df_a = a.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    return df_a


def create_matrix_m():
    matrix = matrix_a.copy()
    for user_id, row in matrix_a.iterrows():
        if int(user_id) % 2 == 0:
            for movie_id in row.index:
                if movie_id % 2 != 0 and matrix_a[movie_id][user_id] != 0:
                    hidden_values.append((user_id, movie_id))
                    matrix.at[user_id, movie_id] = 0
    return matrix


def create_matrix_h():
    return matrix_a.subtract(matrix_m)


def get_user_average_score():
    df = matrix_m.copy()
    return df.replace(0, np.NaN).mean(axis=1, skipna=True).replace(np.NaN, 0)


def get_movie_average_score():
    df = matrix_m.copy()

    return df.replace(0, np.NaN).mean(axis=0, skipna=True).replace(np.NaN, 0)


def get_average_score():
    df = matrix_m.copy()
    return np.nanmean(df.replace(0, np.NaN))


def item_based_collaborative_filtering_a(neighbour, similarity_type, svd):
    sum_r_b = 0
    b_x_i = m + b_u[user] + b_m[movie]
    for nn in neighbour:
        r_x = matrix_m[nn[-1]][user]
        b_x = m + b_u[user] + b_m[nn[-1]]
        sum_r_b += r_x - b_x

    prediction = b_x_i + sum_r_b / len(neighbour)

    if prediction > 5.0:
        prediction = 5.0
    elif prediction < 0.0:
        prediction = 0.5
    ibcfa_predictions[(user, movie)] = prediction


def check_for_zero_values():
    for users in matrix_m.index:
        if b_u[users] != 0:
            b_u[users] -= m
    for movies in matrix_m.columns:
        if b_m[movies] != 0:
            b_m[movies] -= m


def it_pearson_similarity(movie, user, movies_rated):
    # item based
    p_sims = []
    heapq.heapify(p_sims)

    for x in movies_rated:
        p_sim = 0
        # print(seen)
        if it_b_normalized_matrix[movie].tolist().count(0) == len(it_b_normalized_matrix[movie].tolist()) and \
                it_b_normalized_matrix[x].tolist().count(0) == len(it_b_normalized_matrix[x].tolist()):
            p_sim = 1

        elif it_b_normalized_matrix[movie].tolist().count(0) == len(it_b_normalized_matrix[movie].tolist()) or \
                it_b_normalized_matrix[x].tolist().count(0) == len(it_b_normalized_matrix[x].tolist()):

            p_sim = 0

        else:
            p_sim = 1 - spatial.distance.cosine(it_b_normalized_matrix[movie].tolist(),
                                                it_b_normalized_matrix[x].tolist())
        if movie != x and matrix_m[x][user] != 0:
            heapq.heappush(p_sims, (p_sim, x))

    return heapq.nlargest(5, p_sims)


def get_svd_decomposition_matrices():
    mm = []
    for i in matrix_m.values:
        mm.append(list(i))
    A = csc_matrix(mm, dtype=float)
    U, s, Vt = svds(A, k=3)
    S = np.diag(s)

    return U, S, Vt


def users_pearson_sim(user, movie):
    p_cor = []
    heapq.heapify(p_cor)
    pearson = 0
    for users_that_rated_movie in matrix_m[movie].to_numpy().nonzero():
        for user_ in users_that_rated_movie:
            if normalized_matrix.loc[user].tolist().count(0) == len(normalized_matrix.loc[user].tolist()) and \
                    normalized_matrix.loc[user_ + 1].tolist().count(0) == len(
                normalized_matrix.loc[user_ + 1].tolist()):
                pearson = 1
            elif normalized_matrix.loc[user].tolist().count(0) == len(normalized_matrix.loc[user].tolist()) or \
                    normalized_matrix.loc[user_ + 1].tolist().count(0) == len(
                normalized_matrix.loc[user_ + 1].tolist()):
                pearson = 0
            else:
                pearson = 1 - spatial.distance.cosine(normalized_matrix.loc[user].tolist(),
                                                      normalized_matrix.loc[user_ + 1].tolist())
            heapq.heappush(p_cor, (pearson, user_ + 1))
    return heapq.nlargest(5, p_cor)


def get_knn_neighbours(movie, user):
    movies_rated_by_user = [i for i in matrix_m if matrix_m[i][user] != 0]
    it_pearson_neighbours[(movie, user)] = it_pearson_similarity(movie, user, movies_rated_by_user)
    pearson_knn[(movie_id, user_id)] = users_pearson_sim(user_id, movie_id)


def user_based_collaborative_filtering_a(neighbours, similarity_type, svd_type):
    b_x_i = m + b_u[user] + b_m[movie]
    sum_ = 0
    for sim, us in neighbours:
        real_rating = matrix_a[movie_id][us]
        b_y_i = m + b_u[us] + b_m[movie]
        sum_ += real_rating - b_y_i
    prediction = b_x_i + (sum_ / len(pearson_knn[(movie, user)]))
    if prediction > 5.0:
        prediction = 5
    elif prediction < 0.0:
        prediction = 0.5
    ubcfa_predictions[(user, movie)] = prediction


def user_based_collaborative_filtering_b(neighbour, similarity_type, svd):
    similarity_sum = 0
    r_minus_b_sum = 0
    for sim, us in neighbour:
        similarity_sum += 1 + sim
        r_x_j = matrix_m[movie][us]
        b_x = m + b_u[us] + b_m[movie]
        r_minus_b_sum += (1 + sim) * (r_x_j - b_x)

    b_x_i = m + b_u[user] + b_m[movie]

    prediction = b_x_i + r_minus_b_sum / similarity_sum

    if prediction > 5.0:
        prediction = 5.0
    elif prediction < 0.0:
        prediction = 0.5

    ubcfb_predictions[(user, movie)] = prediction


def item_based_collaborative_filtering_b(neighbour, similarity_type, svd):
    similarity_sum = 0
    r_b_sum = 0
    for i in neighbour:
        similarity_sum += 1 + i[0]
        r_x = matrix_m[i[-1]][user]
        b_x = m + b_u[user] + b_m[i[-1]]
        r_b_sum += (1 + i[0]) * (r_x - b_x)

    b_x_i = m + b_u[user] + b_m[movie]

    prediction = b_x_i + r_b_sum / similarity_sum

    if prediction > 5.0:
        prediction = 5.0
    elif prediction < 0.0:
        prediction = 0.5

    ibcfb_predictions[(user, movie)] = prediction


def calculate_pre(methods_predictions):
    _heap = []
    top_values = {i: [] for i in methods_predictions.keys()}
    heapq.heapify(_heap)
    for i in methods_predictions:
        for j in methods_predictions[i]:
            heapq.heappush(_heap, j)
        top_values[i].append(heapq.nlargest(3, _heap))
        _heap.clear()
    sum_ = 0
    for user, movies_with_prediction in top_values.items():

        for rating, movie in movies_with_prediction[0]:
            real_value = matrix_m[movie][user]
            predicted_value = rating
            sum_ += pow(real_value - predicted_value, 2)
    return math.sqrt(sum_)


def get_hybrid_a_prediction():
    file.write(f"Uniform Hybrid Similarity Criterion: Pearson\n")
    for pair in hidden_values:
        hyb = 0.8 * ibcfa_predictions[pair] + 0.2 * ubcfa_predictions[pair]
        real = matrix_a[pair[-1]][pair[0]]
        root_mean_sqrt_error = pow((hyb - real), 2)
        rmse_hybrid_a.append(root_mean_sqrt_error)
        file.write(f"User: {pair[0]} Movie: {pair[-1]}\n")
        file.write(f"--Hybrid prediction: {hyb}\n")
        file.write(f"--Real value: {real}\n")


def get_hybrid_b_prediction():
    file.write(f"\n\nUniform Hybrid Similarity Criterion: Pearson\n")
    for pair in hidden_values:
        hyb = 0.7 * ibcfb_predictions[pair] + 0.3 * ubcfb_predictions[pair]
        real = matrix_a[pair[-1]][pair[0]]
        root_mean_sqrt_error = pow((hyb - real), 2)
        rmse_hybrid_b.append(root_mean_sqrt_error)
        file.write(f"User: {pair[0]} Movie: {pair[-1]}\n")
        file.write(f"--Hybrid prediction: {hyb}\n")
        file.write(f"--Real value: {real}\n")


if __name__ == '__main__':
    file = open("hybrid_output.txt", "w")
    file.write("~HYBRID RECOMMENDER SYSTEM~\n\n")
    file.write(f"Dataset: rate.csv\n")
    hidden_values = []

    matrix_a = create_matrix_a()
    matrix_m = create_matrix_m()
    matrix_h = create_matrix_h()

    seen = {i: [] for i in matrix_m.index}

    m = get_average_score()
    b_u = get_user_average_score()
    b_m = get_movie_average_score()
    check_for_zero_values()

    u, s, vt = get_svd_decomposition_matrices()

    it_b_normalized_matrix = matrix_m.copy()

    for i in matrix_m:
        it_b_normalized_matrix[i] = matrix_m[i].replace(0, np.nan) - b_m[i]
    it_b_normalized_matrix = it_b_normalized_matrix.replace(np.nan, 0).copy()
    normalized_matrix = matrix_m.copy()
    normalized_matrix = normalized_matrix.sub(b_u, axis='rows')

    pearson_knn = dict()
    it_pearson_neighbours = dict()
    for user_id, movie_id in hidden_values:
        get_knn_neighbours(movie_id, user_id)

    ibcfa_predictions = dict()
    ubcfa_predictions = dict()

    ibcfb_predictions = dict()
    ubcfb_predictions = dict()

    rmse_hybrid_a = []
    rmse_hybrid_b = []

    for movie, user in it_pearson_neighbours.keys():
        if it_pearson_neighbours.get((movie, user)):
            item_based_collaborative_filtering_a(it_pearson_neighbours.get((movie, user)), "PSim", None)
            item_based_collaborative_filtering_b(it_pearson_neighbours.get((movie, user)), "PSim", None)
        else:
            ibcfa_predictions[(user, movie)] = m + b_u[user] + b_m[movie]
            ibcfb_predictions[(user, movie)] = m + b_u[user] + b_m[movie]


    for movie, user in pearson_knn.keys():
        if pearson_knn.get((movie, user)):
            user_based_collaborative_filtering_a(pearson_knn.get((movie, user)), "PSim", None)
            user_based_collaborative_filtering_b(pearson_knn.get((movie, user)), "PSim", None)
        else:
            ubcfa_predictions[(user, movie)] = m + b_u[user] + b_m[movie]
            ubcfb_predictions[(user, movie)] = m + b_u[user] + b_m[movie]

    get_hybrid_a_prediction()
    print("---------")
    get_hybrid_b_prediction()

    print(f"RMSE hybrid a: {math.sqrt(sum(rmse_hybrid_a))}")
    print(f"RMSE hybrid b: {math.sqrt(sum(rmse_hybrid_b))}")
