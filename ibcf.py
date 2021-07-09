import heapq
import math

import pandas as pd
import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from scipy import spatial


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


def change_df_values(matrix_df, my_array):
    for i, j in zip(matrix_df.index, my_array):
        for movie, rating in zip(matrix_df.columns, j):
            matrix_df.at[i, movie] = rating


def get_svd_decomposition_matrices():
    mm = []
    for i in matrix_m.values:
        mm.append(list(i))
    A = csc_matrix(mm, dtype=float)
    U, s, Vt = svds(A, k=3)
    S = np.diag(s)

    # print("Approximate SVD decomposition for SPARSE matrix M")
    # print(f"U = ")
    # print(U)
    # print(f"S= ")
    # print(S)
    # print(f'Vt= ')
    # print(Vt)
    # print("--------")
    return U, S, Vt


def svdb():
    print("Users projection using svdb: ")
    print(np.matmul(u, s))
    print("Movies Projection Using svdb")
    print(np.matmul(vt.transpose(), np.sqrt(s)))


def svda():
    print("Users projection using svda: ")
    print(u)
    print("Movies Projection Using svda")
    print(np.matmul(vt.transpose(), np.sqrt(s)))


def it_cosine_similarity(movie, user, svd):
    similarities = []
    heapq.heapify(similarities)

    index = matrix_m.columns.get_loc(movie)
    if svd == "b":
        movies = np.matmul(vt.transpose(), np.sqrt(s))
    else:
        movies = np.matmul(vt.transpose(), s)

    for en, i in enumerate(movies):
        current_movie = matrix_m.columns[en]
        if np.size(i) - np.count_nonzero(i) == np.size(i) and \
                np.size(movies[index]) - np.count_nonzero(movies[index]) == np.size(movies[index]):
            cosine_distance = 1
        elif np.size(i) - np.count_nonzero(i) == np.size(i) or \
                np.size(movies[index]) - np.count_nonzero(movies[index]) == np.size(movies[index]):
            cosine_distance = 0
        else:
            cosine_distance = 1 - spatial.distance.cosine(movies[index], i)
        if movie != current_movie and matrix_m[current_movie][user] != 0:
            heapq.heappush(similarities, (cosine_distance, current_movie))

        # print(f"CoSim({movie},{current_movie}) = {cosine_distance}")
        # print(1 - spatial.distance.cosine(movies[index], i))

    return heapq.nlargest(5, similarities)


def it_pearson_similarity(movie, user, movies_rated):
    # item based
    p_sims = []
    heapq.heapify(p_sims)

    for x in movies_rated:
        p_sim = 0
        # print(seen)
        if normalized_matrix[movie].tolist().count(0) == len(normalized_matrix[movie].tolist()) and \
                normalized_matrix[x].tolist().count(0) == len(normalized_matrix[x].tolist()):
            p_sim = 1

        elif normalized_matrix[movie].tolist().count(0) == len(normalized_matrix[movie].tolist()) or \
                normalized_matrix[x].tolist().count(0) == len(normalized_matrix[x].tolist()):

            p_sim = 0

        else:
            p_sim = 1 - spatial.distance.cosine(normalized_matrix[movie].tolist(), normalized_matrix[x].tolist())
        if movie != x and matrix_m[x][user] != 0:
            heapq.heappush(p_sims, (p_sim, x))

    return heapq.nlargest(5, p_sims)


def get_kk_neighbours(movie, user):
    movies_rated_by_user = [i for i in matrix_m if matrix_m[i][user] != 0]
    it_pearson_neighbours[(movie, user)] = it_pearson_similarity(movie, user, movies_rated_by_user)
    it_cosine_neighbours_svdb[(movie, user)] = it_cosine_similarity(movie, user, "b")
    it_cosine_neighbours_svda[(movie, user)] = it_cosine_similarity(movie, user, "a")


def check_for_zero_values():
    for users in matrix_m.index:
        if b_u[users] != 0:
            b_u[users] -= m
    for movies in matrix_m.columns:
        if b_m[movies] != 0:
            b_m[movies] -= m


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

    real_value = matrix_a[movie][user]
    root_mean_square_error = pow((prediction - real_value), 2)

    file.write(f"Movie:{movie}, User: {user}\n")
    file.write(f"--Prediction: {prediction}\n")
    file.write(f"--Actual Rating: {real_value}\n")

    if similarity_type == "PSim":
        pre_ibcfa_psim[user].append((prediction, movie))
        rmse_ibcfa_psim.append(root_mean_square_error)
    else:
        if svd == "b":
            pre_ibcfa_cosim_svdb[user].append((prediction, movie))
            rmse_ibcfa_cosim_svdb.append(root_mean_square_error)
        else:
            pre_ibcfa_cosim_svda[user].append((prediction, movie))
            rmse_ibcfa_cosim_svda.append(root_mean_square_error)

    # print(f"Movie: {movie} User: {user}")
    # print(f"Prediction: {b_x_i + sum_r_b / len(neighbour)}")
    # print(f"real value {matrix_a[movie][user]}")


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

    real_value = matrix_a[movie][user]
    root_mean_square_error = pow((prediction - real_value), 2)

    file.write(f"Movie:{movie}, User: {user}\n")
    file.write(f"--Prediction: {prediction}\n")
    file.write(f"--Actual Rating: {real_value}\n")

    if similarity_type == "PSim":
        pre_ibcfb_psim[user].append((prediction, movie))
        rmse_ibcfb_psim.append(root_mean_square_error)
    else:
        if svd == "b":
            pre_ibcfb_cosim_svdb[user].append((prediction, movie))
            rmse_ibcfb_cosim_svdb.append(root_mean_square_error)
        else:
            pre_ibcfb_cosim_svda[user].append((prediction, movie))
            rmse_ibcfb_cosim_svda.append(root_mean_square_error)

    # print(f"Movie: {movie} User: {user}")
    # print(f"prediction {b_x_i + r_b_sum / similarity_sum}")
    # print(f"real value {matrix_a[movie][user]}")


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


if __name__ == '__main__':
    file = open("ibcf_output.txt", "w")
    file.write(f"~ItemBased RecommenderSystem~\n\n")
    hidden_values = []
    mat_time = time.time()
    matrix_a = create_matrix_a()
    matrix_m = create_matrix_m()
    matrix_h = create_matrix_h()
    mat_end = time.time()
    seen = {i: [] for i in matrix_m.index}
    statistics_time = time.time()
    m = get_average_score()
    b_u = get_user_average_score()
    b_m = get_movie_average_score()
    check_for_zero_values()

    statistics_end = time.time()

    svd_time = time.time()
    u, s, vt = get_svd_decomposition_matrices()
    svd_end = time.time()

    normalized_matrix = matrix_m.copy()

    for i in matrix_m:
        normalized_matrix[i] = matrix_m[i].replace(0, np.nan) - b_m[i]
    normalized_matrix = normalized_matrix.replace(np.nan, 0).copy()

    it_pearson_neighbours = dict()
    it_cosine_neighbours_svdb = dict()
    it_cosine_neighbours_svda = dict()
    nn_time = time.time()

    for user_id, movie_id in hidden_values:
        get_kk_neighbours(movie_id, user_id)
    nn_end = time.time()

    rmse_ibcfa_psim = []
    pre_ibcfa_psim = {i: [] for i in matrix_m.index if i % 2 == 0}

    rmse_ibcfa_cosim_svdb = []
    pre_ibcfa_cosim_svdb = {i: [] for i in matrix_m.index if i % 2 == 0}

    rmse_ibcfb_psim = []
    pre_ibcfb_psim = {i: [] for i in matrix_m.index if i % 2 == 0}

    rmse_ibcfb_cosim_svdb = []
    pre_ibcfb_cosim_svdb = {i: [] for i in matrix_m.index if i % 2 == 0}

    rmse_ibcfa_cosim_svda = []
    pre_ibcfa_cosim_svda = {i: [] for i in matrix_m.index if i % 2 == 0}

    rmse_ibcfb_cosim_svda = []
    pre_ibcfb_cosim_svda = {i: [] for i in matrix_m.index if i % 2 == 0}

    file.write(f"IBCFa Similarity Criterion: Pearson\n")
    for movie, user in it_pearson_neighbours.keys():
        if it_pearson_neighbours.get((movie, user)):
            item_based_collaborative_filtering_a(it_pearson_neighbours.get((movie, user)), "PSim", None)
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            pre_ibcfa_psim[user].append((prediction, movie))
            rmse_ibcfa_psim.append(root_mean_square_error)

    file.write(f"IBCFb Similarity Criterion: Pearson\n")
    for movie, user in it_pearson_neighbours.keys():
        if it_pearson_neighbours.get((movie, user)):
            item_based_collaborative_filtering_b(it_pearson_neighbours.get((movie, user)), "PSim", None)
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            pre_ibcfb_psim[user].append((prediction, movie))
            rmse_ibcfb_psim.append(root_mean_square_error)

    file.write(f"\nIBCFa Similarity Criterion: Cosine SVD type: b\n")
    for movie, user in it_cosine_neighbours_svdb.keys():
        if it_cosine_neighbours_svdb.get((movie, user)):
            item_based_collaborative_filtering_a(it_cosine_neighbours_svdb.get((movie, user)), "CoSim", "b")
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            pre_ibcfa_cosim_svdb[user].append((prediction, movie))
            rmse_ibcfa_cosim_svdb.append(root_mean_square_error)

    file.write(f"\nIBCFb Similarity Criterion: Cosine SVD type: b\n")
    for movie, user in it_cosine_neighbours_svdb.keys():
        if it_cosine_neighbours_svdb.get((movie, user)):
            item_based_collaborative_filtering_b(it_cosine_neighbours_svdb.get((movie, user)), "CoSim", "b")
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            pre_ibcfb_cosim_svdb[user].append((prediction, movie))
            rmse_ibcfb_cosim_svdb.append(root_mean_square_error)

    file.write(f"\nIBCFa Similarity Criterion: Cosine SVD type: a\n")
    for movie, user in it_cosine_neighbours_svdb.keys():
        if it_cosine_neighbours_svda.get((movie, user)):
            item_based_collaborative_filtering_a(it_cosine_neighbours_svda.get((movie, user)), "CoSim", "a")
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            pre_ibcfa_cosim_svda[user].append((prediction, movie))
            rmse_ibcfa_cosim_svda.append(root_mean_square_error)

    file.write(f"\nIBCFb Similarity Criterion: Cosine SVD type: a\n")
    for movie, user in it_cosine_neighbours_svdb.keys():
        if it_cosine_neighbours_svda.get((movie, user)):
            item_based_collaborative_filtering_b(it_cosine_neighbours_svda.get((movie, user)), "CoSim", "a")
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            pre_ibcfb_cosim_svda[user].append((prediction, movie))
            rmse_ibcfb_cosim_svda.append(root_mean_square_error)

    print(f"nn time: {nn_end - nn_time}")
    print(f"RMSE IBCFa PSim: {math.sqrt(sum(rmse_ibcfa_psim))}")
    print(f"PRE IBCFa Psim: {calculate_pre(pre_ibcfa_psim)}")

    print(f"RMSE IBCFa CoSim SVDb: {math.sqrt(sum(rmse_ibcfa_cosim_svdb))}")
    print(f"PRE IBCFa CoSim SVDb: {calculate_pre(pre_ibcfa_cosim_svdb)}")

    print(f"RMSE IBCFb PSim: {math.sqrt(sum(rmse_ibcfb_psim))}")
    print(f"PRE IBCFb Psim: {calculate_pre(pre_ibcfb_psim)}")

    print(f"RMSE IBCFb CoSim SVDb: {math.sqrt(sum(rmse_ibcfb_cosim_svdb))}")
    print(f"PRE IBCFb CoSim SVDb: {calculate_pre(pre_ibcfb_cosim_svdb)}")

    print(f"RMSE IBCFa CoSim SVDa: {math.sqrt(sum(rmse_ibcfa_cosim_svda))}")
    print(f"PRE IBCFa CoSim SVDa: {calculate_pre(pre_ibcfa_cosim_svda)}")

    print(f"RMSE IBCFb CoSim SVDa: {math.sqrt(sum(rmse_ibcfb_cosim_svda))}")
    print(f"PRE IBCFb CoSim SVDa: {calculate_pre(pre_ibcfb_cosim_svda)}")
