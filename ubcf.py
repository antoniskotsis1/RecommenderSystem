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


def cosine_similarity(movie, user, svd):
    similarities = []
    heapq.heapify(similarities)

    index = user - 1
    if svd == "b":
        users = np.matmul(u, np.sqrt(s))
    else:
        users = u

    for en, i in enumerate(users):
        current_user = en + 1
        if np.size(i) - np.count_nonzero(i) == np.size(i) and \
                np.size(users[index]) - np.count_nonzero(users[index]) == np.size(users[index]):
            cosine_distance = 1
        elif np.size(i) - np.count_nonzero(i) == np.size(i) or \
                np.size(users[index]) - np.count_nonzero(users[index]) == np.size(users[index]):
            cosine_distance = 0
        else:
            cosine_distance = 1 - spatial.distance.cosine(users[index], i)
        if user != current_user and matrix_m[movie][current_user] != 0:
            heapq.heappush(similarities, (cosine_distance, current_user))

    return heapq.nlargest(5, similarities)


def p_sim(user, movie):
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


def get_user_average_score():
    df = matrix_m.copy()
    return df.replace(0, np.NaN).mean(axis=1, skipna=True).replace(np.NaN, 0)


def get_movie_average_score():
    df = matrix_m.copy()
    return df.replace(0, np.NaN).mean(axis=0, skipna=True).replace(np.NaN, 0)


def check_for_zero_values():
    for users in matrix_m.index:
        if b_u[users] != 0:
            b_u[users] -= m


def get_average_score():
    df = matrix_m.copy()
    return np.nanmean(df.replace(0, np.NaN))


def get_knn_neighbours(movie_id, user_id):
    pearson_knn[(movie_id, user_id)] = p_sim(user_id, movie_id)
    cosine_knn_svda[(movie_id, user_id)] = cosine_similarity(movie_id, user_id, "a")
    cosine_knn_svdb[(movie_id, user_id)] = cosine_similarity(movie_id, user_id, "b")


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

    real_value = matrix_a[movie][user]
    root_mean_square_error = pow((prediction - real_value), 2)

    file.write(f"Movie:{movie}, User: {user}\n")
    file.write(f"--Prediction: {prediction}\n")
    file.write(f"--Actual Rating: {real_value}\n")

    if similarity_type == "PSim":
        pre_ubcfa_psim[user].append((prediction, movie))
        rmse_ubcfa_psim.append(root_mean_square_error)
    else:
        if svd_type == "b":
            pre_ubcfa_cosim_svdb[user].append((prediction, movie))
            rmse_ubcfa_cosim_svdb.append(root_mean_square_error)
        else:
            pre_ubcfa_cosim_svda[user].append((prediction, movie))
            rmse_ubcfa_cosim_svda.append(root_mean_square_error)


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

    real_value = matrix_a[movie][user]
    root_mean_square_error = pow((prediction - real_value), 2)

    file.write(f"Movie:{movie}, User: {user}\n")
    file.write(f"--Prediction: {prediction}\n")
    file.write(f"--Actual Rating: {real_value}\n")

    if similarity_type == "PSim":
        pre_ubcfb_psim[user].append((prediction, movie))
        rmse_ubcfb_psim.append(root_mean_square_error)
    else:
        if svd == "b":
            pre_ubcfb_cosim_svdb[user].append((prediction, movie))
            rmse_ubcfb_cosim_svdb.append(root_mean_square_error)
        else:
            pre_ubcfb_cosim_svda[user].append((prediction, movie))
            rmse_ubcfb_cosim_svda.append(root_mean_square_error)


def get_svd_decomposition_matrices():
    mm = []
    for i in matrix_m.values:
        mm.append(list(i))
    A = csc_matrix(mm, dtype=float)
    U, s, Vt = svds(A, k=3)
    S = np.diag(s)

    return U, S, Vt


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
    file = open("ubcf_output.txt", "w")
    file.write(f"~UserBasedCF RECOMMENDER SYSTEM~\n\n")
    hidden_values = []
    matrix_a = create_matrix_a()
    matrix_m = create_matrix_m()
    normalized_matrix = matrix_m.copy()
    m = get_average_score()
    b_u = get_user_average_score()
    b_m = get_movie_average_score()
    check_for_zero_values()
    u, s, vt = get_svd_decomposition_matrices()
    normalized_matrix = normalized_matrix.sub(b_u, axis='rows')
    pearson_knn = dict()
    cosine_knn_svda = dict()
    cosine_knn_svdb = dict()
    for user_id, movie_id in hidden_values:
        get_knn_neighbours(movie_id, user_id)

    rmse_ubcfa_psim = []
    pre_ubcfa_psim = {i: [] for i in matrix_m.index if i % 2 == 0}

    pre_ubcfb_psim = {i: [] for i in matrix_m.index if i % 2 == 0}
    rmse_ubcfb_psim = []

    rmse_ubcfa_cosim_svda = []
    rmse_ubcfa_cosim_svdb = []

    pre_ubcfa_cosim_svda = {i: [] for i in matrix_m.index if i % 2 == 0}
    pre_ubcfa_cosim_svdb = {i: [] for i in matrix_m.index if i % 2 == 0}

    rmse_ubcfb_cosim_svda = []
    rmse_ubcfb_cosim_svdb = []

    pre_ubcfb_cosim_svda = {i: [] for i in matrix_m.index if i % 2 == 0}
    pre_ubcfb_cosim_svdb = {i: [] for i in matrix_m.index if i % 2 == 0}
    file.write(f"UBCFa SIMILARITY: Pearson\n")
    for movie, user in pearson_knn.keys():
        if pearson_knn.get((movie, user)):
            user_based_collaborative_filtering_a(pearson_knn.get((movie, user)), "PSim", None)
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            rmse_ubcfa_psim.append(root_mean_square_error)
            pre_ubcfa_psim[user].append((prediction, movie))

    file.write(f"\nUBCFb SIMILARITY: Pearson\n")
    for movie, user in pearson_knn.keys():
        if pearson_knn.get((movie, user)):
            user_based_collaborative_filtering_b(pearson_knn.get((movie, user)), "PSim", None)
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            rmse_ubcfb_psim.append(root_mean_square_error)
            pre_ubcfb_psim[user].append((prediction, movie))

    file.write(f"\nUBCFa SIMILARITY: Cosine SVD type: a\n")
    for movie, user in cosine_knn_svda.keys():
        if cosine_knn_svda.get((movie, user)):
            user_based_collaborative_filtering_a(cosine_knn_svda.get((movie, user)), "CoSim", "a")
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            rmse_ubcfa_cosim_svda.append(root_mean_square_error)
            pre_ubcfa_cosim_svda[user].append((prediction, movie))

    file.write(f"\nUBCFb SIMILARITY: Cosine SVD type: a\n")
    for movie, user in cosine_knn_svda.keys():
        if cosine_knn_svda.get((movie, user)):
            user_based_collaborative_filtering_b(cosine_knn_svda.get((movie, user)), "Cosim", "a")
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            rmse_ubcfb_cosim_svda.append(root_mean_square_error)
            pre_ubcfb_cosim_svda[user].append((prediction, movie))

    file.write(f"\nUBCFb SIMILARITY: Cosine SVD type: b\n")
    for movie, user in cosine_knn_svdb.keys():
        if cosine_knn_svdb.get((movie, user)):
            user_based_collaborative_filtering_b(cosine_knn_svdb.get((movie, user)), "Cosim", "b")
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            rmse_ubcfb_cosim_svdb.append(root_mean_square_error)
            pre_ubcfb_cosim_svdb[user].append((prediction, movie))

    file.write(f"\nUBCFa SIMILARITY: Cosine SVD type: b\n")
    for movie, user in cosine_knn_svdb.keys():
        if cosine_knn_svdb.get((movie, user)):
            user_based_collaborative_filtering_a(cosine_knn_svdb.get((movie, user)), "CoSim", "b")
        else:
            prediction = m + b_u[user] + b_m[movie]
            real_value = matrix_a[movie][user]
            root_mean_square_error = pow((real_value - prediction), 2)
            rmse_ubcfa_cosim_svdb.append(root_mean_square_error)
            pre_ubcfa_cosim_svdb[user].append((prediction, movie))

    print(f"RMSE ubcfa psim: {math.sqrt(sum(rmse_ubcfa_psim))}")
    print(f"RMSE ubcfb psim: {math.sqrt(sum(rmse_ubcfb_psim))}")

    print(f"PRE ubcfa psim: {calculate_pre(pre_ubcfa_psim)}")
    print(f"PRE ubcfb psim: {calculate_pre(pre_ubcfb_psim)}")

    print(f"RMSE ubcfa CoSim svda: {math.sqrt(sum(rmse_ubcfa_cosim_svda))}")
    print(f"RMSE ubcfb CoSim svda: {math.sqrt(sum(rmse_ubcfb_cosim_svda))}")

    print(f"RMSE ubcfa CoSim svdb: {math.sqrt(sum(rmse_ubcfa_cosim_svdb))}")
    print(f"RMSE ubcfb CoSim svdb: {math.sqrt(sum(rmse_ubcfb_cosim_svdb))}")

    print(f"PRE ubcfa CoSim SVDa: {calculate_pre(pre_ubcfa_cosim_svda)}")
    print(f"PRE ubcfb CoSim SVDa: {calculate_pre(pre_ubcfb_cosim_svda)}")

    print(f"PRE ubcfa CoSim SVDb: {calculate_pre(pre_ubcfa_cosim_svdb)}")
    print(f"PRE ubcfb CoSim SVDb: {calculate_pre(pre_ubcfb_cosim_svdb)}")
