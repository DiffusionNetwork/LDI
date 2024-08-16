import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

def cal_log(x):
    if x <= 0:
        return 0
    result = np.log2(x)
    return result

def cal_fraction_info(x, y):
    len_x = len(x) * 1.0
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    x_y_dict = {}
    mutual_info = 0
    for xi in x_unique:
        for yi in y_unique:
            xi_index = np.where(x == xi)[0]; len_xi_index = len(xi_index) * 1.0
            yi_index = np.where(y == yi)[0]; len_yi_index = len(yi_index) * 1.0
            x_y_count = len(set(yi_index) & set(xi_index)) * 1.0
            x_y_dict['x_' + str(xi)] = len_xi_index
            x_y_dict['y_' + str(yi)] = len_yi_index
            x_y_dict['x_' + str(xi) + '_y_' + str(yi)] = x_y_count
            x_y_mutual = 0
            if x_y_count > 0:
                x_y_mutual = (x_y_count / len_x) * cal_log((len_x * x_y_count) / (len_xi_index * len_yi_index))
            mutual_info = mutual_info + x_y_mutual
    H_y = 0
    for yi in y_unique:
        yi_count = x_y_dict['y_' + str(yi)]
        H_y = H_y - (yi_count / len_x) * cal_log(yi_count / len_x)
    fraction_info = mutual_info / H_y
    return fraction_info

# kmeans聚类, data形状为多行一列
def find_kmeans_threshold(data):
    old_centers = np.zeros([2, 1])
    old_centers[0, 0] = data.min()
    old_centers[1, 0] = data.max()
    old_centers[0, 0] = 0.0
    while True:
        old_dist = pairwise_distances(data, old_centers, metric='euclidean')
        old_labels = np.argmin(old_dist, axis = 1).reshape([-1, 1])
        zero_index = np.where(old_labels == 0)[0]
        one_index = np.where(old_labels == 1)[0]
        zero_mean = data[zero_index].mean()
        one_mean = data[one_index].mean()
        new_centers = np.array([[zero_mean], [one_mean]])
        if one_mean < zero_mean:
            new_centers = np.array([[one_mean], [zero_mean]])
        new_centers = np.array([[0.0], [new_centers[1][0]]])
        new_dist = pairwise_distances(data, new_centers, metric='euclidean')
        new_labels = np.argmin(new_dist, axis = 1).reshape([-1, 1])
        if np.sum(new_labels == old_labels) == new_labels.shape[0]:
            break
        old_centers = new_centers
    labels = new_labels
    return old_centers, labels

# 给定数组的k组合
def kcombine(glist, k):
    def kcombine_helper(glist, glistbegin, pos, result, kresult):
        if pos == len(result):
            tc = result.copy()
            tc.sort()
            kresult.append(tc)
            return
        for gi in range(glistbegin, len(glist), 1):
            result[pos] = glist[gi]
            kcombine_helper(glist, gi + 1, pos + 1, result, kresult)

    kresult = []
    result = np.zeros(k)
    kcombine_helper(glist, 0, 0, result, kresult)
    return kresult

# 读取multree的图结构
def get_multree_graph_dict(graph_path, node_num):
    graph_dict = {}
    graph_matrix = np.zeros([node_num, node_num]).astype(int)
    fp = open(graph_path)
    line = fp.readline()
    while True:
        line = fp.readline()
        if (not line):
            break
        line_array = line.split('/')
        graph_matrix[int(line_array[0])][int(line_array[1])] = 1
    for i in range(node_num):
        graph_dict[i] = np.where(graph_matrix[:,i] > 0)[0].tolist()
    return graph_dict

if __name__ == "__main__":
    record_data_path = 'record_states.txt'
    multree_path = 'multree_graph.txt'
    record_df = pd.read_csv(record_data_path, sep = '\t', header = None)
    record_data = record_df.values
    node_num = record_data.shape[1]
    fio = np.zeros([node_num, node_num], dtype = np.float32)
    for ni in range(node_num):
        for nj in range(ni+1, node_num, 1):
            fio[ni][nj] = cal_fraction_info(record_data[:, ni:ni+1], record_data[:, nj:nj+1])
            fio[nj][ni] = cal_fraction_info(record_data[:, nj:nj+1], record_data[:, ni:ni+1])
    frs = np.array(fio).reshape([-1, 1])
    kcs, kls = find_kmeans_threshold(frs)
    kt = min(frs[ np.where(kls.reshape(-1) == 0)[0], :].max(), frs[np.where(kls.reshape(-1) == 1)[0], :].max())
    fios = np.zeros(fio.shape); fios[fio > kt] = 1
    old_graph_dict = get_multree_graph_dict(multree_path, node_num)
    graph_dict = {}
    for i in range(0, node_num, 1):
        i_fraction_info = fios[:, i]
        i_fraction_info[i] = -1
        i_big_mutual_info_index = np.where(i_fraction_info == 1)[0]
        i_parents_score_list = []
        if len(i_big_mutual_info_index) > 0:
            max_parents_num = len(i_big_mutual_info_index)
            for pi in range(1, max_parents_num + 1, 1):
                all_pi_parents_combination = kcombine(i_big_mutual_info_index, pi)
                for ai in range(len(all_pi_parents_combination)):
                    ai_parent = np.sort(all_pi_parents_combination[ai]).astype(int)
                    N =  record_data[:, i:i+1].shape[0] * 1.0; NW = 0.0
                    ki = record_data[:, ai_parent].shape[1] * 1.0
                    x_dict = {}; Fi_dict = {}; xFi_dict = {}
                    for i in range( record_data[:, i:i+1].shape[0]):
                        x_i =  record_data[:, i:i+1][i, 0]; Fi_i = tuple(record_data[:, ai_parent][i, :])
                        xFi_i = (x_i, ) + Fi_i; x_dict[x_i] = x_dict.get(x_i, 0) + 1.0
                        Fi_dict[Fi_i] = Fi_dict.get(Fi_i, 0) + 1.0; xFi_dict[xFi_i] = xFi_dict.get(xFi_i, 0) + 1.0
                    for xFiv, xFiv_count in xFi_dict.items():
                        xv = xFiv[0]; xv_count = x_dict[xv]
                        Fiv = xFiv[1:];  Fiv_count = Fi_dict[Fiv]
                        NW = NW + xFiv_count * cal_log((N * xFiv_count) / (xv_count * Fiv_count))
                    a_num = 0; m_num = 0; old_Fi = old_graph_dict[i]
                    for fi in ai_parent:
                        if fi not in old_Fi:
                            fi_parents = old_graph_dict[fi]; m_num = m_num + (i not in fi_parents)
                    for fi in old_Fi:
                        if fi not in ai_parent:
                            a_num = a_num + 1
                    score = 4.0 * np.power(2, ki) - NW + (a_num + m_num + ki) * cal_log(node_num)
                    i_parents_score_list.append([ai_parent, score])
            i_parents_score_list = sorted(i_parents_score_list, key = lambda x:x[1])
            i_parents = set(i_parents_score_list[0][0])
            i_parents = np.sort(list(i_parents))
            graph_dict[i] = i_parents

