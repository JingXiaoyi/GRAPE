import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

main_file = '../Dataset/'
from sklearn.model_selection import train_test_split


def dataset_split():
    inter_file = main_file + 'valid_data.txt'
    processed_path = main_file + 'processed/'
    green_file = main_file + 'recipe_three_scores.csv'
    df_inter = pd.read_csv(inter_file, dtype={'users': int, 'items': int, 'ratings': int, 'time': int})
    df_green = pd.read_csv(green_file,
                           dtype={'recipeid': int, 'env_score': float, 'nutri_score': float, 'meal_score': float})
    print(f'len df_green: {len(df_green)}')
    print(f'mean env: {np.mean(df_green["env_score"])}, '
          f'mean nutri: {np.mean(df_green["nutri_score"])}, '
          f'mean meal: {np.mean(df_green["meal_score"])}')
    # print(df_inter)

    # filter all inter which the item have green score
    # if_in_green = [False for _ in range(min(df_green['recipeid']) + 1)]
    # for item in df_green['recipeid']:
    #     if_in_green[item] = True

    # print(df_inter)
    print(f'len(df_inter) before filter have green scores: {len(df_inter)}')
    ll = list(df_green['recipeid'])
    df_inter = df_inter[df_inter['items'].isin(ll)]
    df_inter.reset_index(drop=True, inplace=True)
    # print(df_inter)
    print(f'len(df_inter) after filter have green scores: {len(df_inter)}')

    # fiter user with interactions less than 10
    dict_num_user_inters = {}
    for i in range(len(df_inter)):
        user_now = df_inter['users'][i]
        if user_now not in dict_num_user_inters:
            dict_num_user_inters[user_now] = 0
        dict_num_user_inters[user_now] += 1

    inter_limit_num = 10
    list_user_limit = []
    for user in dict_num_user_inters.keys():
        if dict_num_user_inters[user] >= inter_limit_num:
            list_user_limit.append(user)

    print(f'before filter inter_nums, len df_inter: {len(df_inter)}')
    df_inter = df_inter[df_inter['users'].isin(list_user_limit)]
    print(f'after filter inter_nums, len df_inter: {len(df_inter)}')

    df_train, df_val_test = train_test_split(df_inter, test_size=0.2, random_state=2024)
    df_valid, df_test = train_test_split(df_val_test, test_size=0.5, random_state=2024)

    print(f'before filter cold start, len train: {len(df_train)}, len valid: {len(df_valid)},'
          f' len test: {len(df_test)}, in total: {len(df_train) + len(df_valid) + len(df_test)}')
    # filter validation and test to ensure each user and item have occured in train set
    list_user_train = list(df_train['users'])
    list_item_train = list(df_train['items'])

    df_green = df_green[df_green['recipeid'].isin(list_item_train)]
    print(f'len item in df_green: {len(df_green)}')
    print(f'mean env: {np.mean(df_green["env_score"])}, '
          f'mean nutri: {np.mean(df_green["nutri_score"])}, '
          f'mean meal: {np.mean(df_green["meal_score"])}')

    df_valid = df_valid[df_valid['users'].isin(list_user_train)]
    df_valid = df_valid[df_valid['items'].isin(list_item_train)]
    df_test = df_test[df_test['users'].isin(list_user_train)]
    df_test = df_test[df_test['items'].isin(list_item_train)]
    print(f'before filter cold start, len train: {len(df_train)}, len valid: {len(df_valid)},'
          f' len test: {len(df_test)}, in total: {len(df_train) + len(df_valid) + len(df_test)}')

    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    dict_map_user_idx, dict_map_item_idx = {}, {}
    num_user, num_item = 0, 0

    # replace user, item id with index
    for i in range(len(df_train)):
        user_now = df_train['users'][i]
        item_now = df_train['items'][i]
        if user_now not in dict_map_user_idx:
            dict_map_user_idx[user_now] = num_user
            num_user += 1
        if item_now not in dict_map_item_idx:
            dict_map_item_idx[item_now] = num_item
            num_item += 1
    print(f'num user: {num_user}, num item: {num_item}')

    def replace_with_idx(df, dict_map_user_idx, dict_map_item_idx):
        list_user = list(df['users'])
        list_item = list(df['items'])
        for i in range(len(list_user)):
            list_user[i] = dict_map_user_idx[list_user[i]]
            list_item[i] = dict_map_item_idx[list_item[i]]
        df['users'] = list_user
        df['items'] = list_item
        return df
    df_train = replace_with_idx(df_train, dict_map_user_idx, dict_map_item_idx)
    df_valid = replace_with_idx(df_valid, dict_map_user_idx, dict_map_item_idx)
    df_test = replace_with_idx(df_test, dict_map_user_idx, dict_map_item_idx)

    list_item_green = list(df_green['recipeid'])
    for i in range(len(df_green)):
        list_item_green[i] = dict_map_item_idx[list_item_green[i]]
    df_green['recipeid'] = list_item_green
    df_green.reset_index(drop=True, inplace=True)

    # set normalized green value
    print('Normalizing green value')
    list_env = list(df_green['env_score'])
    max_env = max(list_env)
    normal_env = [1-(i/max_env) for i in list_env]
    df_green['n_env_score'] = normal_env

    list_nutri = list(df_green['nutri_score'])
    max_nutri = max(list_nutri)
    normal_nutri = [(i/max_nutri) for i in list_nutri]
    df_green['n_nutri_score'] = normal_nutri

    list_meal = list(df_green['meal_score'])
    max_meal = max(list_meal)
    normal_meal = [(i/max_meal) for i in list_meal]
    df_green['n_meal_score'] = normal_meal

    save_train_file = processed_path + 'green_rec_train.csv'
    save_valid_file = processed_path + 'green_rec_valid.csv'
    save_test_file = processed_path + 'green_rec_test.csv'
    save_green_file = processed_path + 'green_rec_green.csv'
    df_train.to_csv(save_train_file, header=True, index=True, sep=',')
    df_valid.to_csv(save_valid_file, header=True, index=True, sep=',')
    df_test.to_csv(save_test_file, header=True, index=True, sep=',')
    df_green.to_csv(save_green_file, header=True, index=True, sep=',')

def dataset_split_GRAPE():
    inter_file = main_file + 'valid_data.txt'
    processed_path = main_file + 'processed/'
    green_file = main_file + 'recipe_three_scores.csv'
    df_inter = pd.read_csv(inter_file, dtype={'users': int, 'items': int, 'ratings': int, 'time': int})
    df_green = pd.read_csv(green_file,
                           dtype={'recipeid': int, 'env_score': float, 'nutri_score': float, 'meal_score': float})
    print(f'len df_green: {len(df_green)}')
    print(f'mean env: {np.mean(df_green["env_score"])}, '
          f'mean nutri: {np.mean(df_green["nutri_score"])}, '
          f'mean meal: {np.mean(df_green["meal_score"])}')
    # print(df_inter)

    # filter all inter which the item have green score
    # if_in_green = [False for _ in range(min(df_green['recipeid']) + 1)]
    # for item in df_green['recipeid']:
    #     if_in_green[item] = True

    # print(df_inter)
    print(f'len(df_inter) before filter have green scores: {len(df_inter)}')
    ll = list(df_green['recipeid'])
    df_inter = df_inter[df_inter['items'].isin(ll)]
    df_inter.reset_index(drop=True, inplace=True)
    # print(df_inter)
    print(f'len(df_inter) after filter have green scores: {len(df_inter)}')

    # fiter user with interactions less than 10
    dict_num_user_inters = {}
    for i in range(len(df_inter)):
        user_now = df_inter['users'][i]
        if user_now not in dict_num_user_inters:
            dict_num_user_inters[user_now] = 0
        dict_num_user_inters[user_now] += 1

    inter_limit_num = 10
    list_user_limit = []
    for user in dict_num_user_inters.keys():
        if dict_num_user_inters[user] >= inter_limit_num:
            list_user_limit.append(user)

    print(f'before filter inter_nums, len df_inter: {len(df_inter)}')
    df_inter = df_inter[df_inter['users'].isin(list_user_limit)]
    print(f'after filter inter_nums, len df_inter: {len(df_inter)}')

    # filter validation and test to ensure each user and item have occured in train set
    list_user_train = list(df_inter['users'])
    list_item_train = list(df_inter['items'])

    df_green = df_green[df_green['recipeid'].isin(list_item_train)]
    print(f'len item in df_green: {len(df_green)}')
    print(f'mean env: {np.mean(df_green["env_score"])}, '
          f'mean nutri: {np.mean(df_green["nutri_score"])}, '
          f'mean meal: {np.mean(df_green["meal_score"])}')


    df_inter.reset_index(drop=True, inplace=True)

    dict_map_user_idx, dict_map_item_idx = {}, {}
    num_user, num_item = 0, 0

    # replace user, item id with index
    for i in range(len(df_inter)):
        user_now = df_inter['users'][i]
        item_now = df_inter['items'][i]
        if user_now not in dict_map_user_idx:
            dict_map_user_idx[user_now] = num_user
            num_user += 1
        if item_now not in dict_map_item_idx:
            dict_map_item_idx[item_now] = num_item
            num_item += 1
    print(f'num user: {num_user}, num item: {num_item}')

    def replace_with_idx(df, dict_map_user_idx, dict_map_item_idx):
        list_user = list(df['users'])
        list_item = list(df['items'])
        for i in range(len(list_user)):
            list_user[i] = dict_map_user_idx[list_user[i]]
            list_item[i] = dict_map_item_idx[list_item[i]]
        df['users'] = list_user
        df['items'] = list_item
        return df
    df_inter = replace_with_idx(df_inter, dict_map_user_idx, dict_map_item_idx)

    list_item_green = list(df_green['recipeid'])
    for i in range(len(df_green)):
        list_item_green[i] = dict_map_item_idx[list_item_green[i]]
    df_green['recipeid'] = list_item_green
    df_green.reset_index(drop=True, inplace=True)

    # set normalized green value
    print('Normalizing green value')
    list_env = list(df_green['env_score'])
    max_env = max(list_env)
    min_env = min(list_env)
    normal_env = [(1-(i/max_env)) % 0.01 -1 for i in list_env]
    df_green['n_env_score'] = normal_env

    list_nutri = list(df_green['nutri_score'])
    max_nutri = max(list_nutri)
    min_nutri = min(list_nutri)
    normal_nutri = [(1-(i/max_nutri)) % 0.01 -1 for i in list_nutri]
    df_green['n_nutri_score'] = normal_nutri

    list_meal = list(df_green['meal_score'])
    max_meal = max(list_meal)
    min_meal = min(list_meal)
    normal_meal = [(1-(i/max_meal)) % 0.01 -1 for i in list_meal]
    df_green['n_meal_score'] = normal_meal

    print(f'max_env: {max_env}, max_nutri: {max_nutri}, max_meal: {max_meal}')
    print(f'min_env: {min_env}, min_nutri: {min_nutri}, min_meal: {min_meal}')
    print(f'mean_env: {np.mean(list_env)}, mean_nutri: {np.mean(list_nutri)}, mean_meal: {np.mean(list_meal)}')

    save_inter_file = processed_path + 'green_baseline.inter'
    save_green_file = processed_path + 'green_baseline.item'
    '''
    
    df_inter = pd.read_csv(inter_file, dtype={'users': int, 'items': int, 'ratings': int, 'time': int})
    df_green = pd.read_csv(green_file,
                           dtype={'recipeid': int, 'env_score': float, 'nutri_score': float, 'meal_score': float})
    '''

    # leave-one-out

    print(f'len df_inter: {len(df_inter)}')
    dict_user_latest, dict_user_2_latest = {}, {} # user: [time, item, i]
    list_test, list_valid = [], []
    for i in range(len(df_inter)):
        user_now = df_inter['users'][i]
        item_now = df_inter['items'][i]
        time_now = df_inter['time'][i]
        if user_now not in dict_user_latest:
            dict_user_latest[user_now] = [time_now, item_now, i]
        if time_now > dict_user_latest[user_now][0]:
            dict_user_2_latest[user_now] = dict_user_latest[user_now].copy()
            dict_user_latest[user_now] = [time_now, item_now, i]
    print(f'\nlen dict_user_latest: {len(dict_user_latest)}, len dict_user_2_latest: {len(dict_user_2_latest)}')

    for key in dict_user_latest.keys():
        list_test.append(dict_user_latest[key][1])
    for key in dict_user_latest.keys():
        list_valid.append(dict_user_2_latest[key][1])

    # generate dict_item_green_value
    dict_item_green_value = {}
    list_item_df_g = []
    for i in range(len(df_green)):
        item_now = df_green['recipeid'][i]
        dict_item_green_value[item_now] = [df_green['env_score'][i],
                                 df_green['nutri_score'][i],
                                 df_green['meal_score'][i]]


    list_test_green_value = []
    for key in list_valid:
        list_test_green_value.append(dict_item_green_value[key])
    print('mean env')
    print(np.mean([i[0] for i in list_test_green_value]))
    print('mean nutri')
    print(np.mean([i[1] for i in list_test_green_value]))
    print('mean meal')
    print(np.mean([i[2] for i in list_test_green_value]))


    # df_train =



    # df_train.to_csv(save_train_file, header=True, index=True, sep=',')
    # df_green.to_csv(save_green_file, header=True, index=True, sep=',')


def eva_green(df_green: pd.DataFrame, generated_list):
    list_recipeid = list(df_green['recipeid'])
    list_env_score, list_nutri_score, list_meal_score = [], [], []
    for item in generated_list:
        if item not in list_recipeid:
            print(f'ERROR_JING: item not found in df_green, item id: {item}')
        list_env_score.append(df_green['env_score'])
        list_nutri_score.append(df_green['nutri_score'])
        list_meal_score.append(df_green['meal_score'])
    return np.mean(list_env_score), np.mean(list_nutri_score), np.mean(list_meal_score)


if __name__ == '__main__':
    dataset_split_GRAPE()