import pandas as pd

df_inter = pd.read_csv('./dataset/Green_Rec/valid_data.txt', sep=',')
df_item = pd.read_csv('./dataset/Green_Rec/recipe_three_scores.csv', sep=',',
                       dtype={'recipeid': int, 'env_score':float, 'nutri_score':float, 'meal_score':float, })
df_inter = df_inter.rename(
    columns={'users': 'user_id:token', 'items': 'item_id:token', 'ratings': 'rating:float', 'time': 'timestamp:float'}
)
df_item = df_item.rename(
    columns={'recipeid': 'item_id:token', 'env_score':'env_score:token',
             'nutri_score':'nutri_score:token', 'meal_score':'meal_score:token', }
)
print(df_item)
print(df_inter)
# item_id:token	env_score:token	nutri_score:token	meal_score:token
# user_id:token	item_id:token	rating:float	timestamp:float
def transfer_float_2_int(list_score, alpha):
    m0, m255 = 0, 0
    min_value = min(list_score)
    for i in range(len(list_score)):
        list_score[i] -= min_value
    max_value = max(list_score)
    for i in range(len(list_score)):
        list_score[i] = int(alpha*(list_score[i])/max_value)
        if list_score[i] == 0:
            m0 += 1
        if list_score[i] == 255:
            m255 += 1
    print(f'max: {max(list_score)}, min: {min(list_score)}')
    print(m0)
    print(m255)
    return list_score
df_item['env_score:token'] = transfer_float_2_int(list(df_item['env_score:token']), 255)
df_item['nutri_score:token'] = transfer_float_2_int(list(df_item['nutri_score:token']), 255)
list_meal = list(df_item['meal_score:token'])

for i in range(len(list_meal)):
    list_meal[i] = int(list_meal[i]/5)

print(f'max: {max(list_meal)}, min: {min(list_meal)}')
df_item['meal_score:float'] = list_meal
df_item.to_csv('./dataset/Green_Rec/Green_Rec.item', sep='\t', index=False,
               columns=['item_id:token','env_score:token','nutri_score:token','meal_score:token'])
df_inter.to_csv('./dataset/Green_Rec/Green_Rec.inter', sep='\t', index=False)

