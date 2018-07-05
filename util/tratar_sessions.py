import pandas as pd



df = pd.read_csv('../input/sessions_reduzido.csv', decimal='.')
df = df.sort_values('user_id')
df = df.iloc[ :5000 , :]

# print(df)

# raise Exception()

# loc[df['column_name'] == some_value]
unique_users_ids = df['user_id'].unique()
unique_actions = df['action'].unique()
# unique_users_ids = df['action_type'].unique()
# unique_users_ids = df['user_id'].unique()
# print(unique_users_ids)

print(" - - - - ACTIONS  - - - - ")
for action in unique_actions:
	print(action)	

print(df['action'].value_counts())	

# print(df.loc[df['user_id'] == 'zzzlylp57e'])

final_df = pd.DataFrame(columns=['user_id', 'count_actions', 'sum_secs_elapsed']);
i = 0
for user_id in unique_users_ids:
	user_df = df.loc[df['user_id'] == user_id]
	# final_df.set_value(user_id, user_df.size, user_df['secs_elapsed'].sum())
	final_df.loc[i] = [user_id, user_df.size, user_df['secs_elapsed'].sum()]
	i+=1
	# print(user_id,' : ', user_df.size,' : ' ,user_df['secs_elapsed'].sum())
	# print(user_df['action'].value_counts())
final_df.to_csv('../results/sessions_processed_2.csv')

	