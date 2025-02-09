import pandas as pd


# Prints out all different values in a column and the number of occurences
def analyze_column(df, col):
    g = df.groupby(col).count()
    print(g['id'])
    pass


# Basic one hot encoding, the function will split a feature into several features, one of each
# value in the original feature list
def one_hot_encode(df, feature):
    feature_labels = df[feature].drop_duplicates().values

    for ft in feature_labels:
        feature_name = (feature + "_" + str(ft)).replace(" ", "_").replace("-", "").replace("/", "").replace("(", "").replace(
            ")", "")
        df[feature_name] = 0
        # print(feature_name)
        df.loc[(df[feature] == ft), feature_name] = 1
    df.drop(feature, axis=1, inplace=True)
    return df


# Split a date into two separet colummns for year and month
# Days and the different time units of the days cycle to much to realistically have anything to do
# With the final decision
## This is unused for now, i have decided not to use dates at all since there is no real correlation between the training
# dates and test dates
def split_date_ym(df, date_column):
    df[date_column + '_Y'] = df[date_column].dt.year
    df[date_column + '_M'] = df[date_column].dt.month
    df[date_column + '_Q'] = df[date_column].dt.quarter
    one_hot_encode(df, date_column + '_M')
    df.drop(date_column, axis=1, inplace=True)
    return df


test_users_url = '../input/test_users.csv'
train_users_url = '../input/train_users_2.csv'

train_users = pd.read_csv(train_users_url, index_col=False)
test_users = pd.read_csv(test_users_url, index_col=False)

users = pd.concat([train_users, test_users])
# users = train_users

# Fix up dates
users['date_account_created'] = pd.to_datetime(users['date_account_created'], format='%Y-%m-%d')
users['timestamp_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')

# Drop date first booking since its useless information
users.drop(['date_first_booking'], axis=1, inplace=True)

users.reset_index(inplace=True)
users.drop('index', axis=1, inplace=True)

# Fix ages
users.loc[(users['age'] > 1900), 'age'] = 2015 - users.age

# Remove trolls by placing them in bucket 5-10
users.loc[users['age'] > 95, 'age'] = 7
users.loc[users['age'] < 15, 'age'] = 7

# put the users who didnt have any age in bucket 10-15
users['age'] = users['age'].fillna(13)

# use bucketing to place users into agegroups in 5 year ranges
for i in range(5, 95, 5):
    col_name = '{}-{}'.format(i, i + 5)
    users[col_name] = 0
    users.loc[(users['age'] > i) & (users['age'] <= (i + 5)), col_name] = 1

features_to_1h = ['affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'first_browser', 'first_device_type', 'gender', 'language', 'signup_app', 'signup_flow', 'signup_method']
for feature in features_to_1h:
    one_hot_encode(users, feature)

# Fill upp the missing account created entries with the first active, might change this to something unique later
# users['date_account_created'].fillna(test_users.timestamp_first_active, inplace=True)

# one-hot encode years and months
split_date_ym(users, 'date_account_created')
split_date_ym(users, 'timestamp_first_active')

print(len(users))
users.to_csv('../input/users_clean_train_2.csv', index=False)