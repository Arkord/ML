import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#pd.options.display.max_rows = 120;
#pd.options.display.min_rows = 20;

df = pd.read_csv('../datasets/PS_20174392719_1491204439457_log.csv')

df = df.drop(
    ['nameOrig', 'nameDest', 'isFlaggedFraud'],
    axis=1
)

# storing the fraudulent data into a dataframe
df_fraud = df[df['isFraud'] == 1]

# storing the non-fraudulent data into a dataframe
df_nofraud = df[df['isFraud'] == 0]
df_nofraud = df_nofraud.head(12000)

# Joining both datasets together
df = pd.concat(
    [df_fraud, df_nofraud],
    axis=0
)

# converting the type column to categorical
df['type'] = df['type'].astype('category')

# integer encoding the type column
type_encode = LabelEncoder()
df['type'] = type_encode.fit_transform(df.type)

# nominal variables require one-hot encode
type_one_hot = OneHotEncoder()
type_one_hot_encode = type_one_hot.fit_transform(
    df.type.values.reshape(-1, 1)
).toarray()

ohe_variable = pd.DataFrame(
    type_one_hot_encode,
    columns=[
        "type_" + str(int(i)) for i in range(type_one_hot_encode.shape[1])
    ]
)

df = pd.concat(
    [
        df, 
        ohe_variable
    ],
    axis=1
)

# Dropping the original variable type
df = df.drop('type', axis=1)

#print(df.head())
#print(df)

df = df.fillna(0)
df.to_csv('../datasets/fraud_prediction.csv')

print(df.isnull().any())