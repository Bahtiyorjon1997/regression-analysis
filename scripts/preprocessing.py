import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_inspect_data(file_path):
    df = pd.read_csv(file_path)
    print(df.info())
    print(df.describe())
    return df


def preprocess_data(df):
    scaler = StandardScaler()
    features = df[['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen',
                   'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']]
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled['BodyFat'] = df['BodyFat']
    return df_scaled


if __name__ == "__main__":
    file_path = 'data/raw/bodyfat.csv'
    load_and_inspect_data(file_path)
    df = load_and_inspect_data(file_path)
    print('data frame', df)

    df_processed = preprocess_data(df)
    print("data processed", df_processed)

    df_processed.to_csv('data/processed/bodyfat_processed.csv', index=False)
