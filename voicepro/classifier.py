import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def generate() :

    # 읽어오기
    df = pd.read_csv('./data.csv', encoding='euc-kr')
    df.columns = ['singer'] + [x for x in range(len(df.columns) - 1)]
    name = df['singer']
    del df['singer']

    # 스케일링
    scale = StandardScaler()
    scale.fit(df)
    scaled_X = scale.transform(df)

    # 라벨링
    k = 7
    model = KMeans(n_clusters=k, algorithm='auto')
    model.fit(scaled_X)
    predict = pd.DataFrame(model.predict(scaled_X))
    predict.columns = ['predict']
    string = 'abcdefg'
    for i in range(k):
        predict['predict'] = predict['predict'].replace(i, string[i])

    # 저장
    df_label = pd.concat([name, df, predict], axis=1)
    df_label.to_csv('./data_label.csv', encoding='euc-kr')


if __name__ == '__main__':
    generate()