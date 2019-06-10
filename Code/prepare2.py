import pandas as pd
import json


def main():
    Card_Base = pd.read_csv('./raw/CardBase.csv', dtype={0: 'str', 1: 'str', 2: 'float', 3: 'str'})
    Customer_Base = pd.read_csv('./raw/CustomerBase.csv', dtype={0: 'str', 1: 'float', 2: 'str', 3: 'str'})
    Fraud_Base = pd.read_csv('./raw/FraudBase.csv', dtype={0: 'str', 1: 'float'})
    mydateparser = lambda x: pd.datetime.strptime(x, "%d/%m/%Y")
    Transaction_Base = pd.read_csv('./raw/TransactionBase.csv', dtype={0: 'str', 2: 'str', 3: 'float', 4: 'str'},
                                   parse_dates=[1], date_parser=mydateparser)

    list_fraud = list(Fraud_Base['Transaction_ID'])
    count = 0
    F_count = 0
    file_json = {}
    for i in range(len(Card_Base)):
        Card_ID = Card_Base.loc[i, 'Card_Number']
        data = pd.DataFrame(Transaction_Base[Transaction_Base['Credit_Card_ID'] == Card_ID])
        data = data.sort_values(by=['Transaction_Date'])
        data = data.reset_index(drop=True)
        bo = False
        for j in list_fraud:
            index = data.index[data['Transaction_ID'] == j].tolist()
            if index != list():
                if index[0] < 6:
                    bo = True
                else:
                    file_json[count + 1] = False
                    F_count += 1
                    data = data.truncate(before=0, after=index[0])
        if len(data) < 6:
            bo = True
        if bo:
            continue
        count += 1
        path_file = './data2/TransactionID_' + str(count) + '.csv'
        data.to_csv(path_file, index=False)
    with open('./raw/dict_fraud_transaction.json', 'w') as f:
        json.dump(file_json, f)


if __name__ == "__main__":
    main()
