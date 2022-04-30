from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# create dataset from csv, use quality as label
def createDataset(path, mode):
    with open(path) as f:
        f_csv = csv.reader(f)
        next(f_csv)
        if(mode == 0):
            features = []
            labels = []
            for row in f_csv:
                float_lst = [float(num) for num in row]
                features.append(float_lst[1:len(float_lst) - 1])
                labels.append(int(float_lst[-1]))
            return features, labels
        else:
            features = []
            for row in f_csv:
                float_lst = [float(num) for num in row]
                features.append(float_lst[1:len(float_lst)])
            return features

train_data_path = 'train.csv'
test_data_path = 'test.csv'
output_path = 'output.txt'
train_data, train_labels = createDataset(train_data_path, 0)
test_data = createDataset(test_data_path, 1)

n_estimators = [10, 30, 50]
with open(output_path, mode='w') as f:
    for n in n_estimators:
        f.write('Number of trees: %d\n' %n)
        rForest = RandomForestClassifier(n_estimators=n)

        rForest = rForest.fit(train_data, train_labels)

        train_predict = rForest.predict(train_data)
        accuracy = accuracy_score(train_labels, train_predict)
        test_predict = rForest.predict(test_data)

        df = pd.DataFrame({'label' : test_predict})
        df.to_csv('submission_'+str(n)+'.csv', index = True, sep = ',')
        f.write('accuracy: %.3f\n' %accuracy)
