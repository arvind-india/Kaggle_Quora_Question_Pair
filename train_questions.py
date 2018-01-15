import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, cross_validation

df = pd.read_csv('train.csv')
df.drop(['id'],1,inplace=True)
df = df.dropna()

def handle_non_numerical_data(df):
	columns = df.columns.values # column names into list

	for column in columns:
		text_digit_vals = {}

		def convert_to_int(val):
			return text_digit_vals[val]

		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = list(set(column_contents))
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1

			df[column] = list(map(convert_to_int,df[column]))

	return df

df = handle_non_numerical_data(df)
'''
with open('data_dump.pickle','rb') as int_string:
	df = pickle.load(int_string)
print df
'''
ft = np.array(df.drop(['is_duplicate'],1))
lt = np.array(df['is_duplicate'])

features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(ft,lt,test_size=0.2)

# RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test,pred)
print(accuracy) # 0.704271686166

# AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test,pred)
print(accuracy) # 0.68163941725

# GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test,pred)
print(accuracy) # 0.661159068985
