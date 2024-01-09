import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import os

features = ['Designation', 'City', 'Gender', 'Rating', 'Salary', 'WorkLifeBalance', 'CurrentLocation', 'Appraisal', 'Recommend', 'Opportunities', 'CurrentRole', 'Infrastructure']

def create_map(values):
	counter = 0
	mapping = {}
	for value in values:
		mapping[value] = counter
		counter += 1
	return mapping

def visualize_tree(tree, feature_names, name):
	with open("{}.dot".format(name), 'w') as f:
		export_graphviz(tree, out_file=f, feature_names=feature_names)

	command = ["dot", "-Tpng", "{}.dot".format(name), "-o", "{}.png".format(name)]
	command = " ".join(command)

	try:
		os.system(command)
	except:
		exit("Could not run dot, ie graphviz, to  produce visualization")


def encode_columns(df):
	gender_map = {'M': 0,'F': 1} 
	df.Gender = [gender_map[item] for item in df.Gender]

	result_map = {"Sustain": 0, "Enhance":1}
	try:
		df.Employer = [result_map[item] for item in df.Employer]
		df.Employee = [result_map[item] for item in df.Employee]
	except:
		pass

	designation_map = create_map(df["Designation"].unique())
	df.Designation = [designation_map[item] for item in df.Designation]

	city_map = create_map(df["City"].unique())
	df.City = [city_map[item] for item in df.City]
	return df

def read_csv(year):
	df = pd.read_csv("{}.csv".format(year))
	df = encode_columns(df)	
	return df

def training(year, target, loops=1):
	df = read_csv(year)
	zeros = df[df[target] == 0]
	ones = df[df[target] == 1]
	mean_accuracy = 0.0
	max_accuracy = 0.0
	max_tree = None

	for i in range(loops):
		train0, test0 = train_test_split(zeros, test_size=0.5, shuffle=True)
		train1, test1 = train_test_split(ones, test_size=0.2, shuffle=True)
		train =  pd.concat([train0, train1])
		test = pd.concat([test0, test1])

		y = train[target]
		X = train[features]
		dt = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=len(features), min_samples_leaf=5)
		dt.fit(X, y)

		y_pred = dt.predict(test[features])
		accuracy = accuracy_score(test[target],y_pred)*100
		accuracy = round(accuracy, 5)
		if accuracy > max_accuracy:
			max_accuracy = accuracy
			max_tree = dt
		mean_accuracy += accuracy
	mean_accuracy /= loops
	mean_accuracy = round(mean_accuracy, 5)
	print ("=====================")
	print ("Total runs: {}".format(loops))
	print ("Target: ", target)
	print ("Max accuracy: {}".format(max_accuracy))
	print ("Mean accuracy: {}".format(mean_accuracy))
	print ("=====================")
	return max_tree

def testing(year, loops):
	df = read_csv(year)
	original_df = pd.read_csv("{}.csv".format(year))

	for target in ["Employer", "Employee"]:
		max_tree = training("2016", target, loops=loops)
		pred = max_tree.predict(df[features])
		original_df[target] = pred
		visualize_tree(max_tree, features, "{}_max_tree".format(target))
	result_map = {0 : "Sustain", 1: "Enhance"}
	original_df.Employer = [result_map[item] for item in original_df.Employer]
	original_df.Employee = [result_map[item] for item in original_df.Employee]
	print ("Csv with predicted values created...")
	original_df.to_csv("{}_predicted.csv".format(year))

testing("2017", 1)
