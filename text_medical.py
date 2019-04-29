from skmultilearn.dataset import load_from_arff

X_train, y_train = load_from_arff("./medical/medical-train.arff", 
    # number of labels
    label_count=45)
X_test, y_test = load_from_arff("./medical/medical-test.arff", 
    # number of labels
    label_count=45)
# print(X_train, y_train)
print(X_train.todense())
print(X_train.shape)
# print(X_test, y_test)