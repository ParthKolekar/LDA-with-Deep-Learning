import json
import os

import numpy
from nltk.corpus import reuters
from sklearn.svm import SVC
from utils import OUTPUTS_DIR, TESTING_SET, TRAINING_SET

svmmodel = SVC(kernel='linear')

# remove things which have multiple classes
TRAINING_SET = list(filter(lambda x: len(reuters.categories(x)) == 1, TRAINING_SET))
TESTING_SET = list(filter(lambda x: len(reuters.categories(x)) == 1, TESTING_SET))

X = []
for i in TRAINING_SET:
    with open(os.path.join(OUTPUTS_DIR, i)) as f:
        X.append(json.load(f))
X = numpy.array(X)

y = []  # Yes, this is a small letter y. No, that is not a mistake.
for i in TRAINING_SET:
    y.append(reuters.categories(i))
y = numpy.array(y)

svmmodel.fit(X, y.ravel())

Z = []
for i in TESTING_SET:
    with open(os.path.join(OUTPUTS_DIR, i)) as f:
        Z.append(json.load(f))
Z = numpy.array(Z)

total = list(map(lambda x: x[0] == x[1], list(zip(svmmodel.predict(Z), map(lambda x: x[0], map(lambda x: reuters.categories(x), TESTING_SET))))))

print(sum(total) / len(total))
