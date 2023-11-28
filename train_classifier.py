import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

labels_dict = {
    0: 'Aa',
    1: 'Bb',
    2: 'Cc',
    3: 'Dd',
    4: 'Ee',
    5: 'Ff',
    6: 'Gg',
    7: 'Hh',
    8: 'Ii',
    9: 'Jj',
    10: 'Kk',
    11: 'Ll',
    12: 'LLll',
    13: 'Mm',
    14: 'Nn',
    16: 'Oo',
    17: 'Pp',
    18: 'Qq',
    19: 'Rr',
    20: 'Ss',
    21: 'Tt',
    22: 'Uu',
    23: 'Vv',
    24: 'Ww',
    25: 'Xx',
    26: 'Yy',
    27: 'Zz'
}

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# Calculate and display the confusion matrix
cm = confusion_matrix(y_predict, y_test)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=list(labels_dict.values()),
            yticklabels=list(labels_dict.values()))
plt.xlabel('Predición del modelo')
plt.ylabel('Datos reales')
plt.title('Matriz de confusión')
plt.show()
