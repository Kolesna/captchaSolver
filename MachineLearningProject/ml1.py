from sklearn.tree import DecisionTreeClassifier
import numpy

dataTrain = [[1],[2],[3],[4],[5],[6],[7],[8],[9]]
dataLabel = ['One','Two','Three','Four','Five','Six','Seven','Eight','Nine']
model = DecisionTreeClassifier(random_state=42)
model.fit(dataTrain, dataLabel)

for dataItem in dataTrain:
    prediction = model.predict([dataItem])
    print(f"Model predicts {dataItem} is {prediction}")

while True:
    try:
        userVal = float(input('New data value: '))
    except (EOFError, KeyboardInterrupt):
        break
    except ValueError:
        print('Enter a number')
        continue
    prediction = model.predict([[userVal]])
    print(f"Model predicts {userVal} is {prediction}")