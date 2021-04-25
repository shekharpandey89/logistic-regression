

# We are importing all the required libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Now,we are reading our csv dataset.

dataset = pd.read_csv("ms_admission.csv")
print(dataset)

# We are now printing all the columns avaiable in the dataset.
print(dataset.columns)

# We assigned the independent variables to X and dependent varibale to y.
X = dataset[['gre', 'gpa','work_experience']]
y = dataset['admitted']

# We are printing to top 5 elements using the head().
print(X.head())
print(y.head())

# We are now dividing the whole dataset into taining and test. For the test we choose 25% and 75% dataset
# for training
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#create prediction model
model = LogisticRegression()
#fit model
model.fit(X_train, y_train)

# Now, we are passing the test data to the model for the prediction.
y_predictions = model.predict(X_test)
print(y_predictions)

print("prediction: {} ".format(accuracy_score(y_test,y_predictions) * 100))
print(classification_report(y_test, y_predictions))

#plotting confusion matrix on heatmap
confusion_matrix = confusion_matrix(y_test, y_predictions)
sns.heatmap(confusion_matrix, annot=True, xticklabels=['not_admitted','admitted'], yticklabels=['not_admitted','admitted'])
# sns.heatmap(confusion_matrix, annot=True)

plt.figure(figsize=(3,3))
plt.show()

# We printing the top 5 elements of the X_test (independent variables) data.
X_test.head()

# We printing the top 5 elements of the y_test (dependent variable) data.
y_test.head()

# We just printing the top 5 results
# We found that from the above truth y_test only one person should be admitted and else not because
# only 341 index value has 1 value (admitted) and else has values 0 (not admitted).

# And in the prediction results we got all results same except the 3rd item which should be 1 (admitted).
# So our predictions is woring fine.
y_predictions[:5]

## Now, we are going to check the prediction for the new dataset. We are going to create a new
# dataframe and we will test new dataframe on the trained model.

new_testData = {'gre': [595,735,682,613,715],
                  'gpa': [2.1,4,3.4,2.4,3],
                  'work_experience': [4,4,5,2,4]
                  }
test_data = pd.DataFrame(new_testData,columns= ['gre', 'gpa','work_experience'])
print(test_data)
y_pred = model.predict(test_data)
# We got the predictions on the new dataset that no one will be select for the admission.
y_pred
