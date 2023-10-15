import numpy as np
import pandas as pd
import bentoml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier


vl_df = pd.read_csv("/Users/alexmirugwe/Work_Git_Projects/data/viral_load.csv")


sample_1 = vl_df[vl_df['suppressed'] == 1].sample(n=600000, random_state=42)
category_2_3 = vl_df[vl_df['suppressed'].isin([2, 3])]
sampled_vl_df = pd.concat([sample_1, category_2_3], ignore_index=True)

#replacing 3 with 2
sampled_vl_df['suppressed'] = sampled_vl_df['suppressed'].replace(3, 2)

sampled_vl_df = sampled_vl_df[['gender',' age','treatment_duration','current_regimen','Indication_for_VL_Testing','arv_adherence','suppressed']]

sampled_vl_df[" age"] = pd.to_numeric(sampled_vl_df[" age"], errors='coerce')

sampled_vl_df["gender"] = sampled_vl_df["gender"].astype("category")
sampled_vl_df["treatment_duration"] = sampled_vl_df["treatment_duration"].astype("category")
sampled_vl_df["suppressed"] = sampled_vl_df["suppressed"].astype("category")
sampled_vl_df["current_regimen"] = sampled_vl_df["current_regimen"].astype("category")

#excluding undefined gender
elements_to_check = ["M","F"]
sampled_vl_df = sampled_vl_df[sampled_vl_df['gender'].isin(elements_to_check)]

#removing left blank in treatment_duration
sampled_vl_df = sampled_vl_df[sampled_vl_df['treatment_duration']!="Left Blank"]
sampled_vl_df = sampled_vl_df[sampled_vl_df['Indication_for_VL_Testing']!="Left Blank"]
sampled_vl_df = sampled_vl_df[sampled_vl_df['gender']!="L"]

#removing undefined current_regiment
elements_to_check = ["Left Blank","."]
sampled_vl_df = sampled_vl_df[~sampled_vl_df['current_regimen'].isin(elements_to_check)]

def map_regimen_line(regimen):
    if regimen.startswith(('1', '3', '4')):
        return 'First line'
    elif regimen.startswith(('2', '8', '5')):
        return 'Second line'
    elif regimen.startswith(('6', '7', '9')):
        return 'Third line'
    else:
        return 'Unknown line'


sampled_vl_df['regimen_line'] = sampled_vl_df['current_regimen'].apply(map_regimen_line)

sampled_vl_df = sampled_vl_df[['gender',' age','treatment_duration','regimen_line','Indication_for_VL_Testing','arv_adherence','suppressed']]

#removing left blank in treatment_duration
sampled_vl_df = sampled_vl_df[sampled_vl_df['Indication_for_VL_Testing']!="Left Blank"]

#corr = sampled_vl_df 
label_encoder = LabelEncoder()
categorical_columns = [col for col in sampled_vl_df.columns if col not in ['age', 'suppressed']]
for col in categorical_columns:
    sampled_vl_df[col] = label_encoder.fit_transform(sampled_vl_df[col])


X = sampled_vl_df.drop(columns=['suppressed'])
y = sampled_vl_df['suppressed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "criterion":("gini", "entropy"), 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 15))), 
    "min_samples_split":[2, 3, 4, 6, 8,10,12], 
    "min_samples_leaf":list(range(1, 25)), 
}

clf_tree = DecisionTreeClassifier(random_state=42)
tree_cv = GridSearchCV(clf_tree, param_grid,scoring="accuracy",
                       n_jobs=-1, verbose=1, cv=10)

tree_cv.fit(X_train, y_train)

#model = DecisionTreeClassifier(random_state=42 ,max_depth=6)
#model.fit(X_train,y_train)


# saving the model to the BentoML local model store
saved_model = bentoml.sklearn.save_model("VL_Model",tree_cv)
print(f"Saved Model: {saved_model}")

#tag="vl_model:wxa2i4c52c526cqz"

#bentoml models list