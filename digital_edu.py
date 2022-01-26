#создай здесь свой индивидуальный проект!
import pandas as pd
df = pd.read_csv('train.csv')


df.drop(['id','bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'relation', 'life_main', 'people_main', 'city', 'last_seen', 'career_start', 'career_end', 'occupation_name'], axis=1, inplace=True)
def sex_apply(sex):
    if sex==2:
        return 0 
    return 1
df['sex']=df['sex'].apply(sex_apply)
df['education_form'].fillna('Full-time', inplace=True)
df[list(pd.get_dummies(df['education_form']).columns)]=pd.get_dummies(df['education_form'])
df.drop(['education_form'],axis=1,inplace=True)
def edu_status_apply(edu_status):
    if edu_status=='Undergraduate applicant':
        return 0
    elif edu_status=='Student (Specialist)' or edu_status=="Student (Bachelor's)" or edu_status=="Student (Master's)":
        return 1
    elif edu_status=="Alumnus (Master's)" or edu_status=="Alumnus (Specialist)" or edu_status=="Alumnus (Bachelor's)":
        return 2
    elif edu_status=="PhD" or edu_status=='Candidate of Sciences':
       return 3 
df['education_status']=df['education_status'].apply(edu_status_apply)



def langs_apply(lang):
    if lang.find('English') !=-1:
        return 1
    else:
        return 0
df['langs']=df['langs'].apply(langs_apply)



def occupation_type_apply(occupation_type):
    if occupation_type=='university':
        return 0 
    else:
        return 1
df['occupation_type']=df['occupation_type'].apply(occupation_type_apply)
df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

X = df.drop('result', axis=1)
Y = df['result']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25)
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)

Y_pred=classifier.predict(X_test)
print('процент правильно предсказанных исходов:',round(accuracy_score(Y_test, Y_pred), 2) * 100)

    
