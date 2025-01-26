import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import mutual_info_score

import os

#load raw dataset
data = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\Data_Science_Salaries.csv'
df = pd.read_csv(data)
df1 = df
df1=df1.drop(['ID'], axis=1)


#drop duplicates, keeping only one of them

df1=df1.drop_duplicates(keep='first')
dfr=df1
print(dfr.shape)

#plot scatter original data
df1["id"] = df1.index
plt.scatter(df1['id'], df1['salary_in_usd'])
plt.show()


#separate the target feature y from the rest X and analyse y
y=dfr['salary_in_usd']
print ("Average: " + str(np.average(y)))
print ("Standard deviation: " + str(np.std(y)))
print("Skewness: " + str(skew(y, axis=0, bias=True)))
print("Kurtosis: " + str(kurtosis(y, axis=0, bias=True)))

#plot salary in USD distribution
plt.hist(y, color='lightgreen', ec='black', bins=15)
plt.show()

# add the salary category feature
lmsedge=np.percentile(y, 33)
mhsedge=np.percentile(y, 66)
print("33th percentile of arr : ", lmsedge)
print("66th percentile of arr : ", mhsedge)

# create a list of our conditions
def salary_group(value):
    if value < lmsedge:
        return "L"
    if lmsedge <= value < mhsedge:
        return "M"
    elif value >= mhsedge:
        return "H"

dfr['salary_group'] = dfr['salary_in_usd'].map(salary_group)
dfmi = dfr
print(dfr.shape)

# function to apply descaling to an array
def descaling (arry):
    miny = np.min(arry, axis=0)
    maxy = np.max(arry, axis=0)
    desarry = np.zeros(len(arry))
    for i in range (0,len(arry)):
       desarry [i] = (arry[i] -  miny) / (maxy - miny)

    return desarry

# function to apply standardization to an array
def standardization (arry):
    meany = np.mean(arry, axis=0)
    stdy = np.std(arry, axis=0)
    stdarry = np.zeros(len(arry))
    for i in range (0,len(arry)):
       stdarry [i] = (arry[i] -  meany) / stdy

    return stdarry

# adjust duplicate means
def make_unique(column):
    seen = set()
    for idx, value in enumerate(column):
        while value in seen:
            value += 1  # Increment by 1 if already seen
        seen.add(value)
        column[idx] = value
    return column

# generate 3 csv files, one for regression with the salary_in_usd values, one for classification
#with the salary_group and one for both with categorical features transformed to numerical ones 
dfc = dfr
dfc = dfc.drop(['salary_in_usd'], axis=1)
classificationfile = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\DS_salaries_classification.csv'
regressionfile = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\DS_salaries_regression.csv'
filewithnumerical = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\DS_salaries_regression_numerical.csv'
yearmappingfile = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\years.csv'
explevelemappingfile = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\experience_levels.csv'
empltypemapping = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\employment_types.csv'
jobtitlemapping = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\job_titles.csv'
emplresidencemapping = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\employee_residences.csv'
remoteratiomapping = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\remote_ratios.csv'
complocationmapping = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\companies_locations.csv'
compsizemapping = r'C:\Users\paolof\Documents\Personal\AI Master\Applied Machine learning\companies_sizes.csv'

dfr.to_csv(regressionfile)
dfc.to_csv(classificationfile)               
print("dfr: " + str(dfr.shape))
print("dfc: " + str(dfc.shape))
print (dfr.head())

# transform categorical features into numerical ones using target encoding applying also scaling
#year
yearv = dfr.groupby('work_year', as_index=False)['salary_in_usd'].mean()

yearv['salary_in_usd'] = make_unique(yearv['salary_in_usd'].tolist())

arry = yearv["salary_in_usd"].to_numpy()
desarray = descaling (arry)
yearv['des_year']=desarray
print (yearv)
yearv.to_csv(yearmappingfile)
# Create the mapping dictionary from DF1
yearmap = yearv.set_index('work_year')['des_year'].to_dict()
# Add the "des_year" column to DF2 based on the mapping
dfr['nd_work_year'] = dfr['work_year'].map(yearmap)


#experience_level
elv = dfr.groupby('experience_level', as_index=False)['salary_in_usd'].mean()

elv['salary_in_usd'] = make_unique(elv['salary_in_usd'].tolist())

arrel = elv["salary_in_usd"].to_numpy()
desarrel = descaling (arrel)
elv['des_el']=desarrel
print (elv)
elv.to_csv(explevelemappingfile)
elmap = elv.set_index('experience_level')['des_el'].to_dict()
dfr['nd_experience_level'] = dfr['experience_level'].map(elmap)


#employment_type
etv = dfr.groupby('employment_type', as_index=False)['salary_in_usd'].mean()

etv['salary_in_usd'] = make_unique(etv['salary_in_usd'].tolist())

arret = etv["salary_in_usd"].to_numpy()
desarret = descaling (arret)
etv['des_et']=desarret
print (etv)
etv.to_csv(empltypemapping)
etmap = etv.set_index('employment_type')['des_et'].to_dict()
dfr['nd_employment_type'] = dfr['employment_type'].map(etmap)

#job_title
jtv = dfr.groupby('job_title', as_index=False)['salary_in_usd'].mean()

jtv['salary_in_usd'] = make_unique(jtv['salary_in_usd'].tolist())

arrjt = jtv["salary_in_usd"].to_numpy()
desarrjt = descaling (arrjt)
jtv['des_jt']=desarrjt
print (jtv)
jtv.to_csv(jobtitlemapping)
jtmap = jtv.set_index('job_title')['des_jt'].to_dict()
dfr['nd_job_title'] = dfr['job_title'].map(jtmap)

#employee_residence
erv = dfr.groupby('employee_residence', as_index=False)['salary_in_usd'].mean()

erv['salary_in_usd'] = make_unique(erv['salary_in_usd'].tolist())

arrer = erv["salary_in_usd"].to_numpy()
desarrer = descaling (arrer)
erv['des_er']=desarrer
print (erv)
erv.to_csv(emplresidencemapping)
ermap = erv.set_index('employee_residence')['des_er'].to_dict()
dfr['nd_employee_residence'] = dfr['employee_residence'].map(ermap)

#remote_ratio
rrv = dfr.groupby('remote_ratio', as_index=False)['salary_in_usd'].mean()

rrv['salary_in_usd'] = make_unique(rrv['salary_in_usd'].tolist())

arrrr = rrv["salary_in_usd"].to_numpy()
desarrrr = descaling (arrrr)
rrv['des_rr']=desarrrr
print (rrv)
rrv.to_csv(remoteratiomapping)
rrmap = rrv.set_index('remote_ratio')['des_rr'].to_dict()
dfr['nd_remote_ratio'] = dfr['remote_ratio'].map(rrmap)

#company_location
clv = dfr.groupby('company_location', as_index=False)['salary_in_usd'].mean()

clv['salary_in_usd'] = make_unique(clv['salary_in_usd'].tolist())

arrcl = clv["salary_in_usd"].to_numpy()
desarrcl = descaling (arrcl)
clv['des_cl']=desarrcl
print (clv)
clv.to_csv(complocationmapping)
clmap = clv.set_index('company_location')['des_cl'].to_dict()
dfr['nd_company_location'] = dfr['company_location'].map(clmap)


#company_size
csv = dfr.groupby('company_size', as_index=False)['salary_in_usd'].mean()

csv['salary_in_usd'] = make_unique(csv['salary_in_usd'].tolist())

arrcs = csv["salary_in_usd"].to_numpy()
desarrcs = descaling (arrcs)
csv['des_cs']=desarrcs
print (csv)
csv.to_csv(compsizemapping)
csmap = csv.set_index('company_size')['des_cs'].to_dict()
dfr['nd_company_size'] = dfr['company_size'].map(csmap)

print (dfr.head())
# save the cleaned dataset with also the descaled numerical feature
dfr.to_csv(filewithnumerical)


#remove the salary_in_usd and salary_group features

x=dfr.drop(['salary_in_usd'], axis=1)
x=x.drop(['salary_group'], axis=1)
x=x.drop(x.columns[[0,1,2,3,4,5,6,7,8]], axis=1)

co=x.columns
nco=len(co)
print(co)




#Compute the mutual information for every pair of features
matrixmi= np.zeros([nco, nco])
i=0

for c1 in co:
    j=0
    for c2 in co:
        matrixmi[i][j]= round(mutual_info_score(x[c1], x[c2]),2)
        #print(c1, c2, i, j, matrixmi[i][j])
        j=j+1
    i=i+1

plt.imshow(matrixmi) 
plt.colorbar()
plt.title( "Heat-map of the Mutual Information of the features of the Data scientist job roles dataset" ) 
plt.show()


#Compute the mutual information for every feature and the target
vectmift= np.zeros([nco])
i=0

for c1 in co:
    vectmift[i] = round(mutual_info_score(x[c1], y),2)
    i=i+1

print (vectmift)



