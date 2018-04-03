import numpy as np
import pandas as pd
from math import log
from pgmpy.models import BayesianModel

#subtracting features of two images
def sub_features(image1, image2):
    image3 = abs(np.subtract(image1, image2))
    return image3;

#Joint Probability clculation for Similar data
def cal_prob():
    cal1 = f1_cpd[image3[0]][image3[3]*vcard[6] + image3[6]]
    cal2 = f2_cpd[image3[1]][image3[0]*vcard[5] + image3[5]]
    cal3 = f3_cpd[image3[2]][image3[5]]
    cal4 = f4_cpd[image3[3]][image3[2]*vcard[5] + image3[5]]
    cal5 = f5_cpd[image3[4]][image3[1]*vcard[7] +image3[7]]
    cal6 = f6_cpd[image3[5]]
    cal7 = f7_cpd[image3[6]][image3[3]*vcard[7] + image3[7]]
    cal8 = f8_cpd[image3[7]][image3[2]]
    cal9 = f9_cpd[image3[8]][image3[4]*vcard[6] + image3[6]]
    joint_prob = cal1*cal2*cal3*cal4*cal5*cal6*cal7*cal8*cal9
    return joint_prob

#Joint Probability clculation for Dissimilar data
def cal_prob2():
    cal1 = f1d_cpd[image3[0]][image3[3]*vcard_d[6] + image3[6]]
    cal2 = f2d_cpd[image3[1]][image3[0]*vcard_d[5] + image3[5]]
    cal3 = f3d_cpd[image3[2]][image3[5]]
    cal4 = f4d_cpd[image3[3]][image3[2]*vcard_d[5] + image3[5]]
    cal5 = f5d_cpd[image3[4]][image3[1]*vcard_d[7] + image3[7]]
    cal6 = f6d_cpd[image3[5]]
    cal7 = f7d_cpd[image3[6]][image3[3]*vcard_d[7] + image3[7]]
    cal8 = f8d_cpd[image3[7]][image3[2]]
    cal9 = f9d_cpd[image3[8]][image3[4]*vcard_d[6] + image3[6]]
    joint_probd = cal1*cal2*cal3*cal4*cal5*cal6*cal7*cal8*cal9
    return joint_probd

#Checking whether similar or not
def check_similarity():
    if cal_prob()>cal_prob2():
        return 1;
    else:
        return 0;

#Calculating Log-likelihood Ratio
def get_llr():
    prob1 = cal_prob()
    if prob1 !=0:
        prob1= log(prob1)
    prob2 = cal_prob2()
    if prob2!=0:
        prob2= log(prob2)
    return (prob1 - prob2)

#read training data from csv file
df = pd.read_csv("AND_Features.csv")
df = df.sort_values(by=['ImageId']) 
df1 = df.copy()

#read testing data from csv file
test_input = pd.read_csv("PGMTestData.csv")

#read test pairs data from csv file
test_pairs = pd.read_csv("PGMTestPairs.csv")

#column names for dataframe for storing similar and dissimilar data
columns = ('f1','f2','f3','f4','f5','f6','f7','f8','f9')

#column names for dataframe for storing test data output
columns_test =("FirstImage","SecondImage","LLR","SameOrDifferent")

#dataframe to store similar data 
df2 = pd.DataFrame(columns=columns)

#dataframe to store dissimilar data 
df3 = pd.DataFrame(columns=columns)

#dataframe for test data output 
test_output= pd.DataFrame(columns=columns_test)

#creating similar data 
k=0
n = len(df['ImageId'])
for i in range(n):
    if i <= len(df['f1'])-4:
        for j in range(i,i+4):
            if df.ImageId.str[:4].iloc[i] == df1.ImageId.str[:4].iloc[j]:
                f1=abs(df.f1.iloc[i] - df1.f1.iloc[j])
                f2=abs(df.f2.iloc[i] - df1.f2.iloc[j])
                f3=abs(df.f3.iloc[i] - df1.f3.iloc[j])
                f4=abs(df.f4.iloc[i] - df1.f4.iloc[j])
                f5=abs(df.f5.iloc[i] - df1.f5.iloc[j])
                f6=abs(df.f6.iloc[i] - df1.f6.iloc[j])
                f7=abs(df.f7.iloc[i] - df1.f7.iloc[j])
                f8=abs(df.f8.iloc[i] - df1.f8.iloc[j])
                f9=abs(df.f9.iloc[i] - df1.f9.iloc[j])
                df2.loc[k] = [f1,f2, f3, f4, f5, f6, f7, f8, f9]
                k+=1
           
#creating dissimilar data             
k=0
for i in range(n):
    if i <= len(df['f1'])-10:
        for j in range(i+3,i+10):
            if df.ImageId.str[:4].iloc[i] != df1.ImageId.str[:4].iloc[j]:
                f1=abs(df.f1.iloc[i] - df1.f1.iloc[j])
                f2=abs(df.f2.iloc[i] - df1.f2.iloc[j])
                f3=abs(df.f3.iloc[i] - df1.f3.iloc[j])
                f4=abs(df.f4.iloc[i] - df1.f4.iloc[j])
                f5=abs(df.f5.iloc[i] - df1.f5.iloc[j])
                f6=abs(df.f6.iloc[i] - df1.f6.iloc[j])
                f7=abs(df.f7.iloc[i] - df1.f7.iloc[j])
                f8=abs(df.f8.iloc[i] - df1.f8.iloc[j])
                f9=abs(df.f9.iloc[i] - df1.f9.iloc[j])
                df3.loc[k] = [f1,f2, f3, f4, f5, f6, f7, f8, f9]
                k+=1

#bayesian model creation for similar data  
and_model = BayesianModel([('f1', 'f2'),
                           ('f2', 'f5'),
                           ('f3', 'f4'),
                           ('f3', 'f8'),
                           ('f4', 'f1'),
                           ('f4', 'f7'),
                           ('f5', 'f9'),
                           ('f6', 'f2'),
                           ('f6', 'f3'),
                           ('f6', 'f4'),
                           ('f7', 'f1'),
                           ('f7', 'f9'),
                           ('f8', 'f5'),
                           ('f8', 'f7'),
                           ])
#taking 80 percent data as training data
nlen = (int(0.8*len(df2)))
train=df2[:nlen]
and_model.fit(train)
a=and_model.get_cpds()
and_model.check_model()
f1_cpd = a[0].get_values()
f2_cpd = a[1].get_values()
f3_cpd = a[2].get_values()
f4_cpd = a[3].get_values()
f5_cpd = a[4].get_values()
f6_cpd = a[5].get_values()
f7_cpd = a[6].get_values()
f8_cpd = a[7].get_values()
f9_cpd = a[8].get_values()

#length of each variable 
vcard = []
for i in range(9):
    vcard.append(a[i].variable_card)

#bayesian model creation for dissimilar data  
and_model2 = BayesianModel([('f1', 'f2'),
                           ('f2', 'f5'),
                           ('f3', 'f4'),
                           ('f3', 'f8'),
                           ('f4', 'f1'),
                           ('f4', 'f7'),
                           ('f5', 'f9'),
                           ('f6', 'f2'),
                           ('f6', 'f3'),
                           ('f6', 'f4'),
                           ('f7', 'f1'),
                           ('f7', 'f9'),
                           ('f8', 'f5'),
                           ('f8', 'f7'),
                           ])
#taking 80 percent data as training data
nlen2 = (int(0.8*len(df3)))
train=df3[:nlen2]
and_model2.fit(train)
b=and_model2.get_cpds()
and_model2.check_model()
f1d_cpd = b[0].get_values()
f2d_cpd = b[1].get_values()
f3d_cpd = b[2].get_values()
f4d_cpd = b[3].get_values()
f5d_cpd = b[4].get_values()
f6d_cpd = b[5].get_values()
f7d_cpd = b[6].get_values()
f8d_cpd = b[7].get_values()
f9d_cpd = b[8].get_values()

#length of each variable 
vcard_d = []
for i in range(9):
    vcard_d.append(b[i].variable_card)

#checking accuracy for Similar testing data
s=0
for i in range(nlen,len(df2)):
    image3= df2.loc[i]
    s = s+check_similarity()
print("Accuracy with Similar testing data", s/(len(df2)-nlen))

#checking accuracy for Dissimilar testing data
w=0
for i in range(nlen2,len(df3)):
    image3= df3.loc[i]
    w = w+check_similarity()
print("Accuracy with Dissimilar Testing data", (len(df3)-nlen2-w)/(len(df3) - nlen2))
 
#checking accuracy for Dissimilar and Similar Combined testing data
df2['f10'] = 1
df3['f10'] = 0
df4= df2[nlen:len(df2)].append(df3[nlen2:len(df3)], ignore_index=True)

count=0
for i in range(len(df4)):
    image3 = df4.loc[i]
    if df4.f10.iloc[i]== check_similarity():
        count+=1
print("Accuracy with combined training data", count/len(df4))

#Determining same or different writer and calculating log-likelihood ratio
for i in range(len(test_pairs)):
    for j in range(len(test_input)):
        if test_pairs.FirstImage.str[:3].iloc[i] == test_input.ImageId.iloc[j]:
            break;
    for l in range(len(test_input)):
        if test_pairs.SecondImage.str[:3].iloc[i] == test_input.ImageId.iloc[l]:
            break;
    image1 = test_input.iloc[j,2:11]
    image2 = test_input.iloc[l,2:11]
    image3 = sub_features(image1, image2)
    s_d = check_similarity()
    llr = get_llr()
    test_output.loc[i] = [test_pairs.FirstImage.str[:3].iloc[i],test_pairs.SecondImage.str[:3].iloc[i],llr,s_d ]

#printing test output
print(test_output)    

#Writing to the csv file
test_output.to_csv("PGMTestOutput.csv", sep=',',encoding='utf-8')            