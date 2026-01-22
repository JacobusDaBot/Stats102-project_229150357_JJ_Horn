import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import seaborn as sb
import statistics as stts
import pandas as pd
from fitter import Fitter, get_common_distributions, get_distributions
import csv
import statsmodels.api as sm
import json
import math
from distfit import distfit

id=[]
occupation=[]
credit_rating=[]
number_of_late_payments=[]
debt_to_income_ratio=[]
credit_utilization_ratio=[]
alldata=dict

allModels=['Fair','Good','Poor']
alloccupations= [
"Engineer",
"Clerk",
"Manager",
"Nurse",
"Salesperson",
"Teacher"]

allcolors="red","green","blue","yellow","orange","black"
with open('credit_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)    
    
    for row in reader:
        id.append(float(row['id']))
        occupation.append(row['occupation'])
        credit_rating.append(row['credit_rating'])
        number_of_late_payments.append(float(row['number_of_late_payments']))
        debt_to_income_ratio.append(float(row['debt_to_income_ratio']))
        credit_utilization_ratio.append(float(row['credit_utilization_ratio']))

    
    def generateMeanStandardDev(thisList):
        
        tmean=stts.mean(thisList)
        tstdev=stts.stdev(thisList)
        tvar=stts.variance(thisList)
        tharmmean=stts.harmonic_mean(thisList)
        tskew= sc.skew(thisList,axis=0,bias=True)
        tkurt=sc.kurtosis(thisList,axis=0,bias=True)
        tgeomean=0
        if 0 in thisList:
            tgeomean=0
        else:
            tgeomean=stts.geometric_mean(thisList)
        #print(tmean,tstdev,tvar)
        return [tmean,tstdev,tvar,tharmmean,tgeomean,tskew,tkurt,]
        
    def eachModelthing(row,query):
        icount=1
        allModels=['Fair','Good','Poor']
        for x in allModels:
                #print(x)
                
                thisoccupation={'id':[],'occupation':[],'credit_rating':[],'number_of_late_payments':[],'debt_to_income_ratio':[],'credit_utilization_ratio':[],'sum_of_each_rating':{}}
                for thisid in id:            
                    thisid=int(thisid)-1
                    if  credit_rating[thisid]==x :
                        
                        thisoccupation['id'].append(1)         
                        thisoccupation['occupation'].append(occupation[thisid])
                        thisoccupation['credit_rating'].append(credit_rating[thisid])
                        thisoccupation['sum_of_each_rating']
                        thisoccupation['number_of_late_payments'].append(float(number_of_late_payments[thisid]))
                        thisoccupation['debt_to_income_ratio'].append(float(debt_to_income_ratio[thisid]))
                        thisoccupation['credit_utilization_ratio'].append(float(credit_utilization_ratio[thisid]))
                        
                        
                data = {
                "credit_utilization_ratio": thisoccupation['credit_utilization_ratio'],
                "number_of_late_payments": thisoccupation['number_of_late_payments'],
                "debt_to_income_ratio":thisoccupation['debt_to_income_ratio'],
                "occupation":thisoccupation['occupation'],
                "credit_rating":thisoccupation['credit_rating'],
                "id":thisoccupation['id']
                    }  
                df = pd.DataFrame(data)
                df.head()
                
                sb.kdeplot(data=df, x=query, multiple="stack",color=allcolors[row],label=x,ax=axes[icount,row]).legend(loc="upper right")
                icount+=1
    def ShowDistribution(collum_name,place):
        data = {
        "credit_utilization_ratio": credit_utilization_ratio,
        "number_of_late_payments": number_of_late_payments,
        "debt_to_income_ratio":debt_to_income_ratio,
        "occupation":occupation,
        "credit_rating":credit_rating,
        "id":id
            }  
        df = pd.DataFrame(data)
        df.head()
        thisfield = df[collum_name].values
        stuff=generateMeanStandardDev(data[collum_name])
        mean=stuff[0]
        stdev=stuff[1]
        vari=stuff[2]
        harmmean=stuff[3]
        geomean=stuff[4]
        linewidth=1
        gx=sb.kdeplot(data=df, x=collum_name, multiple="stack",ax=place,label=collum_name)
        gx.legend(loc="upper right")
        gx.set(xlabel=None)
        gx.set(ylabel=None)
        place.axvline(x = mean,  color="red"   , linewidth =linewidth)
        if harmmean !=0 :            
            place.axvline(x = harmmean,  color="black"   , linewidth =linewidth)
        if geomean !=0 :
            place.axvline(x = geomean,  color="white"   , linewidth =linewidth)
        place.axvline(x = mean-vari  ,color="green" ,linewidth =linewidth)
        
        place.axvline(x =mean+vari, color="green",linewidth =linewidth     )
        
        place.axvline(x =mean-stdev, color="yellow" ,  linewidth =linewidth  )
        place.axvline(x =mean+stdev, color="yellow" ,linewidth =linewidth    )
        jdataDict={
            "name":collum_name,
            "mean":stuff[0],
            "standarddev":stuff[1],
            "variance":stuff[2],
            "harmmean":stuff[3],
            "geomean":stuff[4],
            "skewness":stuff[5],
            "kurtosis":stuff[6]
        }
        return jdataDict
        
    def boxplots(thing1_name):
        data = {
        "credit_utilization_ratio": credit_utilization_ratio,
        "number_of_late_payments": number_of_late_payments,
        "debt_to_income_ratio":debt_to_income_ratio,
        "occupation":occupation,
        "credit_rating":credit_rating,
        "id":id
            }  
        df = pd.DataFrame(data)
        df.head()
        sb.boxplot(df,y=thing1_name,x="credit_rating",hue="occupation").legend( loc='best', borderaxespad=0, )
    def compare2things(thing1_name,thing2_name,df):
       
       
        regression=(sc.linregress(x=data[thing1_name],y=data[thing2_name]))
        #print(regression)
        b=regression[0]
        a=regression[1]
        point1=a+b*min(data[thing1_name])
        point2=a+b*max(data[thing1_name])
       
        pearsonC=sc.pearsonr(data[thing1_name],data[thing2_name])
        
        txtreturn="name1 "+thing1_name+"\nname2"+thing2_name+"\npearson_coefficient"+str(pearsonC[0])+"\npearson_p_value"+str(pearsonC[1])+          "\nregression_line"+f"y={a}+{b}*x"
        jdataDict={
            "name1":thing1_name,
            "name2":thing2_name,
            "pearson_coefficient":pearsonC[0],
            "pearson_p_value":pearsonC[1],
            "regression_line":f"y={a}+{b}*x"
        }
        return jdataDict
        #return txtreturn
    
        
        
        
    print(len(credit_utilization_ratio))
    print(len(number_of_late_payments))
    print(len(debt_to_income_ratio))

    print(len(occupation))

    print(len(credit_rating))

    print(len(id))
    data = {
        "credit_utilization_ratio": credit_utilization_ratio,
        "number_of_late_payments": number_of_late_payments,
        "debt_to_income_ratio":debt_to_income_ratio,
        "occupation":occupation,
        "credit_rating":credit_rating,
        "id":id
            }  
    df = pd.DataFrame(data)
    df.head()  
    
    '''
    incomedist=ShowDistribution("debt_to_income_ratio" )
    latedist=ShowDistribution("number_of_late_payments" )
    creditdist=ShowDistribution("credit_utilization_ratio")
    '''
    f, axes = plt.subplots(3)
    f. set_size_inches((8, 6))
    incomedist=ShowDistribution("debt_to_income_ratio",place=axes[0] )
    latedist=ShowDistribution("number_of_late_payments",place=axes[1] )
    creditdist=ShowDistribution("credit_utilization_ratio",place=axes[2] )
    plt.savefig("Distributions1.png")
    
    
    
    c1=compare2things("debt_to_income_ratio","number_of_late_payments",df)
    c2=compare2things("credit_utilization_ratio","debt_to_income_ratio",df)
    c3=compare2things("number_of_late_payments","credit_utilization_ratio",df)
  
    f = open("c1.txt", "w")
    f.write( json.dumps(c1,indent=4,))
    f.close()
    f = open("c2.txt", "w")
    f.write( json.dumps(c2,indent=4,))
    f.close()
    f = open("c3.txt", "w")
    f.write( json.dumps(c3,indent=4,))
    f.close()
    
    plt.tight_layout()
    f, axes = plt.subplots(1)    
    f. set_size_inches((10, 5))
    boxplots("debt_to_income_ratio")
    plt.savefig("BoxC.png")
    plt.tight_layout()
    f, axes = plt.subplots(1)    
    f. set_size_inches((10, 5))
    boxplots("credit_utilization_ratio")
    plt.savefig("BoxM.png")
    plt.tight_layout()
    f, axes = plt.subplots(1)    
    f. set_size_inches((10, 5))
    boxplots("number_of_late_payments")
    plt.savefig("BoxW.png")
    
    
    ########
    #plt.show()
    out={
        "Probability_Distributions":{
        "incomedist":incomedist,
        "latedist":latedist,
        "creditdist":creditdist}
    }
    jsout=json.dumps(out,indent=4,)
    #put json into a text file:
    txtComparisons=out["Probability_Distributions"]
    global txtout
    txtout=""
    for l in txtComparisons:
        
        for x in txtComparisons[l]:
            #print(x +"    \t"+str(txtComparisons[l][x]))
            num=len(x)
            strx=x+"="*(20-num)
            if type(txtComparisons[l][x])!=str:
                txtout+=(strx+str(round(float(txtComparisons[l][x]),4)))
            else: 
                txtout+=(strx +str(txtComparisons[l][x]))
            txtout+="\n"
      
        txtout+="\n"
      
    f = open("numbers.txt", "w")
    f.write(txtout)
    f.close()