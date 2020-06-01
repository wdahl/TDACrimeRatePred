# INCIDENT_NUMBER
# OFFENSE_CODE
# OFFENSE_CODE_GROUP
# OFFENSE_DESCRIPTION
# DISTRICT
# REPORTING_AREA
# SHOOTING
# OCCURRED_ON_DATE
# YEAR
# MONTH
# DAY_OF_WEEK
# HOUR
# UCR_PART
# STREET
# Lat
# Long
# Location
import numpy as np
import time
from ripser import ripser
from persim import plot_diagrams
import numpy as np
from ripser import Rips
import pandas as pd
import matplotlib.pyplot as plt

times=[]
number_crimes=[]
top_num=10
dim1_col_list=[]
dim2_col_list=[]
top_dim_1=[]
top_dim_2=[]

for x in range(0,top_num):
    dim1_col_list.append("dim1_value"+str(x))
    dim2_col_list.append("dim2_value"+str(x))
    top_dim_1.append(0)
    top_dim_1.append(0)



# print(dim1_col_list)


# data = np.random.random((100,2))
# print(data)
#diagrams = ripser(data)['dgms']

# diagrams = rips.fit_transform(data)
# rips.plot(diagrams)

#str = unicode(str, errors='replace')


data= pd.read_csv('crime.csv', engine='python')
#print(data)
data_top = data.head()
#print(data_top)

out_data = pd.DataFrame(columns=['District', 'num_crime',  "num_dim1", *dim1_col_list, "num_dim2",*dim2_col_list])
#print(out_data)

#print((data['OCCURRED_ON_DATE']))
#

data['OCCURRED_ON_DATE'], data['B'] = data['OCCURRED_ON_DATE'].str.split(' ', 1).str
#print((data['OCCURRED_ON_DATE']))
data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'],format='%Y-%m-%d')


data=data.sort_values(by='OCCURRED_ON_DATE')

districts = data['DISTRICT'].unique()
s = pd.Series(districts)
s.to_csv("districts.CSV")

train_data, test_data  = np.array_split(data, 2)

#print(train_data)
#print(test_data )
# print(train_data['OCCURRED_ON_DATE'])
# mask = (data['OCCURRED_ON_DATE'] < '2018-1-1')
# train_data = data.loc[np.invert(mask)]


# print(mask)

df = train_data[['Lat','Long','DISTRICT','OCCURRED_ON_DATE']]
df=df.dropna()
districts = df['DISTRICT'].unique()
print(districts)

df.index = df.OCCURRED_ON_DATE

print(df)
startDate = df.index[0] #seed the while loop, format Timestamp
endDate=df.index[-1]



while (startDate >= df.index[0]) & (startDate < df.index[-1]):


    stopDate = (startDate + pd.offsets.DateOffset(months=1))#stopDate also Timestamp

    print(startDate)
    print(stopDate)

    dfMonth = df[(df['OCCURRED_ON_DATE'] > startDate) & (df['OCCURRED_ON_DATE'] <= stopDate)]


    print(dfMonth)


    for district in districts:
        start = time.time()
        top_dim_1 = [0] * top_num
        top_dim_2 = [0] * top_num

        disDf = dfMonth.loc[dfMonth['DISTRICT'] == district]

        if(len(disDf)<5):
            print(district)
            print(startDate)
            print(stopDate)
            print("empty")
        else:
            #df.replace([np.inf, -np.inf], 100)
            #print(df)
            #print(df[['Lat','Long']].values)
            location=disDf[['Lat','Long']].values

            diagrams = ripser(location)['dgms']

            #plot_diagrams(diagrams, show=True)

            #makes infi reasonable for regression
            diagrams=np.array(diagrams)
            diagrams[1][diagrams[1] >= 1E308] = 10
            diagrams[0][diagrams[0] >= 1E308] = 10



            # print(dim_1)
            # print(dim_1[:,0].reshape(1,len(dim_1[:,0])))
            #print(((dim_1[:,1]-dim_1[:,0]).reshape(1,len(dim_1[:,0]))).T)

            dim_1=diagrams[0]
            per=((dim_1[:,1]-dim_1[:,0]).reshape(1,len(dim_1[:,0])))
            sort_per=np.sort(per)
            num_dim1 = sort_per.size
            temp_top_dim_1=np.nan_to_num(sort_per[0,len(sort_per.T)-top_num:len(sort_per.T)])
            for i in range(0,len(temp_top_dim_1)):
                top_dim_1[i]=temp_top_dim_1[i]

           #print(top_dim_1)

            dim_2=diagrams[1]
            per=((dim_2[:,1]-dim_2[:,0]).reshape(1,len(dim_2[:,0])))
            sort_per=np.sort(per)
            num_dim2=sort_per.size
            temp_top_dim_2=np.nan_to_num(sort_per[0,len(sort_per.T)-top_num:len(sort_per.T)])
            for i in range(0,len(temp_top_dim_2)):
                top_dim_2[2]=temp_top_dim_2[i]




    # print(sort_per.size)
    # print(num_dim1)
    # print(len(sort_per))
    # print(top_dim_1)
    # print(top_dim_2)

            num_crime=len(disDf)
            print(num_crime)



            print(out_data)

            print(district)
            dat = [[district,num_crime, num_dim1,*top_dim_1 , num_dim2,*top_dim_2]]
            temp = pd.DataFrame(dat,columns=['District', 'num_crime',  "num_dim1", *dim1_col_list, "num_dim2",*dim2_col_list])

            print(temp)

            out_data=out_data.append(temp)
            print(out_data)
            end = time.time()
            times.append(end - start)
            number_crimes.append(len(disDf))



    print(startDate)
    print(df.index[-1])
    print(df.index[0])

    startDate = stopDate


    print((startDate >= df.index[0]) & (startDate < df.index[-1]))



data=data.sort_values(by='OCCURRED_ON_DATE')
out_data.to_csv("train_file.CSV")


dim1_col_list=[]
dim2_col_list=[]
top_dim_1=[]
top_dim_2=[]

for x in range(0,top_num):
    dim1_col_list.append("dim1_value"+str(x))
    dim2_col_list.append("dim2_value"+str(x))
    top_dim_1.append(0)
    top_dim_1.append(0)

out_data=[]
out_data = pd.DataFrame(columns=['District', 'num_crime',  "num_dim1", *dim1_col_list, "num_dim2",*dim2_col_list])


df = test_data[['Lat','Long','DISTRICT','OCCURRED_ON_DATE']]
df=df.dropna()
districts = df['DISTRICT'].unique()
print(districts)

df.index = df.OCCURRED_ON_DATE

print(df)
startDate = df.index[0] #seed the while loop, format Timestamp
endDate=df.index[-1]

while (startDate >= df.index[0]) & (startDate < df.index[-1]):

    stopDate = (startDate + pd.offsets.DateOffset(months=1))#stopDate also Timestamp

    print(startDate)
    print(stopDate)

    dfMonth = df[(df['OCCURRED_ON_DATE'] > startDate) & (df['OCCURRED_ON_DATE'] <= stopDate)]


    print(dfMonth)


    for district in districts:
        start = time.time()
        top_dim_1 = [0] * top_num
        top_dim_2 = [0] * top_num

        disDf = dfMonth.loc[dfMonth['DISTRICT'] == district]

        if(len(disDf)<5):
            print(district)
            print(startDate)
            print(stopDate)
            print("empty")
        else:
            #df.replace([np.inf, -np.inf], 100)
            #print(df)
            #print(df[['Lat','Long']].values)
            location=disDf[['Lat','Long']].values

            diagrams = ripser(location)['dgms']

            #plot_diagrams(diagrams, show=True)

            #makes infi reasonable for regression
            diagrams=np.array(diagrams)
            diagrams[1][diagrams[1] >= 1E308] = 10
            diagrams[0][diagrams[0] >= 1E308] = 10



            # print(dim_1)
            # print(dim_1[:,0].reshape(1,len(dim_1[:,0])))
            #print(((dim_1[:,1]-dim_1[:,0]).reshape(1,len(dim_1[:,0]))).T)

            dim_1=diagrams[0]
            per=((dim_1[:,1]-dim_1[:,0]).reshape(1,len(dim_1[:,0])))
            sort_per=np.sort(per)
            num_dim1 = sort_per.size
            temp_top_dim_1=np.nan_to_num(sort_per[0,len(sort_per.T)-top_num:len(sort_per.T)])
            for i in range(0,len(temp_top_dim_1)):
                top_dim_1[i]=temp_top_dim_1[i]

           #print(top_dim_1)

            dim_2=diagrams[1]
            per=((dim_2[:,1]-dim_2[:,0]).reshape(1,len(dim_2[:,0])))
            sort_per=np.sort(per)
            num_dim2=sort_per.size
            temp_top_dim_2=np.nan_to_num(sort_per[0,len(sort_per.T)-top_num:len(sort_per.T)])
            for i in range(0,len(temp_top_dim_2)):
                top_dim_2[2]=temp_top_dim_2[i]




    # print(sort_per.size)
    # print(num_dim1)
    # print(len(sort_per))
    # print(top_dim_1)
    # print(top_dim_2)

            num_crime=len(disDf)
            print(num_crime)



            print(out_data)

            print(district)
            dat = [[district,num_crime, num_dim1,*top_dim_1 , num_dim2,*top_dim_2]]
            temp = pd.DataFrame(dat,columns=['District', 'num_crime',  "num_dim1", *dim1_col_list, "num_dim2",*dim2_col_list])

            print(temp)

            out_data=out_data.append(temp)
            print(out_data)
            end = time.time()
            times.append(end - start)
            number_crimes.append(len(disDf))

    print(startDate)
    print(df.index[-1])
    print(df.index[0])
    startDate = stopDate


    print((startDate >= df.index[0]) & (startDate < df.index[-1]))

data=data.sort_values(by='OCCURRED_ON_DATE')
out_data.to_csv("test_file.CSV")


print(number_crimes)
print(times)
plt.plot(number_crimes,times,'ro')

plt.xlabel('Number of data points')
plt.ylabel('run time in seconds')
plt.show()