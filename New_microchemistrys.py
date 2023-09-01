# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:53:59 2019

@author: hanni
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', None)

df = pd.read_csv("01_11_17_Otolith_B_2.csv", error_bad_lines=False)  # read file
print(df)
# separate signals from backgound
start_row = 0  # To store starting row
end_row = 0  # To store ending row
last_row = df.index[-1]
finding_start = True  # Indicate whether finding starting or ending row
finding_end = True

# Check every row starting from row 4
for r in df['43Ca'].index[4:]:
    # print(r)
    # We are finding starting row
    if finding_start:
        if (df.loc[r, '43Ca'] > (4 * (df.loc[0:r - 1, '43Ca'].max()))):
            start_row = r - 1  # Update the solution
            finding_start = False  # Indicate that we're finding ending row now
           # print(df.loc[r, '43Ca'], (5 * (df.loc[0:r - 1, '43Ca'].max())))
    else:
        # Check for the first row since the starting row that is less than 10 times compared to the previous 4 rows
        # if df.loc[r,'43Ca'] != 0 and df.loc[r-4,'43Ca']/df.loc[r,'43Ca'] > 10:
        if (df.loc[r, '43Ca'] == 0 and finding_end == True):
            end_row = r  # Update the solution
            finding_end = False
            # finding_start = True#print(l)


def filter_noise(df, start_row, end_row, last_row, colname):
    from sklearn.linear_model import LinearRegression  
    
    df1 = df.iloc[np.r_[0:start_row, end_row + 1:last_row + 1]]
    #To calculate the range I am starting from end_row+1, so I am starting from row 272, that should be okay, no?
    
    noise_time = []
    noise_i = []
    loop = True
    
    #The loop will iterately remove the peaks and re-calcualte the 3-std to find the new peaks
    while loop is True :
        mean = df1[colname].mean()
        std = df1[colname].std()
        noise_i = df1.loc[df1[colname] > (mean + 3 * std), 'Time'].tolist()
        print("mean", mean, "std", std)
        df1 = df1[~df1['Time'].isin(noise_i)]
        noise_time = noise_time + noise_i

        #If no peak is found than no need to iterate
        if len(noise_i) == 0:
            loop = False
    #print(colname, "Noises", noise_time, "\n")
    
    return noise_time

start_row = start_row - 1
#print("check",start_row)
noise_43Ca = filter_noise(df, start_row, end_row, last_row, '43Ca')
#print(noise_43Ca)
noise_11B = filter_noise(df, start_row, end_row, last_row, '11B')
#print(noise_11B)
noise_25Mg = filter_noise(df, start_row, end_row, last_row, '25Mg')
#print(noise_25Mg)
noise_31P = filter_noise(df, start_row, end_row, last_row, '31P')
noise_34S = filter_noise(df, start_row, end_row, last_row, '34S')
noise_55Mn = filter_noise(df, start_row, end_row, last_row, '55Mn')
noise_57Fe = filter_noise(df, start_row, end_row, last_row, '57Fe')
noise_63Cu = filter_noise(df, start_row, end_row, last_row, '63Cu')
noise_66Zn = filter_noise(df, start_row, end_row, last_row, '66Zn')
noise_85Rb = filter_noise(df, start_row, end_row, last_row, '85Rb')
noise_88Sr = filter_noise(df, start_row, end_row, last_row, '88Sr')
noise_138Ba = filter_noise(df, start_row, end_row, last_row, '138Ba')
noise_208Pb = filter_noise(df, start_row, end_row, last_row, '208Pb')
noise_232Th = filter_noise(df, start_row, end_row, last_row, '232Th')
noise_238U = filter_noise(df, start_row, end_row, last_row, '238U')

all_filter = noise_43Ca + noise_11B + noise_25Mg + noise_31P + noise_34S + noise_55Mn + noise_57Fe + noise_63Cu + noise_66Zn + noise_85Rb + noise_88Sr + noise_138Ba + noise_208Pb + noise_232Th + noise_238U

df_filtered = df[~df['Time'].isin(all_filter)]


# Area under curve for all columns

from sklearn.linear_model import BayesianRidge, LinearRegression  

import pandas as pd
import statsmodels.formula.api as sm

def regression(df_exnoise,df, colname):
    df_exnoise1 = df_exnoise[['Time',colname]]
    df_exnoise1.columns = ['x', 'y']
    result = sm.ols(formula="y ~ x", data=df_exnoise1).fit()
    print(result.params)
    print(result.summary())

    c_peak=df.iloc[start_row:end_row]

    A_peak=c_peak[['Time',colname]]
    A_peak.columns = ['x', 'y']
    print(A_peak.columns)
    A_peak['pred']=A_peak['x'].apply(lambda x: result.params['Intercept']+x * result.params['x'] )
    print(A_peak['pred'])
    A_peak['diff']=A_peak['y']- A_peak['pred']
    f_area =A_peak['diff'].sum()

    return f_area




df1 = df.iloc[np.r_[0:start_row, end_row + 1:last_row + 1]]
df_exnoise = df1[~df1['Time'].isin(all_filter)]

area_43Ca =  regression(df_exnoise,df, '43Ca')
area_11B =  regression(df_exnoise,df, '11B')
area_25Mg =  regression(df_exnoise,df, '25Mg')
area_31P = regression(df_exnoise, df, '31P')
area_34S = regression(df_exnoise, df, '34S')
area_55Mn = regression(df_exnoise, df, '55Mn')
area_57Fe = regression(df_exnoise, df, '57Fe')
area_63Cu = regression(df_exnoise, df, '63Cu')
area_66Zn = regression(df_exnoise, df, '66Zn')
area_85Rb = regression(df_exnoise, df, '85Rb')
area_88Sr = regression(df_exnoise, df, '88Sr')
area_138Ba = regression(df_exnoise, df, '138Ba')
area_208Pb = regression(df_exnoise, df, '208Pb')
area_232Th = regression(df_exnoise, df, '232Th')
area_238U = regression(df_exnoise, df, '238U')


data = {'25Mg' : area_25Mg,'31P' : area_31P,'34S' : area_34S,'43Ca' : area_43Ca,'55Mn' : area_55Mn, '57Fe' : area_57Fe, '63Cu' : area_63Cu, '88Sr' : area_88Sr,'138Ba' : area_138Ba, '208Pb' : area_208Pb, '238U' : area_238U,'11B' : area_11B, '66Zn' : area_66Zn, '85Rb' : area_85Rb, '232Th' : area_232Th}
Area_df =pd.DataFrame(data, index=[0])
#print(Area_df)

#Area_df.to_excel (r'C:\Users\hanni\OneDrive\Documents\lopez_lab\new_data_proc\raw_data\export_dataframe.xlsx', index = None, header=True)