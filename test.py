import pandas as pd 
from multiGroupTest import multiGroupTest
#%%
df = pd.read_hdf('/home/bgeurten/PyProjects/dallas-dlc-seperate-multi-animal-analysis/BjoernDataMedianCILongInd4BoxPlotBest.h5')

df2 = df.loc[df['dataType']=='psd preStim']
df2 = df2.loc[df2['species']=='Medauroidea extradentata']
df2 = df2.loc[df2['adult'] == True]
df2 = df2.loc[df2['water'] == True]

idList = list()

for i,row in df2.iterrows():
    if row['sex'] == 'female':
        idStr = 'f'
    else:
        idStr = 'm'
    if row['light']:
        idStr += 'l'
    else:
        idStr += 'd'
    if row['wind']:
        idStr += 'w'
    else:
        idStr += 's'
    idList.append(idStr)

x = multiGroupTest(list(df2['power spectral density']),idList,'Fisher:medianDiff')

res = x.main()

res.to_csv('/media/gwdg-backup/BackUp/Bjoern/statsMedianDiff_FDR_BH.csv')

#%%
idList = list()
df2 = df.loc[df['dataType']=='psd preStim']
df2 = df2.loc[df2['species']=='Medauroidea extradentata']
df2 = df2.loc[df2['adult'] == True]
df2 = df2.loc[df2['water'] == True]
df2 = df2.loc[df2['sex'] == 'male']

for i,row in df2.iterrows():
    if row['movement direction'] == 'vertical':
        idStr = 'v'
    else:
        idStr = 'h'
    if row['light']:
        idStr += 'l'
    else:
        idStr += 'd'
    if row['wind']:
        idStr += 'w'
    else:
        idStr += 's'
    idList.append(idStr)

x = multiGroupTest(list(df2['power spectral density']),idList,'Fisher:medianDiff')

res = x.main()

res.to_csv('/media/gwdg-backup/BackUp/Bjoern/statsMedianDiff_maleDirection_FDR_BH.csv')
#%%
idList = list()
df2 = df.loc[df['dataType']=='psd preStim']
df2 = df2.loc[df2['species']=='Medauroidea extradentata']
df2 = df2.loc[df2['adult'] == True]
df2 = df2.loc[df2['water'] == True]
df2 = df2.loc[df2['sex'] == 'female']

for i,row in df2.iterrows():
    if row['movement direction'] == 'vertical':
        idStr = 'v'
    else:
        idStr = 'h'
    if row['light']:
        idStr += 'l'
    else:
        idStr += 'd'
    if row['wind']:
        idStr += 'w'
    else:
        idStr += 's'
    idList.append(idStr)

x = multiGroupTest(list(df2['power spectral density']),idList,'Fisher:medianDiff')

res = x.main()

res.to_csv('/media/gwdg-backup/BackUp/Bjoern/statsMedianDiff_maleDirection_FDR_BH.csv')