import pandas as pd 
from multiGroupTest import multiGroupTest

#%%
file_pos_tra  = '/media/dataSSD/YegiTra/allAnaComb.h5'
file_pos_expo = '/media/gwdg-backup/BackUp/Yegi/all_expo_data.h5'


df_tra = pd.read_hdf(file_pos_tra)
x = multiGroupTest(list(df_tra['speed_max_mm/s']),df_tra['key'],'Fisher:medianDiff')
res = x.main()
res.to_csv('/media/dataSSD/YegiTra/stats/max_speed_stats.csv')

x = multiGroupTest(list(df_tra['activity']),df_tra['key'],'Fisher:medianDiff')
res = x.main()
res.to_csv('/media/dataSSD/YegiTra/stats/activity_stats.csv')

x = multiGroupTest(list(df_tra['radius_median_mm']),df_tra['key'],'Fisher:medianDiff')
res = x.main()
res.to_csv('/media/dataSSD/YegiTra/stats/radius_median_mm_stats.csv')

x = multiGroupTest(list(df_tra['speed_median_mm/s']),df_tra['key'],'Fisher:medianDiff')
res = x.main()
res.to_csv('/media/dataSSD/YegiTra/stats/median_speed_stats.csv')

df_expo = pd.read_hdf('/media/gwdg-backup/BackUp/Yegi/all_expo_data.h5')

idList = list()
for i,row in df_expo.iterrows():
    if row['sex'] == 'female':
        idStr = 'f'
    else:
        idStr = 'm'
    if row['infection']:
        idStr += 'i'
    else:
        idStr += 'h'
    if row['treatment']:
        idStr += 't'
    idList.append(idStr)

x = multiGroupTest(list(df_expo['rel. expo rate, mm2/h']),idList,'Fisher:medianDiff')
res = x.main()
res.to_csv('/media/dataSSD/YegiTra/stats/rel_exporate_stats.csv')

x = multiGroupTest(list(df_expo['exploredArea__mm2']),idList,'Fisher:medianDiff')
res = x.main()
res.to_csv('/media/dataSSD/YegiTra/stats/expoArea_mm2_stats.csv')


x = multiGroupTest(list(df_expo['abs. expo rate, mm2/h']),idList,'Fisher:medianDiff')
res = x.main()
res.to_csv('/media/dataSSD/YegiTra/stats/exporate_stats.csv')
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