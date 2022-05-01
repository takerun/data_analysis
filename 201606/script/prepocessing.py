# coding:utf-8
import numpy as np
import pandas as pd

### Constants
# Train
TRAIN_PREC = "../future-temparture-prediction/Precipitation_Train_Feature.tsv"
TRAIN_SUN = "../future-temparture-prediction/SunDuration_Train_Feature.tsv"
TRAIN_TEMPER = "../future-temparture-prediction/Temperature_Train_Feature.tsv"
# Test
TEST_PREC = "../future-temparture-prediction/Precipitation_Test_Feature.tsv"
TEST_SUN = "../future-temparture-prediction/SunDuration_Test_Feature.tsv"
TEST_TEMPER = "../future-temparture-prediction/Temperature_Test_Feature.tsv"
# 欠損値
nan = 0

# Function
def splitIdPerPlace(temper_data):
    placeId = dict()
    for i, df in temper_data.groupby("targetplaceid"):
        placeId[i] = df.index.values
    return placeId
### Class
# predict temperature from temper, sun, prec
class TemperatureData(object):
    """コンストラクタ"""
    def __init__(self, data):
        self.Precipitation = pd.DataFrame()
        self.SunDuration = pd.DataFrame()
        self.Temperature = pd.DataFrame()
        self.TemperPlaceId = list()
        # train data(DataFrame)
        if data == "train":
            self.Precipitation = pd.read_csv(TRAIN_PREC, sep='\t').fillna(nan)
            self.SunDuration = pd.read_csv(TRAIN_SUN, sep='\t').fillna(nan)
            self.Temperature = pd.read_csv(TRAIN_TEMPER, sep='\t').fillna(nan)
            self.TemperPlaceId = splitIdPerPlace(self.Temperature)

        # test data(DataFrame)
        if data == "test":
            self.Precipitation = pd.read_csv(TEST_PREC, sep='\t').fillna(nan)
            self.SunDuration = pd.read_csv(TEST_SUN, sep='\t').fillna(nan)
            self.Temperature = pd.read_csv(TEST_TEMPER, sep='\t').fillna(nan)
            self.TemperPlaceId = splitIdPerPlace(self.Temperature)

        # listed data(DataFrame)
        if isinstance(data,list):
            PREC, SUN, TEMPER = data
            self.Precipitation = pd.read_csv(PREC, sep='\t').fillna(nan)
            self.SunDuration = pd.read_csv(SUN, sep='\t').fillna(nan)
            self.Temperature = pd.read_csv(TEMPER, sep='\t').fillna(nan)
            self.TemperPlaceId = splitIdPerPlace(self.Temperature)

    """メソッド"""
    def temperVec(self):
        return self.Temperature.loc[:, ['place%d' % i for i in xrange(11)]].values

    def temperWVec(self):
        yearnum = 5
        placenum = len(self.TemperPlaceId)
        datasize_year = 360
        stk = np.array([])
        for n in xrange(yearnum):
            start = datasize_year*n
            end = datasize_year*(n+1) -1
            tempers_year = self.Temperature.loc[start:end,  ['place%d' % i for i in xrange(placenum)]].values
            for i in xrange(datasize_year-1):
                vec = np.r_[tempers_year[i],tempers_year[i+1]]
                stk = np.append(stk, vec)
            last = np.append(tempers_year[datasize_year-1],np.zeros((placenum)))
            stk = np.append(stk, last)
        result = stk.reshape(datasize_year*5, placenum*2)
        return result

    def temperWVecPlus(self):
        yearnum = 5
        placenum = len(self.TemperPlaceId)
        datasize_year = 360
        stk = np.array([])
        for n in xrange(yearnum):
            start = datasize_year*n
            end = datasize_year*(n+1) -1
            tempers_year = self.Temperature.loc[start:end,  ['place%d' % i for i in xrange(placenum)]].values
            for i in xrange(datasize_year-1):
                vec = np.r_[tempers_year[i],tempers_year[i+1], (tempers_year[i]+tempers_year[i+1])/2]
                stk = np.append(stk, vec)
            last = np.append(tempers_year[datasize_year-1],np.zeros((placenum*2)))
            stk = np.append(stk, last)
        result = stk.reshape(datasize_year*5, placenum*3)
        return result

    def temperWVecMean(self):
        yearnum = 5
        placenum = len(self.TemperPlaceId)
        datasize_year = 360
        stk = np.array([])
        for n in xrange(yearnum):
            start = datasize_year*n
            end = datasize_year*(n+1) -1
            tempers_year = self.Temperature.loc[start:end,  ['place%d' % i for i in xrange(placenum)]].values
            for i in xrange(datasize_year-1):
                vec = (tempers_year[i]+tempers_year[i+1])/2
                stk = np.append(stk, vec)
            last = tempers_year[datasize_year-1]
            stk = np.append(stk, last)
        result = stk.reshape(datasize_year*5, placenum)
        return result

    def temperVecPerPlace(self):
        rtn = dict()
        for i in xrange(len(self.TemperPlaceId)):
            splited_data = self.Temperature.ix[self.TemperPlaceId[i]]
            splited_data = splited_data.reset_index(drop=True)
            rtn[i] = splited_data.loc[:, ['place%d' % j for j in xrange(11)]].values
        return rtn

    def TSPVecPerPlace(self):
        rtn = dict()
        for i in xrange(len(self.TemperPlaceId)):
            splited_Tdata = self.Temperature.ix[self.TemperPlaceId[i]]
            splited_Sdata = self.SunDuration.ix[self.TemperPlaceId[i]]
            splited_Pdata = self.Precipitation.ix[self.TemperPlaceId[i]]
            splited_Tdata = splited_Tdata.reset_index(drop=True)
            splited_Sdata = splited_Sdata.reset_index(drop=True)
            splited_Pdata = splited_Pdata.reset_index(drop=True)
            rtn[i] = splited_Tdata.loc[:, ['place%d' % j for j in xrange(11)]].values
            rtn[i] = np.c_[rtn[i],splited_Sdata.loc[:, ['place%d' % j for j in xrange(11)]].values]
            rtn[i] = np.c_[rtn[i],splited_Pdata.loc[:, ['place%d' % j for j in xrange(11)]].values]
        return rtn
