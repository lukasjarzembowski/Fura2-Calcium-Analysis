import os
import pandas as pd
from scipy import stats
import dabest


def importrawdata(folderpath, runtime=None, dropcolumns=None):
    #clear target dataframe to really be empty
    targetdf = pd.DataFrame()
    files = os.listdir(folderpath)
    for file in files:
        filepath = folderpath + file
        openfile = pd.read_excel(filepath)

        if dropcolumns is not None:
            openfile = openfile.drop(columns=[dropcolumns])

        if runtime is not None:
            openfile = openfile.truncate(after=runtime, axis=0)

        openfile.columns = [str(cols) for cols in range(len(openfile.columns))]
        openfile = openfile.add_prefix(file.replace('xlsx',''))
        targetdf = pd.concat([targetdf, openfile], axis=1)

    print(len(files), "files have been successfully imported from", folderpath, "into a DataFrame with following shape (rows x columns):", targetdf.shape)
    return targetdf

def filterdata(inputdf, threshold=None):
    #based on https://stackoverflow.com/questions/36992046/pandas-dropping-columns-based-on-value-in-last-row
    initialmean = inputdf.iloc[1,:].mean(axis=0)
    initialsd = inputdf.iloc[1,:].std(axis=0)
    if threshold is None:
        threshold = initialmean + initialsd
        mask = inputdf.head(1).squeeze() < threshold
    if threshold is not None:
        mask = inputdf.head(1).squeeze() < threshold

    filtered = inputdf.loc[:,mask]

    lengthinput = len(inputdf.columns)
    lengthfiltered = len(filtered.columns)
    delta_len = lengthinput - lengthfiltered
    print('Initital Mean: ' + str(initialmean) + '. Initial SD: ' + str(initialsd))
    print('Threshold: ' + str(threshold))
    print('Dataframe was filtered')
    print(str(delta_len) + ' cells were removed')
    print('\n')

    return filtered

def measurementavgs(filtered_df, path):
    #get list of all filenames
    files = os.listdir(path)
    #create a new dataframe that stores all averaged data
    avg = pd.DataFrame()
    #for every file in raw folder search for
    #corresponding column in filtered data
    for file in files:
        #remove the .xlsx ending from filename to find the right column
        filename = file.replace('.xlsx','')
        #store mean of rows at each timepoint for all cells the measurement
        mean = filtered_df.filter(regex=filename).mean(axis=1)
        #attached the calculated mean dataframe to the avg dataframe
        avg = pd.concat([avg,mean], axis=1)


    #make a list of filenames without the .xlsx ending to change column names in avg
    cleanednames = []
    for file in files:
        file = str(file.replace(".xlsx", ""))
        cleanednames.append(file)
    avg.columns = cleanednames

    print('Averages single measurements succesfully calculated!')

    return avg

def calc_totalmean(inputdf, avgs=None):
    totalmeandf=pd.DataFrame()
    totalmeans = inputdf.mean(axis=1)
    totalsd = inputdf.std(axis=1)
    totalmeandf = pd.concat([totalmeans,totalsd], axis=1)
    totalmeandf.columns = ['total_mean','total_sd']
    if avgs is not None:
        avgofavgs = avgs.mean(axis=1)
        sdofavgs = avgs.std(axis=1)
        avgofavgs = pd.DataFrame(avgofavgs)
        sdofavgs = pd.DataFrame(sdofavgs)
        avgofavgs.columns = ['avgs_mean']
        sdofavgs.columns = ['avgs_sd']
        avgstats = pd.concat([avgofavgs,sdofavgs], axis=1)

        totalmeandf = pd.concat([totalmeandf,avgstats], axis=1)
        return totalmeandf

    else:
        return totalmeandf

    print('Means succesfully calculated!')

#change inputdf to a new df that gets returned
def calc_mean(inputdf,start,stop):
    inputdf = inputdf.iloc[start:stop,:].mean(axis=0)
    return inputdf

def calc_max(inputdf,start,stop):
    inputdf = inputdf.iloc[start:stop,:].max(axis=0)
    return inputdf

def calc_slope(inputdf,start,stop):
    diff = [start, stop] #start and stop of the frames to fit in between
    diff = diff[1] - diff[0] #differences = length of the frames where to fit

    x=[] #to do a line fit you need corresponding x values = the number of frames you want to fit your data
    for i in range(diff):
        x.append(i) #creates an array/list that contains as many numbers from 0 till n as needed for the fit as x axis
    slopes = []
    rsqrd = []
    influx = inputdf.iloc[start:stop,:]
    for column in influx.columns:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, influx[column])
        slopes.append(slope)
        r = r_value ** 2
        rsqrd.append(r)
    return slopes, rsqrd

def ca_analysis(filtereddf, parameters_dict, avgs=None):
    results = pd.DataFrame()
    tempdf = pd.DataFrame()
    if avgs is not None:
        for i in avgs:
            filtereddf = filtereddf.drop(columns=[i])

    for key in parameters_dict.keys():
        templist = parameters_dict[key]

        if len(templist) == 3:

            if templist[2] == 'mean':
                tempmeandf = pd.DataFrame()
                tempmeandf[key] = calc_mean(filtereddf,templist[0], templist[1])
                tempdf = pd.concat([tempdf,tempmeandf], axis=1)

            elif templist[2] == 'max':
                tempmaxdf = pd.DataFrame()
                tempmaxdf[key] = calc_max(filtereddf,templist[0],templist[1])
                tempdf = pd.concat([tempdf, tempmaxdf], axis=1)

            elif templist[2] == 'delta':
                tempdf[key] = tempdf[templist[0]] - tempdf[templist[1]]

            elif templist[2] == 'slope':
                slopes, rvalues = calc_slope(filtereddf,templist[0],templist[1])
                tempdf['slope'] = slopes
                tempdf['rsqrd'] = rvalues

            else:
                print('Please make sure your parameter ' + str(key) + ' ranges are marked with "mean", "max", "slope" or "delta"')


        else:
            print('Please make sure your parameter ' + str(key) + ' has following format: "parameter": [frame number start, frame number stop, "operation"] or: "parameter": [parameter b, parameter a, "delta"] when calculating deltas = b-a')

    print('Kinetics succesfully calculated!')

    return tempdf

def loaddata(data1, data2, parameter, name1, name2, rsmpls=None):
    temp = pd.concat([data1[parameter], data2[parameter]], axis=1, sort=True)
    temp.columns = [name1, name2]
    if rsmpls is not None:
        bootstrap = dabest.load(temp, idx=(name1, name2), resamples=rsmpls)
    else:
        bootstrap = dabest.load(temp, idx=(name1, name2))
    return bootstrap

def excelexport(datadict):
    outputname = datadict["filename"]
    writer = pd.ExcelWriter(outputname)
    keylist = list(datadict.keys())[1:]
    for key in keylist:
        dftowrite = datadict[key]
        dftowrite.to_excel(writer, sheet_name=key)
    writer.save()
    print(outputname, "with", len(datadict)-1, "sheets has been saved")

def normalize_data(inputdf):
    normalizeddf = inputdf.copy()
    column_list = list(normalizeddf)
    for column in column_list:
        first_value = normalizeddf[column].iloc[0]
        normalizeddf[column] = normalizeddf[column].apply(lambda x: x/first_value)
    return normalizeddf


def get_responders(inputdf, column_name, threshold = None):
# select all rows in a column that are higher than a set threshold as responders
# and those that are below that threshold as non_responders. If no threshold is passed,
# the standard deviation of all values will be used as lower threshold
# added 2019-01-07

    if threshold is None:
        std = inputdf[column_name].std()
        mask = inputdf[column_name].squeeze() > std
    if threshold is not None:
        mask = inputdf[column_name].squeeze() > threshold

    responders = inputdf[column_name].loc[mask]
    non_responders = inputdf[column_name].loc[~mask]
    return responders, non_responders
