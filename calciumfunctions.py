import os
import pandas as pd
from scipy import stats
import dabest
import dask.dataframe as dd

def importrawdata(folderpath, runtime=None, dropcolumns=None, folder=True, name=None):
    #declare an empty dataframe and list for filenames
    targetdf = pd.DataFrame()
    files = []


    if folder is False:
        if folderpath.endswith(".xlsx"):
            files.append(folderpath)
        else:
            print("File type is currently not supported (.xlsx only)")

    if folder is True:
        for entry in os.scandir(folderpath):
            if entry.name.endswith(".xlsx"):
                files.append(entry)
            else:
                pass

    for file in files:
        if folder is False:
            openfile = pd.read_excel(folderpath)
        elif folder is True:
            filepath = folderpath + file.name
            openfile = pd.read_excel(filepath)

        if dropcolumns is not None:
            openfile = openfile.drop(columns=[dropcolumns])

        if runtime is not None:
            openfile = openfile.truncate(after=runtime, axis=0)

        if folder is False:
            openfile.columns = [str(cols) for cols in range(len(openfile.columns))]
            openfile = openfile.add_prefix(file.replace('xlsx',''))

        if folder is True:
            openfile.columns = [str(cols) for cols in range(len(openfile.columns))]
            openfile = openfile.add_prefix(file.name.replace('xlsx',''))

        targetdf = pd.concat([targetdf, openfile], axis=1)

    if name is not None:
        targetdf.name = str(name)

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

    try:
        print('Dataframe:',  str(inputdf.name))
    except AttributeError:
        print('Dataframe is unnamed')
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
    totalSEM = inputdf.sem(axis=1)
    totalnumber = len(inputdf.columns)
    totalmeandf = pd.concat([totalmeans,totalsd,totalSEM], axis=1)
    totalmeandf.columns = ['total_mean','total_sd','total_SEM']
    totalmeandf['number of cells'] = ""
    totalmeandf.at[0,'number of cells']= totalnumber
    if avgs is not None:
        avgofavgs = avgs.mean(axis=1)
        sdofavgs = avgs.std(axis=1)
        semofavgs = avgs.sem(axis=1)
        avgofavgs = pd.DataFrame(avgofavgs)
        sdofavgs = pd.DataFrame(sdofavgs)
        semofavgs = pd.DataFrame(semofavgs)
        avgofavgs.columns = ['avgs_mean']
        sdofavgs.columns = ['avgs_sd']
        semofavgs.columns = ['avgs_sem']
        avgstats = pd.concat([avgofavgs,sdofavgs,semofavgs], axis=1)

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

def calc_auc(filtereddf, start, stop):
#calculate the area under the curve using the sklearn metrics.auc function
#for every column between the specified timepoints = rows
#added 2020-04-17, also added to ca_analysis function
    column_list = list(filtereddf)
    x = np.arange(0,stop-start)
    auc_list = []
    for i in range(0,len(filtereddf.columns)):
        auc = metrics.auc(x,filtereddf.iloc[start:stop,i])
        auc_list.append(auc)
    return auc_list

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

            elif templist[2] == 'auc':
                tempdf['auc'] = calc_auc(filtereddf, templist[0], templist[1])

            else:
                print('Please make sure your parameter ' + str(key) + ' ranges are marked with "mean", "max", "slope", "auc" or "delta"')

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
# added 2020-01-07
    std = inputdf[column_name].std()
    mean = inputdf[column_name].mean()

    print("Mean:", str(mean))
    print("SD:", str(std))
    if threshold is None:
        mask = inputdf[column_name].squeeze() > std
        print("The threshold used:", str(std))

    if threshold is not None:
        mask = inputdf[column_name].squeeze() > threshold
        print("The threshold used:", str(threshold))

    responders = inputdf[column_name].loc[mask]
    non_responders = inputdf[column_name].loc[~mask]

    print("Total number of cells", str(len(inputdf.index)))
    print("Number of responders:", str(len(responders.index)))
    print("Number of non-responders:", str(len(non_responders.index)))
    print("Percentage of responders:", str(len(responders.index)/len(inputdf.index)*100))
    print('\n')

    return responders, non_responders

def intensity_dataframe(timelapse_image,mask,name=None):
    from skimage import measure
    from skimage.measure import regionprops
    import numpy as np
    import pandas as pd

    #create empty dataframe
    intensity_df = pd.DataFrame()

    #count the number of unique ROIs and remove the background ROI 0
    rois = np.unique(mask).tolist()[1:]
    #create a list with numbered column name and use name argument as prefix if given
    if name is not None:
        col_list = [str(name)+'_roi_%d' % x for x in range(1,len(rois)+1)]
    else:
        col_list = ['roi_%s' %  x for x in range(1,len(rois)+1)]

    #go through every single frame of the timelapse image
    for i in range(len(timelapse_image)):
        #for each frame clear the list of intensities
        intensity_list = []
        #save all region props in a list for every ROI in this particular frame
        props = regionprops(mask, intensity_image=timelapse_image[i])

        #get the mean intensity of every prop and save it to the intensity list
        for j in range(len(props)):
            intensity = props[j].mean_intensity
            intensity_list.append(intensity)

        #for each frame save the list of intensities as new row in the dataframe
        intensity_df = intensity_df.append(pd.DataFrame([intensity_list]))

    print("calculation of intensites for %s frames completed" % str(len(timelapse_image)))

    #add the column names to the dataframe
    intensity_df.columns = col_list

    #reset the index and let it start from 1
    intensity_df.index = np.arange(1, len(intensity_df)+1)

    return intensity_df

def intensity_dataframe_long(timelapse_image,mask,measurement,name):
    from skimage import measure
    from skimage.measure import regionprops
    import numpy as np
    import pandas as pd

    #create empty dataframe
    intensity_df = pd.DataFrame()

    #go through every single frame of the timelapse image
    for i in range(len(timelapse_image)):
        #for each frame clear the list of intensities
        intensity_list = []
        #save all region props in a list for every ROI in this particular frame
        props = regionprops(mask, intensity_image=timelapse_image[i])

        #get the mean intensity of every prop and save it to the intensity list
        for j in range(len(props)):
            intensity = props[j].mean_intensity
            intensity_list.append(intensity)
        #
        #for each frame save the list of intensities as new row in the dataframe
        intensity_df = intensity_df.append(pd.DataFrame([intensity_list]))

    print("calculation of intensites for %s frames completed" % str(len(timelapse_image)))

    #add the column names to the dataframe
    intensity_df["timepoint"] = np.arange(len(timelapse_image))
    intensity_df["measurement"] = measurement
    #reset the index and let it start from 1
    intensity_df.index = np.arange(1, len(intensity_df)+1)
    intensity_df = pd.melt(intensity_df, id_vars=["timepoint", "measurement"], value_vars=intensity_df.columns[:-2], var_name="roi")
    intensity_df["roi"] += 1

    intensity_df.name = str(name)
    intensity_df["group"]= name
    return intensity_df

def filterdata_long(inputdf, threshold=None):
    #this function was implemented with help of Jose A. Jimenez
    #https://stackoverflow.com/questions/62957110/pandas-selecting-multiple-rows-based-on-column-pair/
    initialmean = inputdf.loc[inputdf["timepoint"] == 0].mean().array[-1]
    initialsd = inputdf.loc[inputdf["timepoint"] == 0].std().array[-1]

    if threshold is None:
        threshold = initialmean + initialsd
        pre_activated_t0 = inputdf[(inputdf['timepoint'] == 0) & (inputdf['value'] > threshold)]
    if threshold is not None:
        pre_activated_t0 = inputdf[(inputdf['timepoint'] == 0) & (inputdf['value'] > threshold)]

    pre_activated = inputdf.merge(pre_activated_t0[["measurement", "roi"]], how="inner", on=["measurement", "roi"])
    filtereddf = inputdf.merge(
    pre_activated,
    how="left",
    on=["timepoint", "measurement", "roi", "value"],
    )
    filtereddf = filtereddf[pd.isna(filtereddf["group_y"])]

    filtereddf.drop("group_y", axis=1, inplace=True)
    filtereddf.columns = list(inputdf.columns)

    length_input = len(inputdf[inputdf["timepoint"]==0])
    length_filtered = len(filtereddf[filtereddf["timepoint"]==0])
    delta = length_input - length_filtered

    try:
        print('Dataframe:',  str(inputdf.name))
    except AttributeError:
        print('Dataframe is unnamed')
    print('Initital Mean: ' + str(initialmean) + '. Initial SD: ' + str(initialsd))
    print('Threshold: ' + str(threshold))
    print('Dataframe was filtered')
    print('Total cells: ' + str(length_input))
    print(str(delta) + ' cells were removed')
    print('\n')

    return filtereddf, pre_activated



def calc_mean_melted(inputdf,start,stop):
    meandf = inputdf[(inputdf['timepoint'] >= start) & (inputdf['timepoint'] <= stop)].groupby(["roi","measurement"]).agg(["mean"]).drop(["timepoint"],axis = 1)
    meandf.columns = ["value"]
    return meandf

def calc_max_melted(inputdf,start,stop):
    maxdf = inputdf[(inputdf['timepoint'] >= start) & (inputdf['timepoint'] <= stop)].groupby(["roi","measurement"]).agg(["max"]).drop(["timepoint"],axis = 1)
    maxdf.columns = ["value","group"]
    return maxdf

def calc_slope_melted(inputdf,start,stop):
    from scipy.stats import linregress
    subsectiondf = inputdf[(inputdf['timepoint'] >= start) & (inputdf['timepoint'] <= stop)]
    slope = subsectiondf.groupby(["roi","measurement"]).apply(lambda v: linregress(v.timepoint, v.value)[0])
    rsqrd = subsectiondf.groupby(["roi","measurement"]).apply(lambda v: linregress(v.timepoint, v.value)[2])**2
    slopedf = pd.DataFrame(slope)
    rsqrddf = pd.DataFrame(rsqrd)
    slopedf.columns = ["value"]
    rsqrddf.columns = ["value"]
    return slopedf, rsqrddf

def calc_auc_melted(inputdf,start,stop):
    from sklearn import metrics
    subsectiondf = inputdf.loc[(inputdf['timepoint'] >= start) & (inputdf['timepoint'] <= stop)]
    return subsectiondf.groupby(["roi","measurement","group"]).apply(lambda v: metrics.auc(v.timepoint, v.value)).drop(["timepoint","group"],axis = 1)



def ca_analysis_long(filtereddf, parameters_dict,name=None):
    resultsdf = pd.DataFrame()
    result_dict={}

    for key in parameters_dict.keys():
        templist = parameters_dict[key]

        if len(templist) == 3:

            if templist[2] == 'mean':
                tempmeandf = calc_mean_melted(filtereddf,templist[0], templist[1])
                tempmeandf["parameter"]=key
                result_dict[key] = tempmeandf

            elif templist[2] == 'max':
                tempmaxdf = calc_max_melted(filtereddf,templist[0],templist[1])
                tempmaxdf["parameter"]=key
                result_dict[key] = tempmaxdf

            elif templist[2] == 'delta':
                delta = result_dict[templist[0]]["value"] - result_dict[templist[1]]["value"]
                delta = pd.DataFrame(delta)
                delta["parameter"] = key
                result_dict[key] = delta

            elif templist[2] == 'slope':
                slopes, rsqrd = calc_slope_melted(filtereddf,templist[0],templist[1])
                slopes["parameter"]=key
                rsqrd["parameter"]="r_squared"
                result_dict[key] = slopes
                result_dict[key+"_rsquared"] = rsqrd

            elif templist[2] == 'auc':
                tempaucdf = calc_auc_melted(filtereddf, templist[0], templist[1])
                tempaucdf["parameter"]=key
                result_dict[key] = tempaucdf
            else:
                print('Please make sure your parameter ' + str(key) + ' ranges are marked with "mean", "max", "slope", "auc" or "delta"')

        else:
            print('Please make sure your parameter ' + str(key) + ' has following format: "parameter": [frame number start, frame number stop, "operation"] or: "parameter": [parameter b, parameter a, "delta"] when calculating deltas = b-a')


    for key in result_dict.keys():
        resultsdf = pd.concat([resultsdf,result_dict[key]])

    if name is not None:
        resultsdf["group"]=name
        resultsdf["group_parameter"]=resultsdf.group.str.cat(resultsdf.parameter, sep="_")
        resultsdf.astype(dtype=object)

    resultsdf.reset_index(inplace=True)
    print('Kinetics succesfully calculated!')
    return resultsdf

def get_responders_long(inputdf, parameter, threshold=None, return_trace = False, trace_df = None):

    subset = inputdf[inputdf["parameter"]==parameter].copy()

    std = subset["value"].std()
    mean = subset["value"].mean()


    if threshold is None:
        threshold = mean + std

    print("Mean of parameter %s: %s" % (parameter, str(mean)))
    print("Standard deviation: %s" % (str(std)))
    print("The threshold used:", str(threshold))

    mask = subset["value"] > threshold
    subset["response"] = mask

    nr_responders = subset.response.value_counts().to_list()[0]
    nr_nonresponders = subset.response.value_counts().to_list()[1]
    totalcells = len(subset)
    print("Total number of cells", str(totalcells))
    print("Number of responders:", str(nr_responders))
    print("Number of non-responders:", str(nr_nonresponders))
    print("Percentage of responders:", str(nr_responders/totalcells*100))




    if return_trace == True and trace_df is not None:
        responder = subset[subset["response"]==True]
        non_responder = subset[subset["response"]==False]

        responder_trace = trace_df.merge(responder[["measurement", "roi"]], how="inner", on=["measurement", "roi"]).sort_values(by=["timepoint","roi"])
        non_responder_trace = trace_df.merge(non_responder[["measurement", "roi"]], how="inner", on=["measurement", "roi"]).sort_values(by=["timepoint","roi"])

        print("Responder and non-responder traces returned")
        print('\n')
        return subset,responder_trace, non_responder_trace

    elif return_trace == True and trace_df is None:
        print("Can only get traces if trace df is passed")
        print('\n')
        return subset

    else:
        print('\n')
        return subset



def filterdata_long_dask(inputdf, threshold=None, nr_of_partitions=None):
    #this function was implemented with help of Jose A. Jimenez
    #https://stackoverflow.com/questions/62957110/pandas-selecting-multiple-rows-based-on-column-pair/

    import dask.dataframe as dd

    initialmean = inputdf.loc[inputdf["timepoint"] == 0].mean().array[-1]
    initialsd = inputdf.loc[inputdf["timepoint"] == 0].std().array[-1]

    if threshold is None:
        threshold = initialmean + initialsd
        pre_activated_t0 = inputdf[(inputdf['timepoint'] == 0) & (inputdf['value'] > threshold)]
    if threshold is not None:
        pre_activated_t0 = inputdf[(inputdf['timepoint'] == 0) & (inputdf['value'] > threshold)]


    pre_activated = inputdf.merge(pre_activated_t0[["measurement", "roi"]], how="inner", on=["measurement", "roi"])

    if nr_of_partitions is None:
        nr_of_partitions = 30

    input_dd = dd.from_pandas(inputdf, npartitions=nr_of_partitions)
    preactivated_dd = dd.from_pandas(pre_activated, npartitions=nr_of_partitions)

    merger = dd.merge(input_dd,preactivated_dd, how="left", on=["timepoint", "measurement", "roi", "value"])
    filtereddf = merger.compute()

    filtereddf = filtereddf[pd.isna(filtereddf["group_y"])]
    filtereddf.drop("group_y", axis=1, inplace=True)
    filtereddf.columns = list(inputdf.columns)

    length_input = len(inputdf[inputdf["timepoint"]==0])
    length_filtered = len(filtereddf[filtereddf["timepoint"]==0])
    delta = length_input - length_filtered

    print('Initital Mean: ' + str(initialmean) + '. Initial SD: ' + str(initialsd))
    print('Threshold: ' + str(threshold))
    print('Dataframe was filtered')
    print('Total cells: ' + str(length_input))
    print(str(delta) + ' cells were removed')
    print('\n')

    return filtereddf, pre_activated


def total_means_long(traces_list,kinetics_list,output_total=False):

    combined_traces = pd.DataFrame()
    combined_kinetics = pd.DataFrame()

    for df in kinetics_list:
        means = df.groupby(["parameter","measurement","group","group_parameter"]).mean().drop(["roi"],axis=1).reset_index()
        stds = df.groupby(["parameter","measurement","group","group_parameter"]).std().drop(["roi"],axis=1).reset_index()
        means["statistic"]="mean"
        stds["statistic"] = "sd"
        merged_kinetics = pd.merge(means, stds, how="outer")
        combined_kinetics = pd.concat([combined_kinetics,merged_kinetics],axis=0)

    for df in traces_list:
        meantrace = df.groupby(["measurement","group","timepoint"]).mean().reset_index()
        meantrace["statistic"]="mean"
        sdtrace = df.groupby(["measurement","group","timepoint"]).std().reset_index()
        sdtrace["statistic"]="sd"
        merged_traces = pd.merge(meantrace, sdtrace, how="outer")
        combined_traces = pd.concat([combined_traces,merged_traces],axis=0)

        combined_traces.reset_index()
        combined_kinetics.reset_index()

    if output_total==False:
          return combined_traces, combined_kinetics

    if output_total == True:
        total_traces = pd.concat(traces_list, axis=0).reset_index()
        total_kinetics = pd.concat(kinetics_list,axis=0).reset_index()

        return combined_traces, combined_kinetics,total_traces,total_kinetics
