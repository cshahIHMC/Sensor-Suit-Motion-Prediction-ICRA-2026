
## This file reads from the GaTech Data set and writes it to a single csv.

import os
import pandas as pd
import csv
# weight = 78.9 # AB01: 78.9 kg
# weight = 82.2 # AB02: 82.2 kg
# weight = 113.5# AB03: 113.5 kg
# weight = 71.5 # AB05: 71.5 kg
# weight = 79.1 # AB06: 79.1 kg
# weight = 62.3 # AB07: 62.3 kg
# weight = 87.6 # AB08: 87.6 kg 
# weight = 84.1 # AB09: 84.1 kg
# weight = 67.5 # AB10: 67.5 kg
# weight = 65.1 # AB11: 65.1 kg
# weight = 64.0 # AB12: 64.0 kg
# weight = 67.6 # AB13: 67.6 kg


# Downsampling function - down samples 1000Hz data to 200 Hz

def downSample(df, downsamplingFactor):
    
    df_downSampled = pd.DataFrame()
    
    for col in df:
        # Downsample the array by selecting every 'downsampling_factor'-th data point
        downsampled_value = df[col][::downsamplingFactor]
        df_downSampled[col] = downsampled_value
        
    df_downSampled = df_downSampled.reset_index(drop=True)
    
    return df_downSampled

# Extracts and writes the data to an sql database fromm the keaton scherpeel dataset
# def extract_write_2_csv(dataDir, SubjectName, dataWriteDir, weight=0.0):
def extract_write_2_csv(dataDir, subject_list, dataWriteDir):

    df_subjects = pd.DataFrame()
    
    for SubjectName, weight in subject_list.items():
        
        # Just to know what subject we are iterating over
        print(SubjectName)

        # Get a Subject data path
        subjectDataPath = dataDir + SubjectName

        # Change the target directory
        os.chdir(subjectDataPath)

        # Make a list of the folders containing data for different conditions
        conditions_list = [directory for directory in os.listdir() if os.path.isdir(directory)]


        df_subject = pd.DataFrame()

        for dir in conditions_list:

            # Folder path to the conditions
            conditionFolderPath = subjectDataPath + dir


            all_files = os.listdir(conditionFolderPath)

            # Weed out csv files
            files = [file for file in all_files if file.endswith(".csv")]

            # Initialize a pandas dataframe  for every condition
            df_condition = pd.DataFrame()

            # Condition Name
            conditionName = SubjectName[:-1] + "_" + dir

            for file in files:

                # Get the name of the type of data
                dataName = file[len(conditionName)+1:-4]

                if dataName == "activity_flag" or dataName == "emg" or dataName == "moment" or dataName == "power" or dataName == "grf" or dataName == "moment_filt" or dataName == "imu_real":
                    continue 
                
                # Data Path
                dataPath = conditionFolderPath + "/" + file
                # Read it into a pandas dataframe

                df_data = pd.read_csv(dataPath)

                # Only the EMG Data needs downsampling
                if dataName == "emg":
                    downSamplingFactor = 10
                    df_data = downSample(df_data, downSamplingFactor) 

                df_data.fillna(0, inplace=True)

                ## Add data from the different data folder to one single data frame right next to each other
                df_condition = pd.concat([df_condition, df_data], axis=1)

            ## Remove duplicate cols
            df_condition = df_condition.loc[:, ~df_condition.columns.duplicated()]

            ## Add the name of the condition that we adding data to the database for
            df_condition['condition'] = dir
            df_condition['subject'] = SubjectName[:-1]
            df_condition['weight'] = weight

            # Generate a subject level df
            df_subject = pd.concat([df_subject, df_condition], axis=0, ignore_index=True)      
        
        print(df_subject.shape)
        
        df_subjects = pd.concat([df_subjects, df_subject], axis=0, ignore_index=True)
    
    print("Overall Data")       
    print(df_subjects.shape)
    # for col in df_subject.columns:
        # print(col)
    # df_subjects.to_csv(dataWriteDir,
    #                 index=False,
    #                 float_format='%.6f',      # e.g. 0.123457
    #                 quoting=csv.QUOTE_MINIMAL)

    


def main():
    data_dir = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data/Scheerpeel_Data_set/ProcessedData/"
    # sub_Name = "AB01/"
    data_write_dir = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data/all_subjects_req_sim_data.csv"
    
    sub_list = {
        "AB01/": 78.9,
        "AB02/": 82.2,
        "AB03/": 113.5,
        "AB05/": 71.5,
        "AB06/": 79.1,
        "AB07/": 62.3,
        "AB08/": 87.6,
        "AB09/": 84.1,
        "AB10/": 67.5,
        "AB11/": 65.1   
    }
    # weight = 78.9 # AB01: 78.9 kg
    # weight = 82.2 # AB02: 82.2 kg
    # weight = 113.5# AB03: 113.5 kg
    # weight = 71.5 # AB05: 71.5 kg
    # weight = 79.1 # AB06: 79.1 kg
    # weight = 62.3 # AB07: 62.3 kg
    # weight = 87.6 # AB08: 87.6 kg 
    # weight = 84.1 # AB09: 84.1 kg
    # weight = 67.5 # AB10: 67.5 kg
    # weight = 65.1 # AB11: 65.1 kg
    # weight = 64.0 # AB12: 64.0 kg
    # weight = 67.6 # AB13: 67.6 kg

    
    # extract_write_2_csv(data_dir,sub_Name, data_write_dir, weight=78.90)
    extract_write_2_csv(data_dir,sub_list, data_write_dir)

if __name__ == "__main__":
    raise SystemExit(main())
    
