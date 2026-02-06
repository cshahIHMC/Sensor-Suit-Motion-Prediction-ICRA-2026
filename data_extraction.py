
## This file reads from the GaTech Data set and writes it to a single csv.

import os
import pandas as pd
import csv

# Downsampling function - down samples 1000Hz data to 200 Hz

def downSample(df, downsamplingFactor):
    
    df_downSampled = pd.DataFrame()
    
    for col in df:
        # Downsample the array by selecting every 'downsampling_factor'-th data point
        downsampled_value = df[col][::downsamplingFactor]
        df_downSampled[col] = downsampled_value
        
    df_downSampled = df_downSampled.reset_index(drop=True)
    
    return df_downSampled

# Extracts and writes the data to an sql database fromm the Ryan and Christoph Dataset
def extract_write_2_csv(dataDir, subject_list, dataWriteDir):

    df_subjects = pd.DataFrame()
    
    for SubjectName, SubjectSpecificParams in subject_list.items():
        
        # Just to know what subject we are iterating over
        print(SubjectName)

        # Get a Subject data path
        subjectDataPath = dataDir + SubjectName + "MachineLearning/"

        # Change the target directory
        os.chdir(subjectDataPath)

        # Make a list of the folders containing data for different conditions
        conditions_list = [directory for directory in os.listdir() if os.path.isdir(directory)]

        df_subject = pd.DataFrame()

        for conditionName in conditions_list:
            
            # Folder path to the conditions
            conditionFolderPath = subjectDataPath + conditionName

            # List of all the files in the condition folder
            all_files = os.listdir(conditionFolderPath)

            # Weed out csv files
            files = [file for file in all_files if file.endswith(".csv")]
            
            # Initialize a pandas dataframe  
            df_condition = pd.DataFrame()
            
            for file in files:
 
                # Data Path
                dataPath = conditionFolderPath + "/" + file
        
                # Read it into a pandas dataframe

                df_data = pd.read_csv(dataPath)
                
                # if NAN fill it with 0s
                df_data.fillna(0, inplace=True)
                
                # Add data from the different files in the condition folder on top of each other
                df_condition = pd.concat([df_condition, df_data], axis=0, ignore_index=True)
                
                

            # Remove duplicate cols
            df_condition = df_condition.loc[:, ~df_condition.columns.duplicated()]

            # Add the name of the condition that we adding data to the database for
            df_condition['condition'] = conditionName
            df_condition['subject'] = SubjectName[:-1]
            df_condition['weight'] = SubjectSpecificParams[0]  # Weight is in Kgs
            df_condition['height'] = SubjectSpecificParams[1]  # Height is in m
            df_condition['age'] = SubjectSpecificParams[2]  # Age is in years
            df_condition['sex'] = SubjectSpecificParams[3]  # Sex is M-0 / F-1

            # Generate a subject level df
            df_subject = pd.concat([df_subject, df_condition], axis=0, ignore_index=True)      
            

        print(df_subject.shape)
        
        df_subjects = pd.concat([df_subjects, df_subject], axis=0, ignore_index=True)
    
    print("Overall Data")       
    print(df_subjects.shape)
    # for col in df_subject.columns:
        # print(col)
    df_subjects.to_csv(dataWriteDir,
                    index=False,
                    float_format='%.6f',      # e.g. 0.123457
                    quoting=csv.QUOTE_MINIMAL)

    


def main():
    data_dir = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data - Second Skin/OpenSource_Dataset/Data/"
    # sub_Name = "AB01/"
    data_write_dir = "/home/cshah/workspaces/Sensor-Suit-Motion-Prediction-ICRA-2026/Data - Second Skin/Testing/5_Subjects_all_data.csv"
    
    
    # Subject Information - [Weight(kg), Height(m), Age(yrs), Sex(M-0 / F-1)]]
    sub_list = {
        "AB01/": [86.9, 1.75, 23, 0],   
        # "AB02/": [84.6, 1.72, 22, 0],
        # "AB03/": [65.05, 1.765, 32, 0],
        # "AB04/": [86.4, 1.794, 25, 0],
        # "AB05/": [87.0, 1.59, 27, 1],
        # "AB06/": [71.55, 1.868, 24, 0],
        # "AB08/": [79.0, 1.82, 24, 0],
        # "AB09/": [58.2, 1.701, 24, 0],
        # "AB10/": [92.5, 1.825, 27, 0],
        # "AB11/": [73.5, 1.606, 28, 1] 
        
    }
    
    # extract_write_2_csv(data_dir,sub_Name, data_write_dir, weight=78.90)
    extract_write_2_csv(data_dir,sub_list, data_write_dir)

if __name__ == "__main__":
    raise SystemExit(main())
    
