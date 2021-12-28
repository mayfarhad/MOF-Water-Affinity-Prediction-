# -*- coding: utf-8 -*-
"""
Written by: May Farhad
Last Edited: Dec. 27th 2021

This program predicts the water affinity of a MOF. To use it, specify the 
number of input parameters you want used in the Random Forest model. Then enter
the parameters for the MOF in question.
"""

import pandas as pd
from mendeleev import element
from sklearn.model_selection import cross_val_score
from xgboost import XGBRFClassifier


# create dataset with x number of input parameters
def limited_inputs(data, column_indexes):

    new_data = []

    for i in data:
        temp_data = []
        for a in column_indexes:
            temp_data.append(i[a])
        new_data.append(temp_data)

    return new_data


def accuracy(data, water_affinity, properties):  # apply machine learning model to dataset

    clf = XGBRFClassifier(
        nthread=1, use_label_encoder=False, eval_metric='logloss')
    scores = cross_val_score(clf, data, water_affinity, cv=19)
    clf.fit(data, water_affinity)
    results = clf.predict([properties])

    return scores.mean()*100, scores.std()*100, results


def load_data(input_names): #gets required input parameters for training set from csv
    
    df = pd.read_csv('MOF_Training_Set.csv', header=0)

    df = df.astype({"filename": str, "LCD": float, "PLD": float, "LFPD": float,
                    "cm3_g": float, "ASA_m2_cm3": float, "ASA_m2_g": float,
                    "NASA_m2_cm3": float, "NASA_m2_g": float, "AV_VF": float,
                    "AV_cm3_g": float, "NAV_cm3_g": float, "Has_OMS": str, 
                    "Metal 1": str, "Percent 1": float, "Metal 2": str, 
                    "Percent 2": float, "Hydrophobic / Hydrophilic": int})

    data = df[input_names]
    water = df[["Hydrophobic / Hydrophilic"]]
    new_data = data.values.tolist()
    water_affinity = water.values.tolist()

    
    for i in range(len(input_names)):

        if input_names[i] == "Metal 1" or input_names[i] == "Metal 2":

            for a in new_data:

                a[i] = element(a[i]).group_id #get the atomic number of metals

        elif input_names[i] == "Has_OMS":

            yes_no = {"Yes": 1, "No": 0}
            for a in new_data:

                a[i] = yes_no[a[i]] # change yes to 1 and no to 0 in Has_OMS column

    return new_data, water_affinity


def user_input(): #gets input parameters from user

    inputs = {1: "LCD", 2: "PLD", 3: "LFPD", 4: "cm3_g", 5: "ASA_m2_cm3",
              6: "ASA_m2_g", 7: "NASA_m2_cm3", 8: "NASA_m2_g", 9: "AV_VF",
              10: "AV_cm3_g", 11: "NAV_cm3_g", 12: "Metal 1", 13: "Percent 1",
              14: "Metal 2", 15: "Percent 2", 16: "Has_OMS"}

    print(inputs)
    print("\nPlease type in the numbers of the inputs you would like to include.")
    print("Inputs 12-15 cannot be separated, type in 12 to include them.")
    readin = input("e.g. 4 5 9: ")
    input_parameters = readin.split()
    integer_map = map(int, input_parameters)
    integer_list = list(integer_map)

    properties = []
    input_names = []

    for i in integer_list:

        get_property = input(f"Enter the {inputs[i]}: ")
        input_names.append(inputs[i])
        
        if i == 12: #metals and percentages treated as one input parameter
            integer_list.extend([13, 14, 15]) 

        if i == 12 or i == 14:
            get_property = element(get_property).group_id #get atomic number
    
        properties.append(get_property)
        
    float_properties = map(float, properties)
    list_properties = list(float_properties)

    return list_properties, input_names


if __name__ == "__main__":

    #gets input parameters from user
    properties, input_names = user_input()
    
    #gets required input parameters for training set from csv
    data, water_affinity = load_data(input_names)
    
    #performs Random Forest and predicts MOF water affinity
    mean, std, results = accuracy(data, water_affinity, properties)
    
    affinity = {1: "Hydrophilic", 0: "Hydrophobic"}
    print(f"The MOF is: {affinity[results[0]]}")
