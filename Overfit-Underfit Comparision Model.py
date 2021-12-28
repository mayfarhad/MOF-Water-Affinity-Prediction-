# -*- coding: utf-8 -*-
"""
Written by: May Farhad
Last Edited: Dec. 27th 2021

This program compares the performance of the Random Forest machine learning
model when a different number and combination of inputs is used. 
"""

import csv
import time  
import pandas as pd
from mendeleev import element
from multiprocessing import Pool
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import CheckButtons, Button
from sklearn.model_selection import cross_val_score
from xgboost import XGBRFClassifier
import numpy as np


def writecsv(output_data, filename):  # update csv with score and std of each model 

    with open(filename, 'a+', newline='') as f:

        writer = csv.writer(f)
        writer.writerow(output_data)


def newcsv(filename):  # create new csv

    with open(filename, 'w', newline='') as f:

        writer = csv.writer(f)
        writer.writerow(['Number of Input Parameters',
                        'Avg. Score', 'Avg. std', 'Parameters Used'])


class Plotting:  # create a scatter plot to display the comparisson data

    def __init__(self, output_data, new_output_data, column_names, initial_status):

        self.original_data = output_data # contains results of all models
        self.new_output_data = new_output_data # contains results of the filtered data
        self.column_names = column_names
        columns = list(zip(*new_output_data))

        # set data point colours based on # of input parameters used in model
        output_colour_chart = {3: "red", 4: "orange", 5: "yellow", 6: "green",
                               7: "cyan", 8: "blue", 9: "purple", 10: "black"}
        output_colours = [output_colour_chart[i]
                          for i in columns[0] if i in output_colour_chart]

        plt.scatter(columns[1], columns[2], color=output_colours)
        plt.subplots_adjust(0.4)
        plt.xlabel('Average Score (%)')
        plt.ylabel('Average Standard Deviation (%)')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.title('Random Forest ML Model Accuracy')

        red_patch = mpatches.Patch(color='red', label='redpatch')
        orange_patch = mpatches.Patch(color='orange', label='orangepatch')
        yellow_patch = mpatches.Patch(color='yellow', label='yellowpatch')
        green_patch = mpatches.Patch(color='green', label='greenpatch')
        cyan_patch = mpatches.Patch(color='cyan', label='cyanpatch')
        blue_patch = mpatches.Patch(color='blue', label='bluepatch')
        purple_patch = mpatches.Patch(color='purple', label='purplepatch')
        black_patch = mpatches.Patch(color='black', label='blackpatch')

        plt.legend([red_patch, orange_patch, yellow_patch, green_patch,
                    cyan_patch, blue_patch, purple_patch, black_patch],
                   ['3', '4', '5', '6', '7', '8', '9', '10'],
                   loc="best", title="Number of Input Parameters")

        # Create checkbuttons
        ax_checkbox = plt.axes([0.02, 0.05, 0.28, 0.9])
        self.checkBox = CheckButtons(ax_checkbox, column_names, initial_status)

        # Create filter button
        ax_button = plt.axes([0.15, 0.05, 0.15, 0.075])
        filter_button = Button(ax_button, 'Apply Filters')
        filter_button.on_clicked(self.apply_filters)

        # Create csv button
        ax2_button = plt.axes([0.15, 0.31, 0.15, 0.075])
        csv_button = Button(ax2_button, 'Create csv')
        csv_button.on_clicked(self.filtered_csv)

        plt.show()
        while True:
            plt.waitforbuttonpress()

    def filtered_csv(self, cb):  # create new csv with filtered data

        with open('Filtered_model.csv', 'w', newline='') as f:

            writer = csv.writer(f)
            writer.writerow(['Number of Input Parameters',
                            'Avg. Score', 'Avg. std', 'Parameters Used'])
            writer.writerows(self.new_output_data)

    def getfilters(self, cb):  # check which filters are applied

        self.clicked = self.get_status(self.checkBox)  # boolean
        self.show = set()  # string

        for i in range(len(self.clicked)):
            if self.clicked[i] == True:
                self.show.add(self.column_names[i])

        metals = ["Metal 1", "Percent 1", "Metal 2", "Percent 2"]
        if "Metals and percentages" in self.show:
            for i in range(4):
                self.show.add(metals[i])

    def get_status(self, cb):  # check if a filter is applied

        return [l1.get_visible() for (l1, l2) in cb.lines]

    def apply_filters(self, cb):  # create a new plot with the applied filters

        self.getfilters(self.column_names)

        new_data = []

        for i in range(len(self.original_data)):

            if set(self.original_data[i][3]).issubset(self.show):
                new_data.append(self.original_data[i])

        plt.close()
        
        # create new plot with the applied filters
        Plotting(self.original_data, new_data, self.column_names, self.clicked)


def accuracy(data, water_affinity):  # apply machine learning model to dataset

    clf = XGBRFClassifier(
        nthread=1, use_label_encoder=False, eval_metric='logloss')
    scores = cross_val_score(clf, data, water_affinity, cv=19)

    return scores.mean()*100, scores.std()*100


def limited_inputs(data, column_indexes): # create dataset with x number of input parameters

    new_data = []

    for i in data:
        temp_data = []
        for a in column_indexes:
            temp_data.append(i[a])
        new_data.append(temp_data)

    return new_data


def generate_inputs(length):  # create list of all possible input parameter combinations

    return list(combinations(range(13), length))


def load_data(): #loads MOF crytallographic data from csv

    needed = ["LCD", "PLD", "LFPD", "cm3_g", "ASA_m2_cm3",
              "ASA_m2_g", "NASA_m2_cm3", "NASA_m2_g", "AV_VF", "AV_cm3_g",
              "NAV_cm3_g", "Has_OMS", "Metal 1", "Percent 1", "Metal 2",
              "Percent 2", "Hydrophobic / Hydrophilic"]
    
    df = pd.read_csv('MOF_Training_Set.csv', header=0, usecols=needed)
    df = df.astype({"LCD": float, "PLD": float, "LFPD": float,
                    "cm3_g": float, "ASA_m2_cm3": float, "ASA_m2_g": float,
                    "NASA_m2_cm3": float, "NASA_m2_g": float, "AV_VF": float,
                    "AV_cm3_g": float, "NAV_cm3_g": float, "Has_OMS": str,
                    "Metal 1": str, "Percent 1": float, "Metal 2": str,
                    "Percent 2": float, "Hydrophobic / Hydrophilic": int})

    data = df.values.tolist()  # data to be processed by ML model

    yes_no = {"Yes": 1, "No": 0}
    water_affinity = []

    for i in data:

        for a in [12, 14]:  # change metal columns from metal name to periodic groups

            if i[a] != "nan":

                i[a] = element(i[a]).group_id

        i[11] = yes_no[i[11]]  # change yes to 1 and no to 0 in Has_OMS column
        water_affinity.append(i.pop())  # create list for water affinity

    return data, water_affinity


def readin_data(filename):  # read back ML model comparission data from existing csv

    output_data = []

    with open(filename, 'r', newline='') as f:

        reader = csv.reader(f)

        for i, line in enumerate(reader):

            if i == 0:

                continue

            temp = []
            temp.append(int(line[0]))
            temp.append(float(line[1]))
            temp.append(float(line[2]))
            temp.append((line[3].strip('][').replace("'", "").split(', ')))
            output_data.append(temp)

    return output_data


def get_column_names(): #returns names of the input parameters used

    df = pd.read_csv('MOF_Training_Set.csv', header=0)
    column_names_MOF = df.columns.values  
    column_names = np.append(column_names_MOF[1:-5], "Metals and percentages")

    return column_names


if __name__ == "__main__":

    start_time = time.time()
    column_names = get_column_names()

    readin = input(
        "Would you like to read in the previously saved file? (y/n): \n")

    if readin == 'y':  # read back ML model comparission data from existing csv

        filename = input("Enter the name of the file e.g. filename.csv: ")
        output_data = readin_data(filename)

    elif readin == 'n':  # create new comparission data

        print("Enter the length of the combinations you would like. Choose 1-12.")
        enter_inputs = input(
            "e.g. 4 5 9 would produce combinations that are 4, 5, and 9 in length: ")
        input_parameters = enter_inputs.split()
        integer_map = map(int, input_parameters)
        num_inputs = list(integer_map)
        
        data, water_affinity = load_data()  # load MOF crystallographic data from csv

        # create list of all possible input combinations
        with Pool(processes=3) as pool:
            column_indexes = pool.map(generate_inputs, num_inputs)
        pool.close()

        # create csv to save results of ML models
        all_data_csv = 'Overfit-Underfit_outputdata.csv'
        newcsv(all_data_csv)
        output_data = []  # saves results of ML models into list

        for i in column_indexes:  # performs Random Forest on each input combination

            length = len(i[0])  # specifies the length of the combination

            for x in range(len(i)):

                outputs = [length]

                # stores the names of the input parameters used
                columns_used = []
                i[x] = list(i[x])
                for a in i[x]:
                    columns_used.append(column_names[a])

                # columns 12-15 are treated as one. Metals and %metal
                if 12 in i[x]:
                    i[x].extend([13, 14, 15])

                # generate unique dataset for given input parameter combination
                new_data = limited_inputs(data, i[x])
                avg_score, avg_std = accuracy(
                    new_data, water_affinity)  # apply ML model

                outputs.append(avg_score)
                outputs.append(avg_std)
                outputs.append(columns_used)

                output_data.append(outputs)

                # Print results
                print(f"The inputs used were: {i[x]}")
                print(f"Average Score: {avg_score:.2f}%")
                print(f"avg std: {avg_std:.2f}% \n")

                writecsv(outputs, all_data_csv)  # write results to csv

    print(f"My program took {time.time() - start_time} to run")

    # Plot results
    initial_status = [True] * len(column_names)
    Plotting(output_data, output_data, column_names, initial_status)
