# -*- coding: utf-8 -*-
"""
Written by: May Farhad
Last Edited: Dec 27th 2021

This program populates a csv with MOF data from .cif files and the 
CoRE_MOF 2019 dataset. Make sure to update line 36 with the file path.
"""

import os
import io
import csv
import pandas as pd
import numpy as np
from gemmi import cif
from mendeleev import element


def main():

    # read the refcodes and metal percentages from the cif files, water affinity is user specified
    refcodes, elements_percentages, water_affinity = readcif()

    # get crystallographic data from CoRE_MOF 2019 dataset
    crystal_data, column_names = getdata(
        refcodes, elements_percentages, water_affinity)

    # write MOF data into csv file
    writecsv(crystal_data, column_names)


# reads the refcodes and metal percentages from the cif files, water affinity is user specified
def readcif():

    # TODO: update with file path e.g. C:/folder_where_program_is_located
    work_dir = ""
    refcodes = []  # list of CCDC refcodes
    path_names = []  # stores the path to the .cif files of each MOF
    metal_percentages = [] # each element formatted as: [metal 1, %metal1] or [metal 1, %metal 1, metal 2, %metal 2]
    water_affinity = []  # hydrophobic is 0, hydrophilic is 1

    # read refcodes of hydrophobic MOFs received from Dr. Z. Qiao from a folder
    for index in range(1, 607):
        name = "cif/hydrCOREMOF{index}.cif".format(index=index)
        path = os.path.join(work_dir, name)
        with io.open(path, mode="r", encoding="utf-8") as fd:
            content = fd.readline().strip()
            refcodes.append(content[5:])
            path_names.append(path)
            water_affinity.append(0)

    # read refcodes of hydrophobic MOFs from a list in a .txt file
    hydrophobic_MOFs = []  # store the hydrophobic refcodes
    f_hydrophobic = open(os.path.join(
        work_dir, "manual_hydrophobic/manual_hydrophobic.txt"), "r")
    for x in f_hydrophobic:
        hydrophobic_MOFs.append(x.rstrip('\n'))
    f_hydrophobic.close()

    # use stored hydrophobic refcodes to get path of the .cif files
    for i in hydrophobic_MOFs:
        name = "manual_hydrophobic/{i}.cif".format(i=i)
        path = os.path.join(work_dir, name)
        refcodes.append(i)
        path_names.append(path)
        water_affinity.append(0)

    # read refcodes of hydrophilic MOFs from a list in a .txt file
    hydrophilic_MOFs = []  # store the hydrophilic refcodes
    f_hydrophilic = open(os.path.join(
        work_dir, "manual_hydrophilic/manual_hydrophilic.txt"), "r")
    for x in f_hydrophilic:
        hydrophilic_MOFs.append(x.rstrip('\n'))
    f_hydrophilic.close()

    # use stored hydrophilic refcodes to get path of the .cif files
    for i in hydrophilic_MOFs:
        name = "manual_hydrophilic/{i}.cif".format(i=i)
        path = os.path.join(work_dir, name)
        refcodes.append(i)
        path_names.append(path)
        water_affinity.append(1)

    # use function to get the metals and percentage of each metal in each MOF
    for i in range(len(refcodes)):

        # use .cif file of MOF to extract atomic data
        getelements = get_atoms(path_names[i])
        metal_percentages.append(getelements)  # store results

    return refcodes, metal_percentages, water_affinity


def get_atoms(filename):  # use .cif file of MOFs to extract atomic data

    metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
              'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr',
              'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs',
              'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
              'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
              'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th',
              'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
              'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
              'Nh', 'Fl', 'Mc', 'Lv']

    nonmetals = ['H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'P', 'S', 'Cl', 'Ar', 'Se',
                 'Br', 'Kr', 'I', 'Xe', 'At', 'Rn', 'Ts', 'Og']

    # extract elements in .cif and use to calculate the atomic mass % of each present metal
    unique = []
    metal_elements = [] # formatted as: [metal 1, %metal1] or [metal 1, %metal 1, metal 2, %metal 2]
    num_atoms = 0
    total_atomic_mass = 0
    doc = cif.read_file(filename).sole_block()

    for i in doc.find_loop('_atom_site_type_symbol'):  # read the atoms in .cif

        if i in metals or i in nonmetals:

            current_element = element(i)
            current_atomic_mass = current_element.atomic_weight
            total_atomic_mass += current_atomic_mass
            num_atoms += 1

            if i not in unique and i in metals:

                unique.append(i)
                metal_elements.append([i, current_atomic_mass])

            elif i in metals:

                index = unique.index(i)
                metal_elements[index][1] += current_atomic_mass

        else:

            return metal_elements  # if an ion is present in MOF, MOF is excluded from dataset

    for i in metal_elements:  # finds atomic mass percentage of each metal present

        i[1] = i[1]/total_atomic_mass

    return metal_elements


# gets crystallographic data from CoRE_MOF 2019 dataset
def getdata(refcodes, elements_percentages, water_affinity):

    crystal_data = []  # where all the data is stored

    needed = ["filename", "LCD", "PLD", "LFPD", "cm3_g", "ASA_m2_cm3", "ASA_m2_g",
              "NASA_m2_cm3", "NASA_m2_g", "AV_VF", "AV_cm3_g", "NAV_cm3_g",
              "Has_OMS"]  # columns needed from CoRE_MOF 2019 dataset

    # read needed columns into a dataframe, and then into a list
    df = pd.read_csv('2019-11-01-ASR-internal_14142.csv',
                     header=0, usecols=needed)
    data = df.values.tolist()
    woS = pd.read_csv('2019-11-01-ASR-internal_14142.csv',
                      header=0, usecols=["Matched_CSD_of_CoRE"])
    WoS = woS.values.tolist()

    column_names = np.concatenate([needed, [
                                  "Metal 1", "Percent 1", "Metal 2", "Percent 2", "Hydrophobic / Hydrophilic"]])
    columns = list(zip(*data))

    extensions = ["_auto", "_charged", "_clean", "_clean_h", "_clean_h_min",
                  "_combined", "_ion_b", "_manual", "_neutral", "_neutral_b",
                  "_SL", "_SL_part"]

    for i in range(len(refcodes)):  # get data from CoRE_MOF 2019 for each MOF

        # if .cif file was missing atomic data, do not include
        if not elements_percentages[i]:

            continue

        else:

            found = False
            duplicate = False

            for x in range(len(data)):

                if columns[0][x] == refcodes[i] or WoS[x][0] == refcodes[i]:  # if refcode is found

                    # add CoRE_MOF 2019 data into traning set
                    crystal_data.append(data[x])
                    found = True
                    break

                else:

                    for a in extensions:

                        newcode = refcodes[i] + a

                        # if refcode uses an extension
                        if columns[0][x] == newcode and newcode not in refcodes:

                            # add CoRE_MOF 2019 data into traning set
                            crystal_data.append(data[x])
                            found = True
                            break

                        elif newcode in refcodes:  # if a duplicate is found, skip it

                            duplicate = True
                            found = True
                            break

                    if found == True:

                        break

            if found == False or duplicate == True:

                continue  # if refcode is not in CoRE_MOF 2019 dataset or is a duplicate, do not include it in training set

            # change elements_percentages from 2D to 1D list, add to training set
            # if one metal present, halve percentages and represent as two metals
            if len(elements_percentages[i]) == 1:

                elements_percentages[i] = [elements_percentages[i][0][0], elements_percentages[i]
                                           [0][1]/2, elements_percentages[i][0][0], elements_percentages[i][0][1]/2]

            elif len(elements_percentages[i]) == 2:

                elements_percentages[i] = [elements_percentages[i][0][0], elements_percentages[i]
                                           [0][1], elements_percentages[i][1][0], elements_percentages[i][1][1]]

            crystal_data[-1] = [*crystal_data[-1], *elements_percentages[i],
                                *[water_affinity[i]]]

    return crystal_data, column_names


def writecsv(crystal_data, column_names):  # write MOF data into csv file

    with open('MOF_Training_Set.csv', 'w', newline='') as f:

        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerows(crystal_data)


if __name__ == "__main__":
    main()
