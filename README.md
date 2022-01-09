# MOF-Water-Affinity-Prediction-
The following Python scripts aim to use a Random Forest machine learning algorithm to predict the water affinity of Metal-Organic Frameworks (MOFs). The training set is extracted from the Cambridge Structural Database and the CoREMOF 2019 dataset.

## **Prediction Model**

The prediction model is used to determine whether a given MOF is hydrophobic or hydrophilic. It uses a Random Forest model from the XGBoost library through a scikit-learn interface. The model reads in a CSV file of training data and then predicts the water affinity of a user inputted MOF. The user can specify what input parameters are to be used in the model.

## **Overfitting/Underfitting**

This script was created to investigate how the prediction model’s accuracy and precision vary with the number and combination of inputs. This script allows a user to compare how the different combinations of inputs affect the score and the standard deviation of the model’s results.

It operates by reading in a CSV file of training data containing 13 input parameters. It then generates a list of all the possible combinations of input parameters according to the lengths specified by the user. For example, if the user wants all the combinations of length 3, 4, and 10 possible, the program will generate a list of all combinations of those lengths, and then use each combination as input for the model. Basically, each combination will undergo the same process as in the prediction model above, and then its results will be added into a CSV file for later analysis. Finally, a plot is created with filters for visualization.

## **CIF to CSV Converter**

In order to create a training set for the prediction model, a CSV must be created with all the available datapoints. This includes the MOFs and their crystallographic data. The data needed is collected from three different sources: WebCSD, CoREMOF 2019 dataset, and the MOF’s CIF files. Furthermore, additional calculations need to be performed from the information collected from the CIF files.

The code works by reading a TXT file, folder, or both, containing the refcodes and CIF files given to the MOF by the Cambrdige Structural Database. It then searches for these refcodes in the CoREMOF 2019 dataset, and retrieves the crystallographic data attached to them. Additionally, it uses the CIF files of the MOFs to calculate the atomic mass percentage of the metals contained in the MOF. These calculations are stored in columns 14-17, but are treated as one input parameter in the models in an attempt to relate them to each other. It also states the MOFs in the training set as hydrophobic and hydrophilic based on previously collected information from the literature. Finally, it produces a CSV file ready for use in the prediction model. 

### **CIF folders**
Three different folders are used to store CIF files. 
1. cif: these are hydrophobic MOFs received from Dr. Z. Qiao.
2. manual hydrophobic: these are hydrophobic MOFs collected from the literature
3. manual hydrophilic: these are hydrophilic MOFs collected from the literature

### _**To add additional .cif files:**_
Add additional CIF files into either the manual hydrophobic folder or the manual hydrophilic folder. Make sure the file names represent the CCDC refcodes (including or excluding the CoREMOF 2019 name extensions). Finally, add these refcodes into the TXT file available in each folder so that the CIF files can be read by the CIF Reader program. 


This project is licensed under the terms of the GNU General Public License v3.0
