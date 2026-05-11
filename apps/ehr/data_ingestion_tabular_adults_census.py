
import json
import pandas as pd
import os
import numpy as np


### Adult Census
# 10 qubit: AGE: n5, INCOME: b1, WORKCLASS: c4
# Non_Boolean: 10 +1 qubit: AGE: n5, INCOME: c2, WORKCLASS: c4
# 15 qubit: AGE: n5, WORKCLASS: c4, EDUCATION: c6
# 20 qubit: AGE: n5, HOURS PER WEEK: n4 ,INCOME: b1, WORKCLASS: c4, EDUCATION: c6
# Non_Boolean: 20 +1 qubit: AGE: n5, HOURS PER WEEK: n4 ,INCOME: c2, WORKCLASS: c4, EDUCATION: c6

plots_path = "./training_data"

# plots_path = f"{os.getcwd()}/{file_path}"

# data set name
data_set_name_10 = 'adults_census_10'
data_set_name_10_non_boolean = 'adults_census_10_non_boolean'

data_set_name_15 = 'adults_census_15'

data_set_name_20 = 'adults_census_20'
data_set_name_20_non_boolean = 'adults_census_20_non_boolean'

# data_specs
# 10 qubits data specs
data_spec_10 = ['n5', 'b1', 'c4']
data_spec_10_non_boolean = ['n5', 'c2', 'c4']
# 15 qubits data specs
data_spec_15 = ['n5','c4', 'c6']
# 20 qubits data specs
data_spec_20 = ['n5','n4','b1','c4','c6']
data_spec_20_non_boolean = ['n5', 'n4', 'c2', 'c4', 'c6']

# column_names
column_names_10 = ['age', {"income": [{"Above-50K": "1"}, {"Below-50K": "0"}]},
                   {"workclass": [{"empl-unknown": "1000"}, {"govt-employed": "0100"}, {"self-employed": "0010"}, {"unemployed": "0001"}]}]
column_names_10_non_boolean = ['age', {"income": [{"Above-50K": "10"}, {"Below-50K": "01"}]},
                               {"workclass": [{"empl-unknown": "1000"}, {"govt-employed": "0100"}, {"self-employed": "0010"}, {"unemployed": "0001"}]}]

column_names_15 = ['age',
                   {"workclass": [{"empl-unknown": "1000"}, {"govt-employed": "0100"}, {"self-employed": "0010"}, {"unemployed": "0001"}]},
                   {"education": [{"Advanced": "100000"}, {"Bachelors": "010000"}, {"Below-HS": "001000"}, {"HS-grad": "000100"}, {"Some-College": "000010"}, {"Masters-or-equivalent": "000001"}]}]

column_names_20 = ['age', 'hours_per_week',
                   {"income": [{"Above-50K": "1"}, {"Below-50K": "0"}]},
                   {"workclass": [{"empl-unknown": "1000"}, {"govt-employed": "0100"}, {"self-employed": "0010"}, {"unemployed": "0001"}]},
                   {"education": [{"Advanced": "100000"}, {"Bachelors": "010000"}, {"Below-HS": "001000"}, {"HS-grad": "000100"}, {"Some-College": "000010"}, {"Masters-or-equivalent": "000001"}]}]

column_names_20_non_boolean = ['age', 'hours_per_week',
                               {"income": [{"Above-50K": "10"}, {"Below-50K": "01"}]},
                               {"workclass": [{"empl-unknown": "1000"}, {"govt-employed": "0100"}, {"self-employed": "0010"}, {"unemployed": "0001"}]},
                               {"education": [{"Advanced": "100000"}, {"Bachelors": "010000"}, {"Below-HS": "001000"}, {"HS-grad": "000100"}, {"Some-College": "000010"}, {"Masters-or-equivalent": "000001"}]}]
# load data

# adults = './training_data/adult.csv'
adults = './training_data/adult.csv'
adults_df = pd.read_csv(adults)

adults_subset = adults_df[['age','education','workclass','hours_per_week','income']]
#adults_subset.rename( columns={'hours.per.week':'hours_per_week'}, inplace=True)
# process age
#adults_subset = adults_subset[adults_subset['age']<81]s
# process hours_per_week
#adults_subset = adults_subset[adults_subset['hours_per_week']<80]


# process 4 categories for workclass
for value in adults_subset["workclass"].values:
    if value == "Self-emp-inc":
         adults_subset["workclass"].replace(value, "self-employed", inplace=True)
    elif value == "Local-gov" or value == "State-gov" or value == "Federal-gov" or value == "Private":
         adults_subset["workclass"].replace(value, "govt-employed", inplace=True)
    elif value == "Without-pay" or value == "Never-worked" or value == "Self-emp-not-inc":
         adults_subset["workclass"].replace(value, "unemployed", inplace=True)
    # elif value == "Private":
    #     adults_subset["workclass"].replace(value, "private-employed", inplace=True)
    elif value == "?":
        adults_subset["workclass"].replace(value, "empl-unknown", inplace=True)

# process 6 categories for education
for value in adults_subset["education"].values:
    if value == "Some-college" or value == "Assoc-acdm" or value == "Assoc-voc":
        adults_subset["education"].replace(value, "Some-College", inplace=True)
    elif value == "11th" or value == "10th" or value == "7th-8th" or value == "9th" or value == "12th" or value == "5th-6th" or value == "1st-4th" or value == "Preschool":
        adults_subset["education"].replace(value, "Below-HS", inplace=True)
    elif value == "Prof-school" or value == "Masters":
        adults_subset["education"].replace(value, "Masters-or-equivalent", inplace=True)

# process income
for value in adults_subset["income"].values:
    if value == "<=50K":
        adults_subset["income"].replace(value, "Below-50K", inplace=True)
    elif value == ">50K":
        adults_subset["income"].replace(value, "Above-50K", inplace=True)

def save_json(col_names, data_spec, no_of_qubits, data_set_name, path_to_training_data):
    meta = {}
    meta['column_names'] = col_names

    meta['data_spec'] = data_spec
    meta['n_qubits'] = no_of_qubits

    with open(
            path_to_training_data + "/" + data_set_name + "_meta.json", "w"
    ) as fp:
        json.dump(meta, fp)

#encode columns
encoded_education= pd.get_dummies(adults_subset["education"], dtype=int)
encoded_workclass = pd.get_dummies(adults_subset["workclass"], dtype=int)
encoded_income = adults_subset["income"].map(lambda x: '1' if x == "Above-50K" else '0')
encoded_income_non_boolean = pd.get_dummies(adults_subset["income"], dtype=int)

# process numeric data into binary
# calculate the bins for age
age_qubits = 5
age_max = adults_subset["age"].max()
age_min = adults_subset["age"].min()
age_bins =  np.arange(age_min, age_max, age_max/(2**age_qubits))
print(f'ages, {age_min} {age_max}, {age_bins}')
adults_subset.loc[:, 'age'] = np.digitize(adults_subset.loc[:, 'age'], age_bins)
encoded_age = adults_subset.loc[:, 'age'].apply(lambda x: format(x-1, f'0{age_qubits}b'))

# calculate the bins for hours_per_week
hours_per_week_max = adults_subset["hours_per_week"].max()
hours_per_week_min = adults_subset["hours_per_week"].min()
hours_per_week_bins =  np.arange(hours_per_week_min, hours_per_week_max, hours_per_week_max/2**4)

adults_subset.loc[:, 'age'] = np.digitize(adults_subset.loc[:, 'age'], age_bins)
encoded_hours_per_week = pd.Series(np.digitize(adults_subset.loc[:, 'hours_per_week'], hours_per_week_bins), name='hours_per_week').apply(lambda x: format(x-1, '04b'))

#-------------------------------------------------------------------------------
# concat encoded dataframes to make combined dataframes for 10, 15 and 20 qubits
#--------------------------------------------------------------------------------

# adults census 10 qubits
adults_census_10_original = pd.concat([adults_subset["age"], adults_subset["income"], adults_subset["workclass"]], axis= 1)
# boolean adults census 10 qubits
adults_census_10 = pd.concat([encoded_age, encoded_income, encoded_workclass], axis= 1)
adults_census_10['combined'] = adults_census_10.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
# adults census 10 qubits
adults_census_10_non_boolean = pd.concat([encoded_age, encoded_income_non_boolean, encoded_workclass], axis= 1)
adults_census_10_non_boolean['combined'] = adults_census_10_non_boolean.apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# adults census 15 qubits
adults_census_15_original = pd.concat([adults_subset["age"], adults_subset["workclass"], adults_subset["education"]], axis= 1)
adults_census_15 = pd.concat([encoded_age, encoded_workclass, encoded_education], axis= 1)
adults_census_15['combined'] = adults_census_15.apply(lambda row: ''.join(row.values.astype(str)), axis=1)


# adults census 20 qubits
adults_census_20_original = pd.concat([adults_subset["age"], adults_subset["hours_per_week"], adults_subset["income"] , adults_subset["workclass"], adults_subset["education"]], axis= 1)
# boolean adults census 20 qubits
adults_census_20 = pd.concat([encoded_age, encoded_hours_per_week, encoded_income, encoded_workclass, encoded_education], axis= 1)
adults_census_20['combined'] =  adults_census_20.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
# adults census 20 qubits
adults_census_20_non_boolean = pd.concat([encoded_age, encoded_hours_per_week, encoded_income_non_boolean, encoded_workclass, encoded_education], axis= 1)
adults_census_20_non_boolean['combined'] =  adults_census_20_non_boolean.apply(lambda row: ''.join(row.values.astype(str)), axis=1)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Save different datasets to csv
#--------------------------------------------------------------------------------
# save adults census 10 qubits data to csv
adults_census_10_original.to_csv(plots_path + "/adults_census_10_classical_data.csv",index = False)
# boolean data and boolean meta
adults_census_10["combined"].to_csv(plots_path + "/" + data_set_name_10 + ".csv",index = False)
save_json(column_names_10, data_spec_10, len(adults_census_10["combined"].iloc[0]), data_set_name_10, plots_path)
# non-boolean data and non-boolean meta
adults_census_10_non_boolean["combined"].to_csv(plots_path + "/" + data_set_name_10_non_boolean + ".csv",index = False)
save_json(column_names_10_non_boolean, data_spec_10_non_boolean, len(adults_census_10_non_boolean["combined"].iloc[0]), data_set_name_10_non_boolean, plots_path)

# save adults census 15 qubits data to csv
adults_census_15_original.to_csv(plots_path + "/adults_census_15_classical_data.csv",index = False)
# non-boolean data and non-boolean meta as 15 qubit dataset does not have binary category feature
adults_census_15["combined"].to_csv(plots_path + "/" + data_set_name_15 +  ".csv",index = False)
save_json(column_names_15, data_spec_15, len(adults_census_15['combined'].iloc[0]), data_set_name_15, plots_path)

# save adults census 20 qubits data to csv
adults_census_20_original.to_csv(plots_path + "/adults_census_20_classical_data.csv",index = False)
# boolean data and boolean meta
adults_census_20["combined"].to_csv(plots_path + "/" + data_set_name_20 + ".csv",index = False)
save_json(column_names_20, data_spec_20, len(adults_census_20['combined'].iloc[0]), data_set_name_20, plots_path)
#non-boolean data and non-boolean meta
adults_census_20_non_boolean["combined"].to_csv(plots_path + "/" + data_set_name_20_non_boolean + ".csv",index = False)
save_json(column_names_20_non_boolean, data_spec_20_non_boolean, len(adults_census_20_non_boolean['combined'].iloc[0]), data_set_name_20_non_boolean, plots_path)

#-------------------------------------------------------------------------------


