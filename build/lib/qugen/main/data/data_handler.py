# Copyright 2023 QUTAC, BASF Digital Solutions GmbH, BMW Group, 
# Lufthansa Industry Solutions AS GmbH, Merck KGaA (Darmstadt, Germany), 
# Munich Re, SAP SE.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import jax
import jax.numpy as jnp 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from  qugen.main.generator.quantum_circuits.discrete_generator_pennylane import create_register_spec
from qugen.main.data.integral_transform import emp_integral_trans
from typing import List
from sklearn.pipeline import Pipeline

# Transformation classes

class MinMaxNormalizer:
    def __init__(self, reverse_lookup = None, epsilon = 0):
        self.reverse_lookup = reverse_lookup
        self.epsilon = epsilon

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.min = data.min()
        self.max = data.max() - data.min()
        data = (data - self.min) / self.max
        self.reverse_lookup = (self.min, self.max)
        return data / (1 + self.epsilon)

    def transform(self, data: np.ndarray) -> np.ndarray:
        min = data.min()
        max = data.max() - data.min()
        data = (data - min) / max
        return data / (1 + self.epsilon)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = data * (1 + self.epsilon)
        self.min, self.max = self.reverse_lookup
        return data * self.max + self.min


class PITNormalizer():
    def __init__(self, reverse_lookup = None, epsilon = 0):
        self.reverse_lookup = reverse_lookup
        self.epsilon = epsilon

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data)
        epit = df.copy(deep=True).transpose()
        reverse_epit_lookup = epit.copy(deep=True)

        epit.values[::] = [emp_integral_trans(row) for row in epit.values]
        epit = epit.transpose()
        reverse_epit_lookup.values[::] = [np.sort(row) for row in reverse_epit_lookup.values]

        df = epit.copy()
        self.reverse_lookup = reverse_epit_lookup.values
        self.reverse_lookup = jnp.array(self.reverse_lookup)
        return df.values / (1 + self.epsilon)

    def transform(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data)
        epit = df.copy(deep=True).transpose()
        reverse_epit_lookup = epit.copy(deep=True)

        epit.values[::] = [emp_integral_trans(row) for row in epit.values]
        epit = epit.transpose()
        reverse_epit_lookup.values[::] = [np.sort(row) for row in reverse_epit_lookup.values]

        df = epit.copy()
        return df.values / (1 + self.epsilon)

    def _reverse_emp_integral_trans_single(self, values: jnp.ndarray) -> List[float]:
    # assumes non ragged array
        values = values * (jnp.shape(self.reverse_lookup)[1] - 1)
        rows = jnp.shape(self.reverse_lookup)[0]
    # if we are an integer do not use linear interpolation
        valuesL = jnp.floor(values).astype(int)
        valuesH = jnp.ceil(values).astype(int)
    # if we are an integer then floor and ceiling are the same
        isIntMask = 1 - (valuesH - valuesL)
        rowIndexer = jnp.arange(rows)
        resultL = self.reverse_lookup[([rowIndexer], [valuesL])]  # doing 2d lookup as [[index1.row, index2.row],[index1.column, index2.column]]
        resultH = self.reverse_lookup[([rowIndexer], [valuesH])]  # where 2d index tuple would be (index1.row, index1.column)
    # lookup int or do linear interpolation
        return resultL * (isIntMask + values - valuesL) + resultH * (valuesH - values)    

    @partial(jax.jit, static_argnums=(0,))
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data = data * (1 + self.epsilon)
        res = jax.vmap(self._reverse_emp_integral_trans_single)(data)
        # res = [self._reverse_emp_integral_trans_single(row) for row in data]
        return res[:, 0, :]

class TabularDataTransformer:
    def __init__(self, data_spec, reverse_lookup= None):
        self.data_spec = data_spec
        self.reverse_lookup = reverse_lookup

    #register_spec should be something like ['c2', 'n4']  for 2 catagorical, 4 numerical
    # def create_register_spec(self):
    #     sorted_registers = []
    #     # extract numerical registers from data spec
    #     numeric_registers = filter(lambda x: x[0] =='n', self.data_spec)
    #     # sum extracted numerical register qubits
    #     number_of_numeric_qubits = sum(map((lambda x: int(x[1:])), numeric_registers))

        # extract binary categorical registers from data spec
        binary_cat_registers = list(filter(lambda x: x[0] =='b', self.data_spec))


         # extract categorical registers from data spec
        cat_registers = list(filter(lambda x: x[0] =='c', self.data_spec))
        # sort extracted categorical register qubits
        sorted_cat_register = sorted(map((lambda x: int(x[1:])), cat_registers), reverse=True)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        index_list = []
        numeric_block = []
        categorical_block = []
        reverse_map = []
        unsorted_cat_registers = {}
        
        if len(self.data_spec) == 1:
            self.reverse_lookup = [1]
            return data
            

        for ds in self.data_spec:
            index_list.append(int(ds[1:]))
        cum_index_list = np.cumsum(index_list)
        split_data = np.split(data, cum_index_list, axis=1)
        data_ref = list(zip(self.data_spec, split_data, range(len(self.data_spec))))

        for ds in self.data_spec:
            if ds[0] == 'n' or ds[0] == 'b':
                for x in data_ref:
                    if x[0] == ds:
                        numeric_block.append(x[1])
                        reverse_map.append(x[2])
                        data_ref.remove(x)
                        break
        # all that is left is cat data           
        data_ref.sort(key=lambda x: x[0], reverse=True)
        # process categorical bits
        
        for k in data_ref:
            categorical_block.append(k[1])
            reverse_map.append(k[2])

        if len(categorical_block) > 0:
            categorical_array = np.concatenate([np.array(i) for i in categorical_block], axis =1)

        if len(numeric_block) > 0 and len(categorical_block) > 0:
            data_block = np.hstack((np.concatenate(numeric_block, axis=1),categorical_array))
        elif len(numeric_block) > 0 and len(categorical_block) == 0:
            data_block = np.concatenate(numeric_block, axis=1)
        else:
            data_block = categorical_array
        self.reverse_lookup = reverse_map
        return data_block

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        
        index_list = []
        categorical_block = []
        numeric_array = []
        block = [] 
        register_spec = create_register_spec(self.data_spec)
        if len(register_spec) == 1:
            return data
        # [ b1,n7 c3] > n8, c3
        #[b1,c3, c4] > n1, c4, c3 
        #first find all the numeric indices in the data_spec 
        if register_spec[0][0] == 'n':
            index_list = [int(i[1:]) for i in self.data_spec if (i[0]=='n') or  (i[0]=='b')] # 7 , 1 
        else: 
            index_list = []

        # then add the catagorical indexas from the register spec 
        for rs in register_spec[1:]:
            index_list.append(int(rs[1:]))  # 7, 1, 3 
        cum_index_list = np.cumsum(index_list) 
           
        split_data = np.split(data, cum_index_list, axis=1)
        split_data_reordered = list(zip(split_data, self.reverse_lookup))
        split_data_reordered.sort(key=lambda x: x[1])
        data_block = np.concatenate([np.array(i[0]) for i in split_data_reordered], axis=1)

        return data_block

    def decode_tabular_data(data:pd.DataFrame, data_spec, column_names):
        last_index = 0
        headings = []
        for idx, spec in enumerate(data_spec):
            column = column_names[idx]
            # process numerical
            if spec[0] == 'n':
                headings.append(column)
                data.loc[:,str(column)] = ''
                for index, value in enumerate(
                        data.iloc[:, last_index:last_index + int(spec[1:])].apply(lambda row: ''.join(row.values.astype(str)), axis=1)):
                    data.loc[index, column] = (int(value, 2))
                data[column] = data[column].astype('float')


            else:
                col_name = list(column.keys())[0]
                categories = list(column.values())[0]
                data.loc[:,col_name] = ''
                headings.append(col_name)
                for index, value in enumerate(
                        data.iloc[:, last_index:last_index + int(spec[1:])].apply(
                            lambda row: ''.join(row.values.astype(str)), axis=1)):
                    for cat in categories:
                        for cat_name, cat_encoding in cat.items():
                            if value == cat_encoding:
                                data.loc[index,col_name] = cat_name
            last_index = last_index + int(spec[1:])
        data = data[headings]
        return data

    def transform_classical_gan_data(self, data, numerical_columns, categorical_columns):
        num_data = []
        cat_data = []
        num_pipeline = None
        cat_pipeline = None

        if len(numerical_columns) != 0:
            num_pipeline = Pipeline([
                ("scaler", MinMaxScaler()),
            ])
            num_data = num_pipeline.fit_transform(data[numerical_columns])
        if len(categorical_columns) != 0:
            cat_pipeline = Pipeline([
                ("onehotencoder", OneHotEncoder(handle_unknown='ignore', drop='if_binary')),
            ])
            cat_data = cat_pipeline.fit_transform(data[categorical_columns]).toarray()
        if len(num_data) != 0 and len(cat_data) != 0:
            training_data = pd.DataFrame(np.concatenate([num_data, cat_data], axis=1))

        elif len(num_data) == 0:
            training_data = pd.DataFrame(cat_data)
        else:
            training_data = pd.DataFrame(num_data)

        return training_data, num_pipeline, cat_pipeline
    def inverse_transform_classical_gan_data(self, num_pipeline ,cat_pipeline, data, numerical_columns,
                                             categorical_columns):
        num_index = len(numerical_columns) 

        num_data = num_pipeline.inverse_transform(data[:,:num_index])
        cat_data = cat_pipeline.inverse_transform(data[:,num_index:])

        decoded_samples = pd.DataFrame(np.concatenate([num_data, cat_data], axis=1))
        feature_names = np.concatenate([numerical_columns, categorical_columns])
        decoded_samples.columns = feature_names

        # for idx, feature in feature_names:
        #     decoded_samples[feature] = decoded_samples.iloc[idx]
        return decoded_samples

def load_data(data_set, n_train=50000, n_test=10000):
    raw_data = np.load(data_set + '.npy')
    train = raw_data[:n_train]
    return train, []

