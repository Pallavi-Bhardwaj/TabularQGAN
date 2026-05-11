import optax as optax
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

import pennylane as qml
import jax
import jax.numpy as jnp
import math
# import numpy as np
from builtins import sorted

from functools import partial
from itertools import combinations

def create_register_spec(data_spec):
    sorted_registers = []
    # extract numerical registers from data spec
    numeric_registers = filter(lambda x: (x[0] =='n') or (x[0] =='b') , data_spec)
    # sum extracted numerical register qubits
    number_of_numeric_qubits = sum(map((lambda x: int(x[1:])), numeric_registers))

        # extract categorical registers from data spec
    cat_registers = list(filter(lambda x: x[0] =='c', data_spec))
    # sort extracted categorical register qubits
    sorted_cat_register = sorted(map((lambda x: int(x[1:])), cat_registers), reverse=True)

    # append summed numerical and sorted categorical qubits
    if number_of_numeric_qubits > 0:
        sorted_registers.append('n' + str(number_of_numeric_qubits))
    if len(sorted_cat_register) > 0:
        [sorted_registers.append('c' + str(i)) for i in sorted_cat_register]
    return sorted_registers

# Since we're using a qnode with sampling, we cannot use autograd. Instead, use this (analogously to the old implementation)
@partial(jax.jit, static_argnames=["discriminator"])
def compute_gradient_JAX(samples, discriminator, discriminator_weights):
    def criterion(outputs):
        return (-1.0 * jnp.log(outputs) ).mean()

    gradient = []
    for i in range(0, len(samples), 2):
        forward_fake = samples[i]
        backward_fake = samples[i + 1]

        forward_output = discriminator.apply(discriminator_weights, forward_fake).flatten()
        backward_output = discriminator.apply(discriminator_weights, backward_fake).flatten()

        forward_diff = criterion(forward_output)
        backward_diff = criterion(backward_output)
        gradient.append(1 / 2 * (forward_diff - backward_diff))

    return jnp.array(gradient)


    # Only the circuit with one copula block is implemented for now.
    @partial(jax.jit, static_argnames=["n_shots"])
    def qnode_with_variable_random_key(key, weights, n_shots):
        dev = qml.device("default.qubit.jax", prng_key=key, wires=n_qubits, shots=n_shots)
        qnode = qml.QNode(qnode_fn, dev, diff_method=None, interface="jax")
        return qnode(weights)

    # Create a dummy device and a dummy qnode since we need to specify the random key and the number of shots
    # in order to create a device, then the qnode and then run the circuit to calculate the number of parameters by tracing
    # through the computation. We need to do this as what we need inside the actual model handlers is a function
    # with variable random key and shots (qnode_with_variable_random_key_and_shots) which internally creates the device,
    # qnode and runs it, meaning that no QNode object actually exists at this point in the code. Both the dummy_qnode and
    # the qnode_with_variable_random_key_and_shots function actually use the same qnode_fn though (which specifies the
    # circuit)
    dummy_device = qml.device("default.qubit.jax", prng_key=jax.random.PRNGKey(1), wires=n_qubits, shots=1)
    dummy_qnode = qml.QNode(qnode_fn, dummy_device, diff_method=None, interface="jax")
    # Need to pass in a dummy array jnp.zeros((1,)) to get the number of trainable parameters since in some cases this
    # number actually depends on the input array (because it is inferred from it), e.g. in templates like
    return qnode_with_variable_random_key, qml.specs(dummy_qnode)(jnp.zeros((1,)))["num_trainable_params"]


def discrete_standard_circuit_JAX(n_qubits, n_registers, circuit_depth):
    def standard_parametric(weights, n_qubits, n_registers, circuit_depth):
        count = 0
        for _ in range(circuit_depth):
            for k in range(n_qubits):
                qml.RY(weights[count], wires=k)
                count += 1
            for k in range(n_qubits-1):
                qubit_1 = k
                qubit_2 = k + 1
                qml.IsingYY(weights[count], wires=[qubit_1, qubit_2])
                count += 1

            for k in range(n_qubits-1):
                control_qubit = k
                target_qubit = k+1
                qml.CRY(weights[count], wires=[control_qubit, target_qubit])
                count += 1

    def qnode_fn(weights):
        standard_parametric(weights, n_qubits, n_registers=n_registers,
                            circuit_depth=circuit_depth)
        return qml.sample()

    def qnode_with_variable_random_key_and_shots(device_parameters, key, weights):
        if device_parameters["name"] == "default.qubit.jax":
            device_parameters["prng_key"] = key
        # else is using the preconfigured device (class attribute)
        dev = qml.device(**device_parameters)
        qnode = qml.QNode(qnode_fn, dev, diff_method=None, interface="jax")
        return qnode(weights)
    # Create a dummy device and a dummy qnode since we need to specify the random key and the number of shots
    # in order to create a device, then the qnode and then run the circuit to calculate the number of parameters by tracing
    # through the computation. We need to do this as what we need inside the actual model handlers is a function
    # with variable random key and shots (qnode_with_variable_random_key_and_shots) which internally creates the device,
    # qnode and runs it, meaning that no QNode object actually exists at this point in the code. Both the dummy_qnode and
    # the qnode_with_variable_random_key_and_shots function actually use the same qnode_fn though (which specifies the
    # circuit)
    dummy_device = qml.device("default.qubit.jax", prng_key=jax.random.PRNGKey(1), wires=n_qubits, shots=1)
    dummy_qnode = qml.QNode(qnode_fn, dummy_device, diff_method=None, interface="jax")
    # Need to pass in a dummy array jnp.zeros((1,)) to get the number of trainable parameters since in some cases this
    # number actually depends on the input array (because it is inferred from it), e.g. in templates like
    # qml.StronglyEntanglingLayers
    fig, ax = qml.draw_mpl(dummy_qnode)(jnp.zeros((1,)))
    return qnode_with_variable_random_key_and_shots, qml.specs(dummy_qnode)(jnp.zeros((1,)))["num_trainable_params"]

def discrete_tabular_circuit_JAX(n_qubits, circuit_depth, data_spec = None):
    #register_spec should be something like ['c2', 'n4', 'b1']  for 2 catagorical, 4 numerical, 1 boolean

    add_given_rotation = True
    fully_connected = False
    minimally_controlled_gate = True

    def calculate_params(register_spec, circuit_depth=1):
        num_numerical_qubits = 0
        num_largest_cat_qubits = 0
        single_exc_cat_gates = 0
        controlled_single_exc_cat_gates = 0
        num_of_std_circuit_layers_ = 0

        for index, reg_spec in enumerate(register_spec):
            if register_spec[index][0] == 'n':
                num_numerical_qubits = int(register_spec[index][1:])
                num_of_std_circuit_layers_ = 1
            elif register_spec[index][0] == 'c':
                single_exc_cat_gates = sum(int(x[1:]) - 1 for x in register_spec[index:])
                if fully_connected:
                    controlled_single_exc_cat_gates = sum(int(x[1:]) for x in register_spec[index:-1])
                elif minimally_controlled_gate:
                    controlled_single_exc_cat_gates = sum(+1 for x in register_spec[index:-1])
                else:
                    controlled_single_exc_cat_gates = sum(int(x[1:]) - 1 for x in register_spec[index+1:])
                num_largest_cat_qubits = int(register_spec[index][1:])
                break

        if num_numerical_qubits > 0 and num_largest_cat_qubits > 0:
            num_of_std_circuit_layers_ = math.ceil((num_largest_cat_qubits - 1) / num_numerical_qubits)

        num_cat_gates = single_exc_cat_gates + controlled_single_exc_cat_gates

        numerical_parameters = num_of_std_circuit_layers_ * num_standard_parameters
        total_num_parameters  = circuit_depth * (numerical_parameters + num_cat_gates)
        return total_num_parameters

    # numerical_layers_needed = math.ceil(int(register_spec[1][1]) /int(register_spec[0][1]))
    register_spec = create_register_spec(data_spec)
    print(f'data spec {data_spec}')
    print(f'register spec {register_spec}')
    _, num_standard_parameters = discrete_standard_circuit_JAX(int(register_spec[0][1:]), 1, 1)
    total_num_parameters = calculate_params(register_spec, circuit_depth)

    def standard_subcircuit(weights, n_qubits, qubit_offset):

        # if circuit_depth != 1:
        #     raise ValueError("This circuit only supports circuit_depth = 1, other depths may be implemented later.")

        shift = 0
        count = 0
        # for _ in range(circuit_depth):
        for k in range(n_qubits):
            qml.RY(weights[k + shift], wires=k)
            count += 1
        for k in range(n_qubits - 1):
            qubit_1 = k
            qubit_2 = k + 1
            qml.IsingYY(weights[k + shift + n_qubits], wires=[qubit_1 + qubit_offset, qubit_2 + qubit_offset])
            count += 1

        for k in range(n_qubits - 1):
            control_qubit = k
            target_qubit = k + 1
            qml.CRY(weights[k + shift + n_qubits], wires=[control_qubit + qubit_offset, target_qubit + qubit_offset])
            count += 1

    def qnode_fn(weights):
        # marks the start of current register
        # print(f'{weights.shape} weights shape')
        weights_idx = 0
        numerical_qubits= int(register_spec[0][1:])
        for layer in range(circuit_depth):
            qubit_offset = 0
            previous_qubit_offset = 0
        #  take the register spec and order in  first with  numeric  then catagorial and then sort the cat registers largest  to smallest.
        # Use this order to combine the circuit.
        # Use as many standard circuit layers as needed for the largest cat, and use numerial  register as control qubits for largest cat. 
        # Then for each cat register  condition each single excitment gate on a control qubit in the next register  (n-1 control quibts needed per layer)

            for idx, entry in enumerate(register_spec):

                qubits_in_current_reg_spec = int(entry[1:])

                ## Check if string ends with "n", these if conditions are good to have because it checks that each entry  in register spec should start with c or n
                # if not then a value error is thrown at the end
                if entry[0] == "n":
                    standard_subcircuit(
                    weights[weights_idx:weights_idx + num_standard_parameters],
                        numerical_qubits,
                        # circuit_depth=1,
                        qubit_offset=qubit_offset,
                    )
                    weights_idx += num_standard_parameters

                # if it is categorical then entangle it with previous register
                elif entry[0] == "c":
                    if idx > 0:
                        qubits_previous_reg_spec = int(register_spec[idx - 1][1:])
                    else:
                        qubits_previous_reg_spec = 0
                    # copy of start index of current register , valid only for updating in for loop of single excitation
                    start_index_current_register = qubit_offset
                    # number of conditional single excitations required to entangle categorical register with the numerical register
                    no_of_single_excitations_required = qubits_in_current_reg_spec - 1

                    # apply conditional single excitation gates on first/largest categorical variable, largest categorical variable will always be on index 1 in register_spec
                    if layer == 0:
                        qml.RX(jnp.pi, wires=qubit_offset)

                    if add_given_rotation:
                        # First add given rotations within the categorical block
                        weights_idx = single_excitations(no_of_single_excitations_required,
                                                            qubit_offset,
                                                            start_index_current_register, weights, weights_idx)

                    # entangle first/larget cat register with previous numerical register
                    if register_spec[0][0] == 'n' and idx == 1:
                        if minimally_controlled_gate:
                            no_of_single_excitations_required = 1
                        weights_idx = entangle_cat_with_numerical_reg(no_of_single_excitations_required,
                                                                    previous_qubit_offset,
                                                                    qubits_in_current_reg_spec, qubits_previous_reg_spec,
                                                                    start_index_current_register, weights, weights_idx)

                    # entangle rest of cat register with previous cat register
                    elif idx >=1:
                        if fully_connected:
                            no_of_single_excitations_required = qubits_previous_reg_spec
                            weights_idx = full_cond_single_excitations(no_of_single_excitations_required ,previous_qubit_offset, qubits_in_current_reg_spec,
                                                                    start_index_current_register, weights, weights_idx)
                        elif minimally_controlled_gate:
                            weights_idx = cond_single_excitations(1 ,previous_qubit_offset, qubits_in_current_reg_spec,
                                                                    start_index_current_register, weights, weights_idx)
                        else:
                            weights_idx = cond_single_excitations(no_of_single_excitations_required ,previous_qubit_offset, qubits_in_current_reg_spec,
                                                                    start_index_current_register, weights, weights_idx)
                else:
                    raise ValueError(f"Invalid entry {entry} encountered in register_spec. Must start with 'c' or 'n'")

                previous_qubit_offset = qubit_offset
                qubit_offset = qubit_offset + qubits_in_current_reg_spec

        return qml.sample()

    def entangle_cat_with_numerical_reg(no_of_single_excitations_required, previous_qubit_offset,
                                        qubits_in_current_reg_spec, qubits_previous_reg_spec,
                                        start_index_current_register, weights, weights_idx, numerical_register=False):

        # number of conditional single excitations which will be left over and standard circuits needs to be inserted in between
        left_over_single_excitations = max(0, no_of_single_excitations_required - qubits_previous_reg_spec)
        single_excitations_possible = no_of_single_excitations_required - left_over_single_excitations

        while (single_excitations_possible > 0):

            weights_idx = cond_single_excitations(single_excitations_possible, previous_qubit_offset,
                                                        qubits_in_current_reg_spec, start_index_current_register,
                                                        weights, weights_idx)

            # apply standard circuit between single excitation gates for left over qubits
            if left_over_single_excitations > 0 and numerical_register:
                standard_subcircuit(
                    weights[weights_idx:weights_idx + num_standard_parameters],
                    qubits_previous_reg_spec,
                    # circuit_depth=1,
                    qubit_offset=0,
                )
                weights_idx += num_standard_parameters 
            no_of_single_excitations_required = left_over_single_excitations
            left_over_single_excitations = max(0, no_of_single_excitations_required - qubits_previous_reg_spec)
            single_excitations_possible = no_of_single_excitations_required - left_over_single_excitations

        return weights_idx

    def cond_single_excitations(no_of_excitations_required, previous_qubit_offset, qubits_in_current_reg_spec,
                                      start_index_current_register, weights, weights_idx):
        # number of single excitations possible, loop only this number by maintaining start index of previous register for control qubits/wires and start index of current register(qubi_offset)
        for i in range(previous_qubit_offset, previous_qubit_offset + no_of_excitations_required):
            qml.ctrl(qml.SingleExcitation, control=i)(
                weights[weights_idx],
                wires=[start_index_current_register, start_index_current_register + 1])
            # update the current register wires in for loop
            start_index_current_register += 1
            weights_idx +=1
        return weights_idx

    def full_cond_single_excitations(no_of_excitations_required, previous_qubit_offset, qubits_in_current_reg_spec,
                                      start_index_current_register, weights, weights_idx):
        initial_index = start_index_current_register
        current_reg_threshold = (start_index_current_register + qubits_in_current_reg_spec) -1
        # number of single excitations possible, loop only this number by maintaining start index of previous register for control qubits/wires and start index of current register(qubi_offset)
        for i in range(previous_qubit_offset, previous_qubit_offset + no_of_excitations_required):
            if start_index_current_register < current_reg_threshold:
                qml.ctrl(qml.SingleExcitation, control=i)(
                    weights[weights_idx],
                    wires=[start_index_current_register, start_index_current_register + 1])
                # update the current register wires in for loop
                start_index_current_register += 1
                weights_idx +=1
            else:
                start_index_current_register = initial_index
                qml.ctrl(qml.SingleExcitation, control=i)(
                    weights[weights_idx],
                    wires=[start_index_current_register, start_index_current_register + 1])
                # update the current register wires in for loop
                start_index_current_register += 1
                weights_idx +=1
        return weights_idx

    def single_excitations(no_of_excitations_required, previous_qubit_offset,
                                      start_index_current_register, weights, weights_idx):
        initial_index = start_index_current_register
        # number of single excitations possible, loop only this number by maintaining start index of previous register for control qubits/wires and start index of current register(qubi_offset)
        for i in range(previous_qubit_offset, previous_qubit_offset + no_of_excitations_required):
            qml.SingleExcitation(
                weights[weights_idx],
                wires=[initial_index, start_index_current_register + 1])
            # update the current register wires in for loop
            start_index_current_register += 1
            weights_idx +=1
        return weights_idx


    
    # Need to pass in a dummy array jnp.zeros((1,)) to get the number of trainable parameters since in some cases this
    # number actually depends on the input array (because it is inferred from it), e.g. in templates like
    # qml.StronglyEntanglingLayers

    def qnode_with_variable_random_key_and_shots(key, weights, n_shots):
    # Only the circuit with one copula block is implemented for now.
        dev = qml.device("default.qubit.jax", prng_key=key, wires=n_qubits, shots=n_shots)
        qnode = qml.QNode(qnode_fn, dev, diff_method=None, interface="jax")
        return qnode(weights)

    dummy_device = qml.device("default.qubit.jax", prng_key=jax.random.PRNGKey(1), wires=n_qubits, shots=1)
    dummy_qnode = qml.QNode(qnode_fn, dummy_device, diff_method=None, interface="jax")
   # fig, ax = qml.draw(dummy_qnode)(np.random.random(total_num_parameters,))
   # fig.savefig("./" + 'circuit_generator.pdf')
    return qnode_with_variable_random_key_and_shots, total_num_parameters, dummy_qnode
