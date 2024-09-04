
def check_dependencies(dependencies):
    for library in dependencies:
        try:
            __import__(library)
        except ImportError:
            print(f"The library '{library}' is not installed. Please install it using pip:")
            print(f"    pip install {library}")
            sys.exit(1)  # Exit the script if a dependency is missing

# List of required libraries
dependencies = ["numpy", "pandas", "itertools", "sys", "os"]

# Check the dependencies
check_dependencies(dependencies)

# Import Libraries
import numpy as np
import pandas as pd
import itertools
import sys
import os

# Save data function
def save_data(selected_data, save_path):
    return selected_data.to_csv(save_path), print(f'Saved DataFrame to {save_path}')


PATH = input('Please enter location of simulation results: ')
PATH = os.path.normpath(PATH)

df = pd.read_csv(PATH)

SAVE_PATH = input('Please enter save path for price results: ')
SAVE_PATH = os.path.normpath(SAVE_PATH)

print('Please enter which type of simulation: \n 0: no price discimination \n 1: price discrimination \n any other number: exit')
choice = int(input())

if choice == 0:

    # Create Dictionary
    prices = [0.5, 1, 1.5, 2]

    action_index_dict = {combination: index for index, combination in enumerate(prices)}

    action_combinations = list(itertools.product(prices, repeat=2))
        
    state_index_dict = {combination: index for index, combination in enumerate(action_combinations)}

    inverted_state_index_dict = {v: k for k, v in state_index_dict.items()}

    # Modify DataFrame
    df['prices'] = [inverted_state_index_dict.get(x) for x in df.states]
    df[['price_firm_A', 'price_firm_B']] = pd.DataFrame(df['prices'].apply(lambda x: x).tolist(), columns=['price_firm_A', 'price_firm_B'])
    
    save_data(df,SAVE_PATH)

elif choice == 1:

    # Create Dictionary 
    prices = [0.5, 1, 1.5, 2]

    action_combinations_A = list(itertools.product(prices, repeat=2))
    action_combinations_B = list(itertools.product(prices, repeat=2))
    
    state_combinations = [(a, b) for a in action_combinations_A for b in action_combinations_B]
    
    state_index_dict = {combination: index for index, combination in enumerate(state_combinations)}
    inverted_state_index_dict = {v: k for k, v in state_index_dict.items()}

    # Modify DataFrame
    df['prices'] = [inverted_state_index_dict.get(x) for x in df.states]
    df[['price_left_firm_A', 'price_right_firm_A']] = pd.DataFrame(df['prices'].apply(lambda x: x[0]).tolist(), columns=['price_left_firm_A', 'price_right_firm_A'])
    df[['price_left_firm_B', 'price_right_firm_B']] = pd.DataFrame(df['prices'].apply(lambda x: x[1]).tolist(), columns=['price_left_firm_B', 'price_right_firm_B'])
    df['price_bundle_A'] = list(zip(df['price_left_firm_A'], df['price_right_firm_A']))
    df['price_bundle_B'] = list(zip(df['price_left_firm_B'], df['price_right_firm_B']))

    save_data(df,SAVE_PATH)

else:
    print('Exiting program')
    sys.exit(0)