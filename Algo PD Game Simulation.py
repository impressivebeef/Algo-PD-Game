# Import Libraries
import numpy as np
import pandas as pd
import itertools

# Save data function
def save_data(selected_data, save_path):
    return selected_data.to_csv(save_path), print(f'Saved DataFrame to {save_path}')
        

# Define functions used in simulation
class simulation:
    def signal(self,l, b=0.75):
        """
        Generate a signal based on location and signal strength

        Args:
            l (float): Location.
            b (float, optional): signal strength parameter. Defaults to 0.75.

        Returns:
            str: Signal indicating either 'L' (left) or 'R' (right).
        """
        prob_left = 1 - (0.5 + b * l)
    
        return np.random.choice(['L', 'R'], p=[prob_left, 1 - prob_left])

    def payout(self,action_A, action_B, l, v=5):
        """
        Calculate payouts and consumer surplus for two actions in the PD case.

        Args:
            action_A (tuple): Tuple containing prices for action A (left, right).
            action_B (tuple): Tuple containing prices for action B (left, right).
            l (list): List of location values.
            v (float, optional): Value parameter. Default is set at 5.

        Returns:
            tuple: Payout for action A, payout for action B, consumer surplus.
        """
        # Generate signals based on 'l' for both actions
        signals_A = np.array([sim.signal(x) for x in l])
        signals_B = np.array([sim.signal(x) for x in l])
        
        # Generate prices based on signals for both actions
        price_A = np.where(signals_A == 'L', action_A[0], action_A[1])
        price_B = np.where(signals_B == 'L', action_B[0], action_B[1])

        # Calculate surplus for both actions
        surplus_A = v - np.abs(-0.5 - l) - price_A
        surplus_B = v - np.abs(0.5 - l) - price_B
        
        # Calculate payouts and consumer surplus
        payout_A = np.sum(np.where(surplus_A > surplus_B, price_A, 0))
        payout_B = np.sum(np.where(surplus_B >= surplus_A, price_B, 0))
        consumer_surplus = np.sum(np.where(surplus_A > surplus_B, surplus_A, surplus_B))

        

        return payout_A, payout_B, consumer_surplus



    def update(self,Q_old, Q_nextstate, π, α=0.1, δ=0.95):
        """
        Update Q-values based on the payout for a given round.

        Args:
            Q_old (float): Old value of the action-state cell.
            Q_nextstate (list): List of next state action values.
            π (float): Payout for the period.
            α (float, optional): Learning rate. Defaults to 0.1.
            δ (float, optional): Discount factor. Defaults to 0.95.

        Returns:
            float: Updated Q-value.
        """
        # Calculate the new Q-value
        Q_value = (1 - α) * Q_old + α * (π + δ * np.max(Q_nextstate))
        return Q_value

    def choose_action(self,state, Q_matrix, β, period):
        """
        Choose an action based on the given state, Q-matrix, and exploration parameter.

        Args:
            state (int): Current state.
            Q_matrix (matrix): Q-matrix containing action values.
            β (float): Exploration parameter.
            period (int): Current period.

        Returns:
            int: Selected action.
        """
        # Check if the exploration threshold is met
        if np.exp(-β * period) > np.random.uniform(0, 1):
            
            # Randomly choose an action if the threshold is met
            return np.random.choice(np.arange(Q_matrix.shape[1]))  # Randomly select an action from all possible actions
        else:
            
            # Choose the action with the highest Q-value for the given state
            return np.argmax(np.array(Q_matrix[state,:]))  # Select the action with the highest Q-value for the given state

    def payout_no_PD(self,action_A, action_B, l, v=5):
        """
        Calculate payouts and consumer surplus for two actions in the no PD case.

        Args:
            action_A (tuple): Tuple containing prices for action A (left, right).
            action_B (tuple): Tuple containing prices for action B (left, right).
            l (list): List of location values.
            v (float, optional): Value parameter. Default is set at 5.

        Returns:
            tuple: Payout for action A, payout for action B, consumer surplus.
        """
        # Convert actions to numpy arrays for vectorized calculations
        action_A = np.array(action_A)
        action_B = np.array(action_B)
        
        # Calculate surpluses for both actions
        surplus_A = v - np.abs(-0.5 - l) - action_A
        surplus_B = v - np.abs(0.5 - l) - action_B
        
        # Find which action provides higher surplus at each location
        choose_A = surplus_A > surplus_B
        
        # Calculate payouts and consumer surplus
        payout_A = np.sum(np.where(choose_A, action_A, 0))
        payout_B = np.sum(np.where(~choose_A, action_B, 0))
        consumer_surplus = np.sum(np.where(choose_A, surplus_A, surplus_B))

        return payout_A, payout_B, consumer_surplus

# Define Simulation functions
sim = simulation()

class Run_Simulation:
    def simulation_PD(runs, T, M, n, β=0.0000138):
        """
        Simulation of the price discrimination case.

        Args:
            runs (int): Number of simulation runs.
            T (int): Number of periods in each simulation run.
            M (int): Convergence criteria.
            n (int): Number of consumers.
            β (float): exploration parameter. default is set at 0.0000138

        Returns:
            DataFrame: DataFrame containing simulation results.
        """
        # Set location list
        l = np.linspace(-0.5, 0.5, n)
        
        # Define the list of prices
        prices = [0.5, 1, 1.5, 2]

        # Action Dictionary 
        action_combinations = list(itertools.product(prices, repeat=2))

        action_index_dict = {combination: index for index, combination in enumerate(action_combinations)}

        inverted_action_index_dict = {v: k for k, v in action_index_dict.items()}

        # State Dictionary
        state_combinations = list(itertools.product(action_combinations, repeat=2))
        
        state_index_dict = {combination: index for index, combination in enumerate(state_combinations)}
        
        # Initialize variable arrays for storage
        states = np.empty((runs * T), dtype=int)
        period_count = np.empty((runs * T), dtype=int)
        run_count = np.empty((runs * T), dtype=int)
        convergence = np.zeros((runs * T), dtype=int)
        convergence_count_list = np.zeros((runs * T), dtype=int)
        consumer_surplus = np.zeros((runs * T), dtype=float)
        payout_A = np.zeros((runs * T), dtype=float)
        payout_B = np.zeros((runs * T), dtype=float)
        
        # Start simulation loop
        for i in range(runs):
            
            # Set Q-matrices to zero
            Q_matrix_A = np.zeros((len(prices)**4, len(prices)**2))
            Q_matrix_B = np.zeros((len(prices)**4, len(prices)**2))

            # Set convergence count to zero
            convergence_count = 0
            
            # Set initial state randomly
            state = np.random.randint(256)
            
            # Start run
            for j in range(T):
                
                # Let algorithm decide actions (explore/exploit)
                index_action_A = sim.choose_action(state, Q_matrix_A, β, j)
                index_action_B = sim.choose_action(state, Q_matrix_B, β, j)
                action_A = np.array(inverted_action_index_dict.get(index_action_A))
                action_B = np.array(inverted_action_index_dict.get(index_action_B))
                
                # Determine new state
                new_state = state_index_dict.get((tuple(action_A), tuple(action_B)))
            
                # Calculate payout
                payout_round = sim.payout(action_A, action_B, l)
                
                # Store payouts and consumer surplus
                payout_A[i * T+ j] = payout_round[0]
                payout_B[i * T+ j] = payout_round[1]
                consumer_surplus[i * T+ j] = payout_round[2]
                
                # Update Q-matrices
                Q_old_A = Q_matrix_A[state, index_action_A]
                Q_next_state_A = Q_matrix_A[new_state, :]
                Q_matrix_A[state, index_action_A] = sim.update(Q_old_A, Q_next_state_A, payout_round[0]) 
                
                Q_old_B = Q_matrix_B[state, index_action_B]
                Q_next_state_B = Q_matrix_B[new_state, :]
                Q_matrix_B[state, index_action_B] = sim.update(Q_old_B, Q_next_state_B, payout_round[1]) 
                
                # Check if new state == old state
                if state == new_state:
                    convergence_count += 1
                else:
                    convergence_count = 0 
                convergence_count_list[i * T + j] = convergence_count
                
                # Update old state = new state
                state = new_state
                states[i * T + j] = state
                
                # Store the period count and run number
                period_count[i * T + j] = j
                run_count[i * T + j] = i
                
                # Print period and run number (overwrite previous print)
                print(f'run: {i}, period: {j}', end='\r')
                
                # Check if convergence count == convergence criteria
                if convergence_count == M:
                    convergence[i * T + j] = 1
                    break
                    
        # Create DataFrame with simulation results
        df = pd.DataFrame({'run': run_count, 'period': period_count, 'convergence': convergence, 
                        'convergence count': convergence_count_list, 'states': states, 
                        'consumer surplus': consumer_surplus, 'profit firm A': payout_A, 
                        'profit firm B': payout_B})
        
        return df



    def simulation_no_PD(runs, T, M, n, β=0.000069):
        """
        Simulation of the no price discrimination case.

        Args:
            runs (int): Number of simulation runs.
            T (int): Number of periods in each simulation run.
            M (int): Convergence criteria.
            n (int): Number of consumers.
            β (float): exploration parameter. default is set at 0.000069

        Returns:
            DataFrame: DataFrame containing simulation results.
        """
        # Set location list
        l = np.linspace(-0.5, 0.5, n)
        
        # Define the list of prices
        prices = [0.5,1,1.5,2]

        # Action Dictionary 
        action_index_dict = {combination: index for index, combination in enumerate(prices)}

        inverted_action_index_dict = {v: k for k, v in action_index_dict.items()}

        # State Dictionary
        action_combinations = list(itertools.product(prices, repeat=2))
        
        state_index_dict = {combination: index for index, combination in enumerate(action_combinations)}

        # Initialize variable lists for storage
        states = []
        period_count = []
        run_count = []
        convergence = []
        convergence_count_list = []
        consumer_surplus = []
        payout_A = []
        payout_B = []
        
        # Start simulation loop
        for i in range(runs):
            
            # Set Q-matrices to zero
            Q_matrix_A = np.zeros((len(prices)**2, len(prices)))
            Q_matrix_B = np.zeros((len(prices)**2, len(prices)))

            # Set convergence count to zero
            convergence_count = 0
            
            # Set initial state randomly
            state = np.random.randint(16)
            print(state)
            # Start run
            for j in range(T):
                
                # Let algorithm decide actions (explore/exploit)
                index_action_A = sim.choose_action(state, Q_matrix_A, β, j)
                index_action_B = sim.choose_action(state, Q_matrix_B, β, j)
                action_A = inverted_action_index_dict.get(index_action_A)
                action_B = inverted_action_index_dict.get(index_action_B)
                
                # Determine new state
                new_state = state_index_dict.get((action_A,action_B))
            
                # Calculate payout
                payout_round = sim.payout_no_PD(action_A, action_B, l)
                
                # Store payouts and consumer surplus
                payout_A.append(payout_round[0])
                payout_B.append(payout_round[1])
                consumer_surplus.append(payout_round[2])
                
                # Update Q-matrices
                Q_old_A = Q_matrix_A[state, index_action_A]
                Q_next_state_A = Q_matrix_A[new_state, :]
                Q_matrix_A[state, index_action_A] = sim.update(Q_old_A, Q_next_state_A, payout_round[0]) 
                
                Q_old_B = Q_matrix_B[state, index_action_B]
                Q_next_state_B = Q_matrix_B[new_state, :]
                Q_matrix_B[state, index_action_B] = sim.update(Q_old_B, Q_next_state_B, payout_round[1]) 
                
                # Check if new state == old state
                if state == new_state:
                    convergence_count += 1
                else:
                    convergence_count = 0 
                convergence_count_list.append(convergence_count)
                
                # Update old state = new state
                state = new_state
                states.append(state)
                
                # Store the period count and run number
                period_count.append(j)
                run_count.append(i)
                
                # Print period and run number (overwrite previous print)
                print(f'run: {i}, period: {j}', end='\r')
                
                # Check if convergence count == convergence criteria
                if convergence_count == M:
                    convergence.append(1)
                    break
                    
                # Store convergence = False
                convergence.append(0)
        
        # Create DataFrame with simulation results
        df = pd.DataFrame({'run': run_count, 'period': period_count, 'convergence': convergence, 
                        'convergence count': convergence_count_list, 'states': states, 
                        'consumer surplus': consumer_surplus, 'profit firm A': payout_A, 
                        'profit firm B': payout_B})
        
        return df


# Run and store simulations 

PATH = ''

df_no_pd = Run_Simulation.simulation_no_PD(runs=100,T=1000000,M=10000,n=100, β=0.0000092)
save_data(df_no_pd, PATH)

df_pd = Run_Simulation.simulation_PD(runs=100,T=1000000,M=10000,n=100, β=0.0000092)
save_data(df_pd, PATH)

