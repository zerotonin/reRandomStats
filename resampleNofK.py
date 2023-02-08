import random
from itertools import combinations

class getNofK:
    """
    The `getNofK` class generates a list of tuples, each containing k indices out of N.
    The indices can be either all possible unique combinations of k indices out of N, or a specified number of random combinations.
    The class also allows for the generation of the complement of the indices.
    """

    def __init__(self, data_set_a, data_set_b, combination_n, mode='resampling', max_len_possible_4_perms=10, resampling_n=100000):
        """
        Initialize the getNofK class.
        
        Args:
            data_set_a (list): The first data set
            data_set_b (list): The second data set
            combination_n (int or str): The number of random combinations to generate, or 'all' to generate all possible unique combinations
            mode (str, optional): The mode for generating the indices, either 'combinations' or 'resampling'. Defaults to 'resampling'.
            max_len_possible_4_perms (int, optional): The maximum length of the smaller data set for which all possible unique combinations will be generated. Defaults to 10.
            resampling_n (int, optional): The number of random combinations to generate if the mode is set to 'resampling'. Defaults to 100000.
        
        Attributes:
            data_set_a (list): The first data set
            data_set_b (list): The second data set
            combination_n (int or str): The number of random combinations to generate, or 'all' to generate all possible unique combinations
            mode (str): The mode for generating the indices, either 'combinations' or 'resampling'
            max_len_possible_4_perms (int): The maximum length of the smaller data set for which all possible unique combinations will be generated
            resampling_n (int): The number of random combinations to generate if the mode is set to 'resampling'
            combined_data (list): The combined data set of data_set_a and data_set_b
            combined_len (int): The length of the combined data set
            short_len (int): The length of the smaller data set
        """
        self.data_set_a = list(data_set_a)
        self.data_set_b = list(data_set_b)
        self.combination_n = combination_n
        self.max_len_possible_4_perms = max_len_possible_4_perms
        self.resampling_n = resampling_n
        self.mode = mode

        self.combined_data = self.data_set_a + self.data_set_b
        self.combined_len = self.get_combined_len()
        self.short_len = self.get_smaller_set_n()

        if self.combination_n == 'all':
            self.mode = 'combinations'
        else:
            self.mode = 'resampling'
            self.resampling_n = combination_n

        if self.mode == 'combinations' and self.max_len_possible_4_perms < self.short_len:
            self.mode = 'resampling'
            self.combination_n = self.resampling_n
    
    
    def get_combined_len(self):
        """
        This function calculates the total length of the combined data sets A and B.
        """
        self.len_a = self.get_data_len(self.data_set_a)
        self.len_b = self.get_data_len(self.data_set_b)
        return self.len_a + self.len_b
        
    def get_smaller_set_n(self):
        """
        This function returns the length of the smaller of the two data sets A and B.
        """
        if self.len_b < self.len_a:
            return self.len_b
        else:
            return self.len_a

    def get_data_len(self, data):
        """
        This function returns the length of a given data set.
        
        Args:
            data: The data set for which the length is calculated
            
        Returns:
            int: The length of the data set
        """
        return len(list(data))

    def get_all_combinations(self):
        """
        This function generates a list of all possible unique tuples, each containing k indices out of N.
        
        Returns:
            list: A list of tuples containing all possible unique combinations of k indices out of N
        """
        # Using python's built-in combinations function from the itertools module to generate a list of all possible unique tuples
        return list(combinations(range(self.combined_len), self.short_len))


    def get_unique_random_combinations(self):
        """
        This function generates a set of n unique tuples, each containing k randomly selected indices from a list of length N.
        If the function is unable to generate n unique tuples after a certain number of tries, it will return the tuples generated so far.
        
        Returns:
            list: A list of tuples containing k randomly selected indices from a list of length N.
        """
        combinations = set()
        all_indice_list = list(range(self.combined_len))
        tries = 0
        desperation = False
        while (len(combinations) < self.resampling_n) and not desperation:
            # Add a tuple of k randomly selected indices to the set
            combinations.add(tuple(sorted(random.sample(all_indice_list, self.short_len))))
            tries += 1
            if tries > self.combination_n * 10:
                desperation = True
                print(f'getNofK: get_random_combinations: Could not produce more than {len(combinations)} in {tries} tries. So I use those ')
        return [tuple(x) for x in combinations]
    
    def get_random_combinations(self):
        """
        This function generates a set of n tuples, each containing k randomly selected indices from a list of length N.
        This function can create multiple identical combinations
        
        Returns:
            list: A list of tuples containing k randomly selected indices from a list of length N.
        """
        combinations = list()
        all_indice_list = list(range(self.combined_len))
        while (len(combinations) < self.resampling_n):
            # Add a tuple of k randomly selected indices to the set
            combinations.append(tuple(sorted(random.sample(all_indice_list, self.short_len))))
        return combinations



    def complement_indices(self):
        """
        This function generates a list of tuples, each containing two sets of indices.
        The first set is generated from a list of all possible unique combinations of k indices out of N, 
        and the second set is the complement of the first set.
        """
        all_indice_set = set(range(self.combined_len))
        self.data_indices = []
        for indices_a in self.combinations:
            indices_b = tuple(all_indice_set - set(indices_a))
            self.data_indices.append((indices_a, indices_b))

    def main(self):
        """
        This function runs the main logic of the class, depending on the value of the `mode` attribute.
        If `mode` is set to 'combinations', it generates all possible unique combinations of k indices out of N.
        If `mode` is set to 'resampling', it generates n unique tuples of k randomly selected indices out of N.
        Then, it calls the `complement_indices` function to generate the complement of the indices, and sets the `combination_n` attribute to the length of the combinations.
        """
        if self.mode == 'combinations':
            self.combinations = self.get_all_combinations()
        elif self.mode == 'resampling':
            self.combinations = self.get_random_combinations()
        elif self.mode == 'resample_unique':
            self.combinations = self.get_unique_random_combinations()
        else:
            raise ValueError(f'resampleNofK:main: self.mode unknown {self.mode}')
        
        self.complement_indices()
        self.combination_n = len(self.combinations)

    

    def get_shuffled_set(self, combi_i):
        """
        This function generates two shuffled sets of data, A and B, based on a given combination of indices.
        The sets are composed of elements of the original combined data list, selected by the given combination of indices.
        
        Args:
            combi_i (tuple): A tuple of indices used to select elements from the combined data list
            
        Returns:
            tuple: A tuple containing two shuffled sets of data, A and B
        """
        # Using list comprehension to create a shuffled set of data A
        shuffle_set_a = [self.combined_data[i] for i in self.data_indices[combi_i][0]]
        # Using list comprehension to create a shuffled set of data B
        shuffle_set_b = [self.combined_data[i] for i in self.data_indices[combi_i][1]]
        return shuffle_set_a, shuffle_set_b