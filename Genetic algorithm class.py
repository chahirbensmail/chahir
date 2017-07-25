import random
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)
from pandas import read_csv
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""

# Example of running - maximizing based on r2 coefficient and a linear regression model. 
df = pd.read_csv("C:/Users/bensmail/Documents/Python Scripts/Toolkit/ds-toolkit/test_data/boston.csv")
ga = GeneticAlgorithm(df, population_size = 10,
                      iterations = 100,
                      crossover_probability = 0.5,
                      mutation_probability = 0.05,
                      split_data = 0.33)
ga.create_initial_population("pearson", 100)
ga.run()
print(ga.best_individual)
print(ga.features_selected)

"""

class GeneticAlgorithm(object):
    def __init__(self,
                dataframe,
                population_size=10,
                iterations = 100,
                crossover_probability = 0.5,
                mutation_probability = 0.2,
                split_data = 0.33):
        """Instantiate the Genetic Algorithm.
        - dataframe: input data to the Genetic Algorithm
        - int population_size: size of population
        - int iterations: number of iterations to evolve
        - float crossover_probability: probability of crossover operation
        - float mutation_probability: probability of mutation operation
        """
        self.dataframe = dataframe
        self.population_size = population_size
        self.iterations = iterations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.individual_length = self.dataframe.shape[1]-1
        self.split_data = split_data
        # Splitting data
        X = df.iloc[:,0:(df.shape[1]-1)]
        y = df.iloc[:,(df.shape[1]-1)]
        self.X_int, self.X_ext, self.y_int, self.y_ext = train_test_split(X, y, test_size = float(self.split_data))
        
        # Calculate fitness based on r2 coefficient of regression
        def calcul_fitness(population):
            fitness_intern = []
            fitness_extern = []
            for j in range(population.shape[0]):
                selected_predictors = []
                for i in range(len(population[j])):
                    if population[j][i]==1:
                        selected_predictors.append(i)
                model_intern = LinearRegression()
                model_intern.fit(self.X_int.iloc[:,selected_predictors], self.y_int)
                model_extern = LinearRegression()
                model_extern.fit(self.X_ext.iloc[:,selected_predictors], self.y_ext)
                fitness_intern.append(model_intern.score(self.X_int.iloc[:,selected_predictors], self.y_int))
                fitness_extern.append(model_extern.score(self.X_ext.iloc[:,selected_predictors], self.y_ext))
            return fitness_intern, fitness_extern
        
        def calcul_individual_fitness(individual):
            selected_predictors = []
            for i in range(len(individual)):
                if individual[i]==1:
                    selected_predictors.append(i)
            model_intern = LinearRegression()
            model_intern.fit(self.X_int.iloc[:,selected_predictors], self.y_int)
            individual_fitness_intern = model_intern.score(self.X_int.iloc[:,selected_predictors], self.y_int)
            model_extern = LinearRegression()
            model_extern.fit(self.X_ext.iloc[:,selected_predictors], self.y_ext)
            individual_fitness_extern = model_extern.score(self.X_int.iloc[:,selected_predictors], self.y_int)
            return individual_fitness_intern, individual_fitness_extern
        
        def best_individual_intern(population):
            fitness_intern = calcul_fitness(population)[0]
            max_fitness_intern = max(fitness_intern)
            best_individual = population[np.argmax(fitness_intern),:]
            return max_fitness_intern, best_individual
        
        def best_individual_extern(population):
            fitness_extern = calcul_fitness(population)[1]
            max_fitness_extern = max(fitness_extern)
            best_individual = population[np.argmax(fitness_extern),:]
            return max_fitness_extern, best_individual
        
        def worst_individual_extern(population):
            fitness_extern = calcul_fitness(population)[1]
            min_fitness_extern = min(fitness_extern)
            worst_individual = population[np.argmin(fitness_extern),:]
            index_worst_individual = np.argmin(fitness_extern)
            return min_fitness_extern, worst_individual, index_worst_individual

        def random_selection(population):
            return random.choice(population)
        
        def roulette_selection(population, fitness_intern):
            parent2_selected = False
            # Roulette selection
            # Parent 1
            fitness_intern = [0] + fitness_intern
            fitness_intern_sum = np.sum(fitness_intern)
            proba = fitness_intern/fitness_intern_sum
            index_proba = np.row_stack((np.array([x for x in range(11)]), (np.cumsum([proba]))))
            rand = np.random.rand(1)
            for i in range(10):
                if rand>=index_proba[1,i] and rand<index_proba[1,i+1]:
                    selected_individual_1 = i
                    parent1 = list(population[selected_individual_1,:])
            # Parent 2
            while not(parent2_selected):
                rand = np.random.rand(1)
                for i in range(10):
                    if rand>=index_proba[1,i] and rand<index_proba[1,i+1] and i!=selected_individual_1:
                        selected_individual_2 = i
                        parent2 = list(population[selected_individual_2,:])
                        parent2_selected = True
            return parent1, parent2
        
        def basic_crossover(father1,father2, index):
            # Child 1
            child1 = father1[0:index]+father2[index:]
            # Child 2
            child2 = father2[0:index]+father1[index:]
            return child1, child2
        
        def rand_crossover(father1,father2, crossover_probability): 
            child=[]
            for i in range(len(father1)):
                prob=np.random.rand(1)
                if prob < crossover_probability : 
                    child.append(father1[i])
                else : 
                    child.append(father2[i])
            return child

        def mutate(individual):
            # Reverse the bit of a random index in an individual
            mutate_index = random.randrange(len(individual))
            individual[mutate_index] = (0, 1)[individual[mutate_index] == 0]
            return individual
        
        # multipoint crossover
        def multipoint_crossover(parent1,parent2,nbr_point):
            coupure=int(len(parent1)/nbr_point)
            point_depart=0
            child=[]
            for i in range(nbr_point): 
                while(point_depart<len(parent1)):
                    if np.random.rand(1)<0.5 :
                        #child.append(parent1[point_depart:point_depart+coupure])
                        child = child + parent1[point_depart:point_depart+coupure]
                        point_depart = point_depart+coupure
                    else : 
                        #child.append(parent2[point_depart:point_depart+coupure])
                        child = child + parent2[point_depart:point_depart+coupure]
                        point_depart = point_depart+coupure   
            #print(list(itertools.chain(*child)))
            #return list(itertools.chain(*child))
            return child
        
        # partially matched crossover
        def pmx(parent1,parent2):
            size = min(len(parent1), len(parent2))
            p1, p2 = [0]*size, [0]*size
            
            for i in xrange(size):
                p1[parent1[i]] = i
                p2[parent2[i]] = i

            cxpoint1 = random.randint(0, size)
            cxpoint2 = random.randint(0, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else: 
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1
                
            for i in xrange(cxpoint1, cxpoint2):

                temp1 = parent1[i]
                temp2 = parent2[i]

                parent1[i], parent1[p1[temp2]] = temp2, temp1
                parent2[i], parent2[p2[temp1]] = temp1, temp2

                p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
                p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

            return parent1, parent2

        self.calcul_fitness = calcul_fitness
        self.calcul_individual_fitness = calcul_individual_fitness
        self.best_individual_intern = best_individual_intern
        self.best_individual_extern = best_individual_extern
        self.worst_individual_extern = worst_individual_extern
        self.random_selection = random_selection
        self.roulette_selection = roulette_selection
        self.basic_crossover = basic_crossover
        self.rand_crossover = rand_crossover
        self.mutate = mutate
        self.multipoint_crossover = multipoint_crossover
        self.pmx = pmx
        
    def create_random_initial_population(self):
        # Create members of the first population randomly.
        initial_population = []
        for _ in range(self.population_size):
            individual = [random.randint(0, 1) for x in range(self.individual_length)]
            initial_population.append(individual)
        self.initial_population = initial_population
    
    def create_initial_population(self, test, sampling_size):
        """
        - test : "pearson", "spearman" or "kendall"
        - sampling_size : the proportion of the dataset to calculate the coefficients of the test, for each individual of 
                            the initial population
        """
        # Initial population : 10 samplings - using Pearson, Spearman or Kendall test
        initial_population = []
        for i in range(self.population_size):
            # Sampling from the dataframe
            df_sample = self.dataframe.sample(n=sampling_size)
            X = df_sample.iloc[:, 0:df_sample.shape[1]-1]
            Y = df_sample.iloc[:, df_sample.shape[1]-1]
            coefficients = []
            Test_individual = []
            # Calculate the coefficients of the test
            for i in range(X.shape[1]):
                if test == "pearson":
                    coef_p = pearsonr(np.array(X.iloc[:,i]), np.array(Y))[0]
                elif test == "spearman":
                    coef_p = spearmanr(np.array(X.iloc[:,i]), np.array(Y))[0]
                elif test == "kendall":
                    coef_p = kendalltau(np.array(X.iloc[:,i]), np.array(Y))[0]
                rand = np.random.rand(1)
                coefficients.append(coef_p)
                boolean = float(rand) < np.abs(coef_p)
                if boolean:
                    Test_individual.append(1)
                else:
                    Test_individual.append(0)
            initial_population.append(Test_individual)
            index_coefficients = np.argsort(np.abs(coefficients), axis=0)
        self.initial_population = np.array(initial_population)
        self.initial_population_duplicated = np.array(initial_population)

        
    def run(self):
        #self.create_initial_population()
        
        # Example : 100 generations
        max_fitness_intern = []
        max_fitness_extern = []
        mean_fitness_extern = []
        generation = self.initial_population_duplicated

        for k in range(self.iterations):

            # Calculate fitness for each individual of the population
            fitness_intern, fitness_extern = self.calcul_fitness(generation)

            # To plot evolution 
            max_fitness_extern.append(self.best_individual_extern(generation)[0])

            #mean_fitness_extern.append(np.mean(fitness_extern))
            #index_fitness_extern = np.argsort(np.abs(fitness_extern), axis=0)
            #print(index_fitness_extern) 

            # Selection
            parent1, parent2 = self.roulette_selection(generation, fitness_intern)

            # Random crossover
            child = self.rand_crossover(parent1, parent2, self.crossover_probability)
            if self.calcul_individual_fitness(child)[1]>self.worst_individual_extern(generation)[0]:
                generation[self.worst_individual_extern(generation)[2],:] = child

            # Mutation
            can_mutate = np.random.rand(1) < self.mutation_probability
            if can_mutate:
                self.mutate(child)
            
            if k==(self.iterations-1):
                self.best_individual = self.best_individual_extern(generation)[1]
                features_selected = []
                for i in range(len(self.best_individual)):
                    if self.best_individual[i]==1:
                        features_selected.append(i)
                self.features_selected = features_selected
                self.last_generation = generation

        #plt.plot(mean_fitness_extern)
        plt.plot(max_fitness_extern)
