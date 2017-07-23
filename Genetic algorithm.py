# Import data
df = pd.read_csv("C:/Users/bensmail/Documents/Python Scripts/Toolkit/ds-toolkit/test_data/boston.csv")

# Splitting data
X_int, X_ext, y_int, y_ext = train_test_split(X, Y, test_size=0.33)

# Method : replacing worst individuals
# Initial population : 10 samplings - Pearson
initial_population = []
t = 10 # Population size
for i in range(t):
    # Sampling 
    z = 100 # Sampling size
    df_sample = df.sample(n=z)
    X = df_sample.iloc[:, 0:df_sample.shape[1]-1]
    Y = df_sample.iloc[:, df_sample.shape[1]-1]
	# Calculate Pearson coefficients
    Pearson_coefficients = []
    Pearson_individual = []
    for i in range(X.shape[1]):
        coef_p = pearsonr(np.array(X.iloc[:,i]), np.array(Y))[0]
        rand = np.random.rand(1)
        Pearson_coefficients.append(coef_p)
        boolean = float(rand) < np.abs(coef_p)
        if boolean:
            Pearson_individual.append(1)
        else:
            Pearson_individual.append(0)
    initial_population.append(Pearson_individual)
    index_Pearson_coefficients = np.argsort(np.abs(Pearson_coefficients), axis=0)
initial_population = np.array(initial_population)

# Calculate fitness for each individual of the initial population
fitness_intern, fitness_extern = calcul_fitness(initial_population)

# Example : 100 generations
max_fitness_intern = []
max_fitness_extern = []
generation = initial_population
for k in range(300):
    max_fitness_extern.append(best_individual_extern(generation)[0])
    # Parent 1
    fitness_intern_sum = np.sum(fitness_intern)
    proba = fitness_intern/fitness_intern_sum
    index_proba = np.row_stack((np.array([x for x in range(10)]), (np.cumsum([proba]))))
    rand = np.random.rand(1)
    for i in range(10):
        if rand>index_proba[1,i] and rand<index_proba[1,i+1]:
            selected_individual_1 = i
            parent1 = list(generation[selected_individual_1,:])
    # Parent 2
    rand = np.random.rand(1)
    for i in range(10):
        if rand>index_proba[1,i] and rand<index_proba[1,i+1] and i!=selected_individual_1:
            selected_individual_2 = i
            parent2 = list(generation[selected_individual_2,:])
    
    # Random crossing
    child = rand_crossing(parent1, parent2, probability_crossing=0.5)
    if calcul_individual_fitness(child)[1]>worst_individual_extern(generation)[0]:
        generation[worst_individual_extern(generation)[2],:] = child
plt.plot(max_fitness_extern)

# Calcul fitness for the whole population with r2 coefficient of Linear Regression
def calcul_fitness(population):
    fitness_intern = []
    fitness_extern = []
    for j in range(population.shape[0]):
        selected_predictors = []
        for i in range(len(population[j])):
            if population[j][i]==1:
                selected_predictors.append(i)
        model_intern = LinearRegression()
        model_intern.fit(X_int.iloc[:,selected_predictors], y_int)
        model_extern = LinearRegression()
        model_extern.fit(X_ext.iloc[:,selected_predictors], y_ext)
        fitness_intern.append(model_intern.score(X_int.iloc[:,selected_predictors], y_int))
        fitness_extern.append(model_extern.score(X_ext.iloc[:,selected_predictors], y_ext))
    return fitness_intern, fitness_extern

# Calcul fitness for each individual with r2 coefficient of Linear Regression
def calcul_individual_fitness(individual):
    selected_predictors = []
    for i in range(len(individual)):
        if individual[i]==1:
            selected_predictors.append(i)
    model_intern = LinearRegression()
    model_intern.fit(X_int.iloc[:,selected_predictors], y_int)
    individual_fitness_intern = model_intern.score(X_int.iloc[:,selected_predictors], y_int)
    model_extern = LinearRegression()
    model_extern.fit(X_ext.iloc[:,selected_predictors], y_ext)
    individual_fitness_extern = model_extern.score(X_int.iloc[:,selected_predictors], y_int)
    return individual_fitness_intern, individual_fitness_extern

# Return the best individual according to the intern fitness 
def best_individual_intern(population):
    max_fitness_intern = max(calcul_fitness(population)[0])
    best_individual = population[np.argmax(max_fitness_intern),:]
    return max_fitness_intern, best_individual

# Return the best individual according to the extern fitness 
def best_individual_extern(population):
    max_fitness_extern = max(calcul_fitness(population)[1])
    best_individual = population[np.argmax(max_fitness_extern),:]
    return max_fitness_extern, best_individual

# Return the worst inidividual according to the extern fitness
def worst_individual_extern(population):
    min_fitness_extern = min(calcul_fitness(population)[1])
    worst_individual = population[np.argmin(min_fitness_extern),:]
    index_worst_individual = np.argmin(min_fitness_extern)
    return min_fitness_extern, worst_individual, index_worst_individual

# Basic crossing using an index
def basic_crossing(father1,father2, index):
    # Child 1
    child1 = father1[0:index]+father2[index:]
    # Child 2
    child2 = father2[0:index]+father1[index:]
    return child1, child2

# Random crossing using a probability
def rand_crossing(father1,father2, probability_crossing): 
    child=[]
    for i in range(len(father1)):
        prob=np.random.rand(1)
        if prob < probability_crossing : 
            child.append(father1[i])
        else : 
            child.append(father2[i])
    return child