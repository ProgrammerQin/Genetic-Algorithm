import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 40  # DNA length
POP_SIZE = 100  # population size
CROSS_RATE = 0.8  # mating probability (DNA crossover)
MUTATION_RATE = 0.003  # mutation probability
N_GENERATIONS = 100

u_BOUND = [20, 35]
s_BOUND = [350, 500]
k_BOUND = [50000, 80000]
c_BOUND = [1500, 2700]


# find the maximum of the reciprocal of orignial function
def F(u, s, k, c):
    const = np.pi * 30 * 6.5* 1e-06
    part = 50000 * c / (2 * (s**1.5) * (k**0.5) ) + (u + s)*k*k / (2*c*s*s)
    return 1 / ( (const * part)**0.5 )

# find non-zero fitness for selection
def get_fitness(F_values): return F_values + 1e-3 - np.min(F_values)


# convert binary DNA for X to decimal and normalize it to a range(-1, 1)
def translateDNA_1(pop):
    pop_1=pop[:, 0:10]
    return pop_1.dot(2 ** np.arange(np.int(DNA_SIZE/4))[::-1]) / float(2**(DNA_SIZE/4)-1) * (u_BOUND[1]-u_BOUND[0]) + u_BOUND[0]

def translateDNA_2(pop):
    pop_2=pop[:, 10:20]
    return pop_2.dot(2 ** np.arange(np.int(DNA_SIZE/4))[::-1]) / float(2**(DNA_SIZE/4)-1) * (s_BOUND[1]-s_BOUND[0]) + s_BOUND[0]

def translateDNA_3(pop):
    pop_3=pop[:, 20:30]
    return pop_3.dot(2 ** np.arange(np.int(DNA_SIZE/4))[::-1]) / float(2**(DNA_SIZE/4)-1) * (k_BOUND[1]-k_BOUND[0]) + k_BOUND[0]

def translateDNA_4(pop):
    pop_4=pop[:, 30:40]
    return pop_4.dot(2 ** np.arange(np.int(DNA_SIZE/4))[::-1]) / float(2**(DNA_SIZE/4)-1) * (c_BOUND[1]-c_BOUND[0]) + c_BOUND[0]


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]


def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # choose crossover points
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))

#######################
#   plt.ion()  # Turn the interactive plotting mode on
#   X = np.linspace(-1, 1, 100)
#   Y = np.linspace(-1, 1, 100)
#   X, Y = np.meshgrid(X, Y)

#   Z = (3 * X ** 2 + np.sin(5 * np.pi * X)) + (3 * Y ** 4 + np.cos(3 * np.pi * Y)) + 10
#   fig = pyplot.figure()
#   ax = Axes3D(fig)
#   ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                   cmap=cm.nipy_spectral, linewidth=0.08,
#                   antialiased=True)
#########################


for i in range(N_GENERATIONS):
    F_values = F(translateDNA_1(pop), translateDNA_2(pop), translateDNA_3(pop), translateDNA_4(pop))
    # compute function value by extracting DNA

#    sca = ax.scatter(translateDNA_X(pop), translateDNA_Y(pop), 1 / F_values, s=100, c='red');
#    plt.pause(0.1)

    # Following if statement is to remove the points on plotted curve in previous generation
#    if i < (N_GENERATIONS - 1): sca.remove()

    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop)
        child = mutate(child)
        parent[:] = child  # parent is replaced by its child

plt.ioff()  # Turn the interactive plotting mode off
plt.show()

pop_best_1_DNA = pop[np.argmax(fitness), 0:10]
pop_best_1_real = pop_best_1_DNA.dot(2 ** np.arange(np.int(DNA_SIZE / 4))[::-1]) / float(2 ** (DNA_SIZE / 4) - 1) * (
            u_BOUND[1] - u_BOUND[0]) + u_BOUND[0]

pop_best_2_DNA = pop[np.argmax(fitness), 10:20]
pop_best_2_real = pop_best_2_DNA.dot(2 ** np.arange(np.int(DNA_SIZE / 4))[::-1]) / float(2 ** (DNA_SIZE / 4) - 1) * (
            s_BOUND[1] - s_BOUND[0]) + s_BOUND[0]

pop_best_3_DNA = pop[np.argmax(fitness), 20:30]
pop_best_3_real = pop_best_3_DNA.dot(2 ** np.arange(np.int(DNA_SIZE / 4))[::-1]) / float(2 ** (DNA_SIZE / 4) - 1) * (
            k_BOUND[1] - k_BOUND[0]) + k_BOUND[0]

pop_best_4_DNA = pop[np.argmax(fitness), 30:40]
pop_best_4_real = pop_best_4_DNA.dot(2 ** np.arange(np.int(DNA_SIZE / 4))[::-1]) / float(2 ** (DNA_SIZE / 4) - 1) * (
            c_BOUND[1] - c_BOUND[0]) + c_BOUND[0]

print("Best real u value: ", pop_best_1_real)
print("Best real s value: ", pop_best_2_real)
print("Best real k value: ", pop_best_3_real)
print("Best real c value: ", pop_best_4_real)
print("Maximum function value: ", F(pop_best_1_real, pop_best_2_real, pop_best_3_real, pop_best_4_real))

