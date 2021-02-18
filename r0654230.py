import Reporter
import numpy as np
from matplotlib import pyplot as plt


class Params:

    def __init__(self):
        # general
        self.mu = 100  # population size
        self.its = 150  # number of iterations
        self.convdifference = 60  # n. its with small best value improvement for the program to stop: if None, off

        # initialization
        self.initpop = "mix"  # whether an initial pupulation is given. "mix" allows part of the population random and
        # part from a shuffle of the Nearest Neighbor Heuristic (NNH) approximation
        self.nnhfraction = 0.5  # fraction of initial population coming from a shuffle of the nnh individual
        self.shuffler = 1  # shuffle function to be used: 1, 2, 3 (1 is most impactful, 3 is bad)
        self.sr = 0.2

        # variation
        self.k = 2  # k tournament parameter
        self.lambR = 4  # offspring size. lambR: ratio between offspring and population. In Evolution
        # Strategies λ>μ with a great offspring surplus (typically λ/μ ≈ 5 − 7) that induces a large sel. pressure.
        self.mrate = 0.1  # mutation rate
        self.madapted = False  # whether mutation rate is part of self-adaptation

        # Local search
        self.lso_end = 150  # iteration at which to stop local search (knn) in offspring
        self.knn_c = 20  # counter for applying lso in whole population

        # Elimination
        self.elim = "shared_red"
        self.alpha = 0.5  # penalty exponent for shared elimination
        self.sigma = 0.3  # radius of penalty (penalized if less than sigma edges different from average survivor)
        self.twinpenalizer = 1.5  # penalizer for same individuals

        # Reporting
        self.plot = False  # whether convergence plot should be plotted
        self.plotedges = False  # plot edges present in population
        self.print = False  # print each iteration summary
        self.trial = 0  # for labelling graphs


class r0654230:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # Initializers

    def cutdm(self, dm0):
        """Removes infinity values from Distance Matrix"""
        maxdist = np.nanmax(dm0[dm0 != np.inf])
        dm = np.copy(dm0)
        dm[np.isinf(dm)] = 5 * maxdist
        return dm

    def initmrate(self, mu, mrate, madapted):
        """ Initialize mutation rates around a given mrate constant"""
        init_mrates = np.zeros((mu, 1))
        if madapted:  # mutation rate adaptivity
            for i in range(mu):
                init_mrates[i] = np.random.normal(mrate, abs(mrate / 2))
        else:
            for i in range(mu):
                init_mrates[i] = mrate
        return init_mrates

    def initialize(self, mu, n):
        """ Randomly initialize a population of paths. """
        init_pop = np.zeros((mu, n), dtype=int)
        for i in range(mu):
            init_pop[i] = np.random.permutation(n)
        return init_pop

    def nnhinit(self, mu, dm, n, shuffler, sr):
        """ Initialize population based on an NNH approximation. """
        nnhind = np.zeros((1, n), dtype=int)
        nnhind[0, :] = self.nnh(dm, n)
        if shuffler == 3:
            pop = self.shuffler3(nnhind, mu, n, sr)
        elif shuffler == 2:
            pop = self.shuffler2(nnhind, mu, n, sr)
        else:
            pop = self.shuffler1(nnhind, mu, n, sr)
        return pop

    def mixinit(self, mu, dm, n, shuffler, sr, nnhfraction):
        """ initializes a population partly randomly, partly with NNH"""
        popnnh = self.nnhinit(int(mu * nnhfraction), dm, n, shuffler, sr)
        poprand = self.initialize(mu - int(mu * nnhfraction), n)
        pop3 = np.concatenate((popnnh, poprand))
        return pop3

    # NNH and Shufflers

    def nnh(self, dm0, n, repeats=5):
        """ Finds individual close to NNH heuristic by choosing different starting points """
        paths = np.zeros((repeats, n), dtype=int)
        fits = np.zeros((repeats, 1))
        for r in range(repeats):
            s = np.random.randint(n)
            seq = np.zeros((1, n))
            seq[:] = np.arange(0, n, 1)
            dm = np.copy(dm0)
            dm = np.concatenate((seq, dm))  # labels of each city (since columns will be deleted)
            path = [s]  # start in a random city
            next_ = s
            dm = np.delete(dm, s, axis=1)
            for i in range(1, n):
                distances = dm[next_ + 1, :]  # compute distances
                reducedi = np.argmin(distances)  # closest next city (index)
                next_ = int(dm[0, reducedi])  # closest next city
                dm = np.delete(dm, reducedi, axis=1)  # delete city from search space
                path.append(next_)  # add city to path
            fits[r] = dm0[path[n - 1], path[0]] + dm0[path[:n - 1], path[1:]].sum()
            paths[r] = np.array(path)
        besti = np.argmin(fits)  # choose shortest NNH approximation
        return paths[besti, :]

    def shuffler3(self, nnhind, mu, n, shuffl_rate):
        """ Swaps the order of two segments of path """
        pop = np.zeros((mu, n), dtype=int)
        for i in range(0, mu):
            ind = np.copy(nnhind)
            for j in range(n):
                ind = np.roll(ind, -1)
                if np.random.uniform() < shuffl_rate:
                    arc2start = np.random.randint(1, n - 1)
                    arc3start = np.random.randint(arc2start + 1, n)
                    ind[0, 0:arc3start] = np.hstack((ind[0, arc2start:arc3start], ind[0, :arc2start]))
            pop[i, :] = ind
        return pop

    def shuffler2(self, nnhind, mu, n, shuffl_rate):
        """ Swaps position of two adjacent cities. """
        pop = np.zeros((mu, n), dtype=int)
        for i in range(0, mu):
            ind = np.copy(nnhind)
            for j in range(n):
                if np.random.uniform() < shuffl_rate:
                    ind[0, j - 1], ind[0, j] = ind[0, j], ind[0, j - 1]
            pop[i, :] = ind
        return pop

    def shuffler1(self, nnhind, mu, n, shuffl_rate):
        """ Inverts segment of path of maximum 6 cities"""
        pop = np.zeros((mu, n), dtype=int)
        for i in range(0, mu):
            ind = np.copy(nnhind)
            for j in range(n):
                ind = np.roll(ind, -1)
                if np.random.uniform() < shuffl_rate:
                    arclength = np.random.randint(2, 6)  # 2 to 6 cities in subpath
                    ind[0, 0:arclength] = np.flip(ind[0, 0:arclength])
            pop[i, :] = ind
        return pop

    # Variation operators

    def select(self, population, fitness, mrates, k):
        """ Select an individual from the given population based on k-tournament selection """
        indices = np.random.random_integers(0, fitness.shape[0] - 1, k)
        winner_index = indices[fitness[indices].argmin()]
        return population[winner_index], mrates[winner_index]

    def order_crossover(self, p1, p2, mr1, mr2, n):
        """ Perform order crossover with 2 given parents """
        p1 = np.roll(p1, np.random.randint(n))
        p2 = np.roll(p2, np.random.randint(n))
        cpa = np.random.randint(0, n - 1)
        cpb = np.random.randint(0, n - 1)
        if cpa > cpb:
            temp = cpb
            cpb = cpa
            cpa = temp
        offspring = list(p1[cpa:cpb])  # copy segment from parent 1 to offspring
        i = cpb
        while len(offspring) != n:
            if not (p2[i] in offspring):
                offspring.append(p2[i])
            i = i + 1
            if i == n:
                i = 0
        mr3 = mr1 + (np.random.uniform() * 2 - 0.5) * (mr2 - mr1)
        return np.array(offspring), mr3

    def mutate2(self, ind, fit, mut_rate, dm, n):
        """ Randomly change the order inside of an arc """
        if np.random.uniform() < mut_rate:
            # randomly decide which arc to mutate
            arc1 = np.random.randint(n - 4)
            arc2 = arc1 + 5
            if arc1 > arc2:
                arc1, arc2 = arc2, arc1
            ind[arc1:arc2] = np.flip(ind[arc1:arc2])
            indpop = np.zeros((1, n), dtype=int)
            indpop[0, :] = ind
            fit = self.evaluate(indpop, dm)
        return ind, fit

    # Local search operator

    def knn(self, pop, dm, n):
        """ Goes through genotype once to check if it benefits from swapping two adjacent edges """
        mu = pop.shape[0]  # size of population considered
        newpop = np.copy(pop)
        for i in range(mu):
            ind = newpop[i, :]
            check = np.ones(n, dtype=bool)
            trial = 0
            while any(check) and trial < 1:
                for j in range(0, n):
                    if check[j]:
                        path_orig = dm[ind[j - 3], ind[j - 2]] + dm[ind[j - 2], ind[j - 1]] + dm[
                            ind[j - 1], ind[j]]  # path: a-b-c-d
                        path_new = dm[ind[j - 3], ind[j - 1]] + dm[ind[j - 1], ind[j - 2]] + dm[
                            ind[j - 2], ind[j]]  # path: a-c-b-d
                        if path_orig > path_new:
                            ind[j - 1], ind[j - 2] = ind[j - 2], ind[j - 1]
                            check[(j - 1 + n) % n] = 1
                            check[(j + 1) % n] = 1
                        check[j] = 0
                trial += 1
        return newpop

    # Evaluation and elimination

    def evaluate(self, population, distance_matrix):
        """ Calculate the distance of all paths in the given population """
        return distance_matrix[population[:, -1], population[:, 0]] + \
               distance_matrix[population[:, :-1], population[:, 1:]].sum(1)

    def eliminate(self, population, fitness, mrates, mu):
        """ Keeps best mu individuals from the population. @fitness is the fitness of each individual """
        sortorder = np.argsort(fitness)
        population2 = population[sortorder[:mu]]
        fitness2 = fitness[sortorder[:mu]]
        mrates2 = mrates[sortorder[:mu]]
        return population2, fitness2, mrates2

    def evaluateshared(self, population, fitness, surv_fitness, mu_surv, edges, alpha, sigma, n, twinpenalizer):
        """Returns fitness penalized by having less than sigma not common edges with 'edges'. alpha increases
                            penalization, and does not work at 0 """
        shared_fit = np.copy(fitness)
        for i, row in enumerate(population):
            beta = (edges[row[:-1], row[1:]].sum() + edges[row[-1], row[0]]) / (mu_surv * n)
            # beta: how many edges indivudual has in common with each selected survivor
            if beta >= 1 - sigma:  # beta usually around 0.9
                shared_fit[i] = fitness[i] * (1 + beta) ** alpha
            for sfi in range(mu_surv):
                if abs(fitness[i] - surv_fitness[sfi]) < 0.1:  # extra penalty for equal individuals
                    shared_fit[i] = shared_fit[i] * twinpenalizer
        return shared_fit

    def eliminateshared(self, population, fitness, mrates, mu, alpha, sigma, twinpenalizer):
        """ eliminate less fit individuals taking into account how similar individual is to others """
        n = population.shape[1]
        edges = np.zeros((n, n), dtype=int)
        survivors = np.zeros((mu, n), dtype=int)
        surv_fitness = np.zeros(mu)
        surv_mrates = np.zeros((mu, 1))
        minfit_i = np.argmin(fitness)  # choose first individual
        survivors[0] = population[minfit_i, :]
        surv_fitness[0] = fitness[minfit_i]
        surv_mrates[0] = mrates[minfit_i]
        for i in range(1, mu):
            edges[survivors[i - 1, :-1], survivors[i - 1, 1:]] += 1
            edges[survivors[i - 1, -1], survivors[i - 1, 0]] += 1
            sharedfit = self.evaluateshared(population, fitness, surv_fitness, i, edges, alpha, sigma, n, twinpenalizer)
            bestarg = np.argmin(sharedfit)
            survivors[i, :] = population[bestarg, :]
            surv_fitness[i] = fitness[np.argmin(sharedfit)]
            surv_mrates[i] = mrates[bestarg]
        # print(bestarg)
        sortorder = np.argsort(surv_fitness)
        sorted_surv_pop = survivors[sortorder]
        sorted_surv_fit = surv_fitness[sortorder]
        sorted_surv_mrates = surv_mrates[sortorder]
        return sorted_surv_pop, sorted_surv_fit, sorted_surv_mrates, edges

    def elimshared_red(self, pop, fit, mrates, mu, alpha, sigma, twinpenalizer, iteration):
        """Reduced shared elimination, with only part of the population undergoing full process of shared elimination"""
        mu2 = mu // 5  # indivuduals to go full shared elimination
        n = pop.shape[1]
        survivors = np.zeros((mu, n), dtype=int)
        surv_fitness = np.zeros(mu)
        surv_mrates = np.zeros((mu, 1))
        survivors[:mu2, :], surv_fitness[:mu2], surv_mrates[:mu2], edges = self.eliminateshared(pop, fit, mrates,
                                                                                                mu2, alpha, sigma,
                                                                                                twinpenalizer)
        sharedfit = self.evaluateshared(pop, fit, surv_fitness, mu2, edges, alpha, sigma, n, twinpenalizer)
        sortorder = np.argsort(sharedfit)
        survivors[mu2:, :] = pop[sortorder[:mu - mu2], :]
        surv_fitness[mu2:] = fit[sortorder[:mu - mu2]]
        surv_mrates[mu2:] = mrates[sortorder[:mu - mu2]]
        return survivors, surv_fitness, surv_mrates

    def eliminatetwins(self, population, fitness, mrates, mu, twinpenalizer):
        """ Penalizes identical individuals """
        penalized_fitness = np.copy(fitness)
        mu_tot = population.shape[0]
        # 1. Penalization
        for i in range(mu_tot):
            for j in range(i + 1, mu_tot):
                if abs(fitness[i] - fitness[j]) < 0.1:
                    penalized_fitness[j] = fitness[j] * twinpenalizer
        # 2. Selecting best based on penalized fitness
        sorted_pen_fitnesses = np.argsort(penalized_fitness)
        population2 = population[sorted_pen_fitnesses[:mu]]
        fitness2 = fitness[sorted_pen_fitnesses[:mu]]
        mrates2 = mrates[sorted_pen_fitnesses[:mu]]
        # 3. Sorting answers based on actual fitness
        sortorder = np.argsort(fitness2)
        sorted_pop = population2[sortorder]
        sorted_fit = fitness2[sortorder]
        sorted_mrates = mrates2[sortorder]
        return sorted_pop, sorted_fit, sorted_mrates

    # Statistics and plotting

    def get_statistics(self, population, fitness):
        """ Calculate the mean fitness, best fitness and best individual of the given population"""
        mean_fitness = np.mean(fitness)
        i = np.argmin(fitness)
        best_fitness = fitness[i]
        best_individual = population[i]
        i0 = np.where(best_individual == 0)
        rolled_best_individual = np.roll(best_individual, -i0[0][0])
        return mean_fitness, best_fitness, rolled_best_individual

    def plotedges(self, pop, mu, n, trial, iteration):
        edges = np.zeros((n, n), dtype=int)
        neighbours = 50
        for i in range(0, mu):
            edges[pop[i, :-1], pop[i, 1:]] += 1
            edges[pop[i, -1], pop[i, 0]] += 1
        red_edges = np.zeros((n, neighbours + 1), dtype=int)
        for i, r in enumerate(edges):
            for j in range(-neighbours // 2, neighbours // 2 + 1):
                red_edges[i, j + neighbours // 2] = edges[i, (i + j) % n]
        plt.figure()
        plt.imshow(red_edges)
        plt.title(f"Edges for trial {trial} - iteration {iteration}")
        plt.show()
        pass

    # The evolutionary algorithm's main loop

    def optimize(self, filename, p=None):

        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix0 = np.loadtxt(file, delimiter=",")
        file.close()
        distanceMatrix = self.cutdm(distanceMatrix0)
        n = distanceMatrix.shape[0]  # number of cities

        # initialize parameters
        if p is None:
            p = Params()
        if n > 500:  # modifications for larger problems
            p.convdifference = 20
            p.nnhfraction = 0.9
            p.mu = 70
            p.lambR = 3
            p.its = 100
            p.sr = 0.05
            p.sigma = 0.7
            p.knn_c = None

        # Arrays to return for visualization
        means = []
        bests = []
        iterations = []

        # Initialization
        if p.initpop is None:
            pop = self.initialize(p.mu, n)
        elif p.initpop == "nnh":  # approximated nnh heuristic
            pop = self.nnhinit(p.mu, distanceMatrix, n, p.shuffler, p.sr)
        elif p.initpop == "mix":
            pop = self.mixinit(p.mu, distanceMatrix, n, p.shuffler, p.sr, p.nnhfraction)
        else:
            pop = p.initpop

        fit = self.evaluate(pop, distanceMatrix)  # fitnesses of initial population

        sortorder = np.argsort(fit)  # sorting population by fitness
        pop = pop[sortorder]
        fit = fit[sortorder]
        mrates = self.initmrate(p.mu, p.mrate, p.madapted)

        if p.plotedges:  # plotting edges of initial population
            self.plotedges(pop, p.mu, n, p.trial, -1)

        # Iteration loop
        iteration = 0
        convcounter = 0
        knncounter = 0
        decreasing = True
        bestObjective0 = np.Inf
        while iteration < p.its and decreasing:

            # Selection + crossover
            offspring = np.zeros((p.mu * p.lambR, n), dtype=int)
            offmrates = np.zeros((p.mu * p.lambR, 1))
            for i in range(p.mu * p.lambR):
                p1, mr1 = self.select(pop, fit, mrates, p.k)
                p2, mr2 = self.select(pop, fit, mrates, p.k)
                offspring[i], offmrates[i] = self.order_crossover(p1, p2, mr1, mr2, n)

            # Local Search: every 5 iterations
            if iteration % 5 == 4 and (iteration < p.lso_end or iteration >= p.its + p.lso_end):
                offspring = self.knn(offspring, distanceMatrix, n)
            offfit = self.evaluate(offspring, distanceMatrix)

            joined_pop = np.concatenate((pop, offspring))
            joined_fit = np.concatenate((fit, offfit))
            joined_mrates = np.concatenate((mrates, offmrates))

            # Mutation
            mutated_pop = np.copy(joined_pop)
            # protecting top 5 individuals from last generation from mutation
            for i in range(5, joined_pop.shape[0]):
                mutated_pop[i], joined_fit[i] = self.mutate2(joined_pop[i], joined_fit[i],
                                                             joined_mrates[i], distanceMatrix, n)

            # Elimination
            if p.elim == "twin":  # (λ+μ) elimination without copies
                pop, fit, mrates = self.eliminatetwins(mutated_pop, joined_fit, joined_mrates, p.mu, p.twinpenalizer)
            elif p.elim == "l,m":  # (λ,μ) elimination without copies
                offpop = mutated_pop[p.mu:, :]
                offfit = joined_fit[p.mu:]
                pop, fit, mrates = self.eliminatetwins(offpop, offfit, offmrates, p.mu, p.twinpenalizer)
            elif p.elim == "shared":  # shared elimination
                pop, fit, mrates, _ = self.eliminateshared(mutated_pop, joined_fit, joined_mrates,
                                                           p.mu, p.alpha, p.sigma, p.twinpenalizer)
            elif p.elim == "shared_red":  # "reduced" shared elimination
                pop, fit, mrates = self.elimshared_red(mutated_pop, joined_fit, joined_mrates, p.mu, p.alpha, p.sigma,
                                                    p.twinpenalizer, iteration)
            else:  # (λ+μ) elimination
                pop, fit, mrates = self.eliminate(mutated_pop, joined_fit, joined_mrates, p.mu)

            meanObjective, bestObjective, bestSolution = self.get_statistics(pop, fit)

            # Save values to plot (visualization), plot edges, print iteration
            if p.plot:
                means.append(meanObjective)
                bests.append(bestObjective)
                iterations.append(iteration)
            if iteration % 10 == 0 and p.plotedges:
                self.plotedges(pop, p.mu, n, p.trial, iteration)
            if p.print:
                print("Iteration: ", iteration, ", Mean fitness: ", meanObjective, ", Best fitness: ", bestObjective)

            # knn
            if p.knn_c is not None:
                if bestObjective0 - bestObjective > 30:
                    decreasing = True
                    knncounter = 0
                else:
                    knncounter += 1
                    if knncounter > p.knn_c:
                        # print(f"iteration {iteration}: knn")
                        knnpop = self.knn(pop, distanceMatrix, n)
                        knnfit = self.evaluate(pop, distanceMatrix)
                        knnmrates = self.initmrate(p.mu, p.mrate, p.madapted)
                        pop, fit, mrates = self.eliminate(knnpop, knnfit, knnmrates, p.mu)
                        knncounter = -5

            if p.convdifference is not None:
                if bestObjective0 - bestObjective > 30:
                    decreasing = True
                    convcounter = 0
                else:
                    convcounter += 1
                    if convcounter > p.convdifference:
                        decreasing = False
                        print("convergence reached")

            bestObjective0 = bestObjective
            iteration += 1

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
        if p.plotedges:
            self.plotedges(pop, p.mu, n, p.trial, iteration)
        return 0
