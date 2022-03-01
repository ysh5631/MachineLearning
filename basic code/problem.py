import random
import math
from setup import Setup

class Problem(Setup):
    def __init__(self):
        Setup.__init__(self)
        self._solution = []
        self._value = 0
        self._numEval = 0

        self._pFileName = ''
        self._bestSolution = []
        self._bestMinimum = 0
        self._avgMinimum = 0
        self._avgNumEval = 0
        self._sumOfNumEval = 0
        self._avgWhen = 0

    def setVariables(self, parameters):
        Setup.setVariables(self, parameters)
        self._pFileName = parameters['pFileName']

    def getSolution(self):
        return self._solution

    def getValue(self):
        return self._value

    def getNumEval(self):
        return self._numEval

    def randomInit(self):
        pass

    def evaluate(self):
        pass

    def mutants(self):
        pass

    def randomMutant(self, current):
        pass

    def describe(self):
        pass
    
    def storeResult(self, solution, value):
        self._solution = solution
        self._value = value

    def storeExpResult(self,results):
        self._bestSolution = results[0]
        self._bestMinimum = results[1]
        self._avgMinimum = results[2]
        self._avgNumEval = results[3]
        self._sumOfNumEval = results[4]
        self._avgWhen = results[5]

    def report(self):
        aType = self._aType
        if 1 <= aType <= 4:
            print("Average number of evaluations: {0:,}" \
                  .format(round(self._avgNumEval)))
        if 5 <= aType <= 6:
            print("Average iteration of finding the best: {0:,}"
                  .format(self_avgWhen))
        print()

    def reportNumEvals(self):
        if 1 <= self._aType <= 4:
            print()
            print("Total number of evaluations: {0:,}"
                  .format(self._sumOfNumEval))

class Numeric(Problem):
    def __init__(self):
        Problem.__init__(self)
        self._expression = ''
        self._domain = []
        
    def setVariables(self, parameters):
        Problem.setVariables(self,parameters)
        ## Read in a function and its domain from a file
        ## Then, set the relevant class variables
        infile = open(parameters['pFileName'],'r')
        self._expression = infile.readline()
        varNames, low, up = [], [], []
        line = infile.readline()
        while line !='':
            data = line.split(",")
            varNames.append(data[0])
            low.append(float(data[1]))
            up.append(float(data[2]))
            line = infile.readline()
        infile.close()
        self._domain = [varNames,low,up]
    
    def randomInit(self): ###
        domain = self._domain # domain: [varNames, low, up]
        low,up = domain[1],domain[2]
        init = []
        for i in range(len(low)):
            r = random.uniform(low[i],up[i])
            init.append(r)
        return init

    def evaluate(self, current):
        self._numEval += 1
        expr = self._expression         # p[0] is function expression
        varNames = self._domain[0]  # p[1] is domain
        for i in range(len(varNames)):
            assignment = varNames[i] + '=' + str(current[i])
            exec(assignment)
        return eval(expr)
    
    def mutants(self, current): ###
        neighbors = []
        for i in range(len(current)):
            mutant = self.mutate(current,i,n.self._delta,p)
            neighbors.append(mutant)
            mutant = self.mutate(current,i,-self.delta,p)
            neighbors.append(mutant)
        return neighbors     # Return a set of successors

    def mutate(self, current, i, d): ## Mutate i-th of 'current' if legal
        mutant = current[:]
        domain = self._domain        # [VarNames, low, up]
        l = domain[1][i]     # Lower bound of i-th
        u = domain[2][i]     # Upper bound of i-th
        if l <= (mutant[i] + d) <= u:
            mutant[i] += d
        return mutant

    def randomMutant(self, current): ###
        i = random.randint(0, len(current) -1)
        if random.uniform(0,1) > 0.5:
            d = self._delta
        else:
            d = -self._delta
        return self.mutate(current, i, d) # Return a random successor

    def takeStep(self, x, v):
        grad = self.gradient(x, v)
        xCopy = x[:]
        for i in range(len(xCopy)):
            xCopy[i] = xCopy[i] - self._alpha * grad[i]
        if self.isLegal(xCopy):
            return xCopy
        else:
            return x

    def gradient(self, x, v):
        grad = []
        for i in range(len(x)):
            xCopyH = x[:]
            xCopyH[i] += self._dx
            g = (self.evaluate(xCopyH) - v) / self._dx
            grad.append(g)
        return grad

    def isLegal(self, x):
        domain = self._domain
        low = domain[1]
        up = domain[2]
        flag = True
        for i in range(len(low)):
            if x[i] < low[i] or up[i] < x[i]:
                flag = False
                break
        return flag

    def describe(self):
        print()
        print("Objective function:")
        print(self._expression)   # Expression
        print("Search space:")
        varNames = self._domain[0] # p[1] is domain: [VarNames, low, up]
        low = self._domain[1]
        up = self._domain[2]
        for i in range(len(low)):
            print(" " + varNames[i] + ":", (low[i], up[i]))

    def report(self):
        avgMinimum = round(self._avgMinimum, 3)
        print()
        print("Average objective value: {0:,}".format(avgMinimum))
        Problem.report(self)
        print("Best solution found:")
        print(self.coordinate())  # Convert list to tuple
        print("Best value: {0:,.3f}".format(self._bestMinimum))
        self.reportNumEvals()

    def coordinate(self):
        c = [round(value, 3) for value in self._solution]
        return tuple(c)


    # code below
    
    def initializePop(self, size): # Make a population of given size
        pop = []
        for i in range(size):
            chromosome = self.randBinStr()
            pop.append([0, chromosome])
        return pop

    def randBinStr(self):
        k = len(self._domain[0]) * self._resolution
        chromosome = []
        for i in range(k):
            allele = random.randint(0, 1)
            chromosome.append(allele)
        return chromosome

    def evalInd(self, ind):  # ind: [fitness, chromosome]
        ind[0] = self.evaluate(self.decode(ind[1])) # Record fitness

    def decode(self, chromosome):
        r = self._resolution
        low = self._domain[1]  # list of lower bounds
        up = self._domain[2]   # list of upper bounds
        genotype = chromosome[:]
        phenotype = []
        start = 0
        end = r   # The following loop repeats for # variables
        for var in range(len(self._domain[0])): 
            value = self.binaryToDecimal(genotype[start:end],
                                         low[var], up[var])
            phenotype.append(value)
            start += r
            end += r
        return phenotype

    def binaryToDecimal(self, binCode, l, u):
        r = len(binCode)
        decimalValue = 0
        for i in range(r):
            decimalValue += binCode[i] * (2 ** (r - 1 - i))
        return l + (u - l) * decimalValue / 2 ** r

    def crossover(self, ind1, ind2, uXp):
        # pC is interpreted as uXp# (probability of swap)
        chr1, chr2 = self.uXover(ind1[1], ind2[1], uXp)
        return [0, chr1], [0, chr2]

    def uXover(self, chrInd1, chrInd2, uXp): # uniform crossover
        chr1 = chrInd1[:]  # Make copies
        chr2 = chrInd2[:]
        for i in range(len(chr1)): 
            if random.uniform(0, 1) < uXp:
                chr1[i], chr2[i] = chr2[i], chr1[i]
        return chr1, chr2

    def mutation(self, ind, mrF):  # bit-flip mutation
        # pM is interpreted as mrF (factor to adjust mutation rate)
        child = ind[:]    # Make copy
        n = len(ind[1])
        for i in range(n):
            if random.uniform(0, 1) < mrF * (1 / n):
                child[1][i] = 1 - child[1][i]
        return child

    def indToSol(self, ind):
        return self.decode(ind[1])

class TSP(Problem):
    def __init__(self):
        Problem.__init__(self)
        self._numCities = 0
        self._locations = []
        self._distanceTable = []
        
    def setVariables(self, parameters):
        ## Read in a TSP (# of cities, locatioins) from a file.
        ## Then, create a problem instance and return it.
        Problem.setVariables(self, parameters)
        infile = open(self._pFileName, 'r')
        # First line is number of cities
        self._numCities = int(infile.readline())
        cityLocs = []
        line = infile.readline()  # The rest of the lines are locations
        while line != '':
            cityLocs.append(eval(line)) # Make a tuple and append
            line = infile.readline()
        infile.close()
        self._distanceTable = self.calcDistanceTable()
        self._locations = cityLocs

    def calcDistanceTable(self):
        locations = self._locations
        table = []
        for i in range(self._numCities):
            row = []
            for j in range(self._numCities):
                dx = locations[i][0] - locations[j][0]
                dy = locations[i][1] - locations[j][1]
                d = round(math.sqrt(dx**2 + dy**2),1)
                row.append(d)
            table.append(row)
        return table # A symmetric matrix of pairwise distances
    
    def randomInit(self): ###
        n = self._numCities
        init = list(range(n))
        random.shuffle(init)
        return init

    def evaluate(self, current):
        ## Calculate the tour cost of 'current'
        ## 'p' is a Problem instance
        ## 'current' is a list of city ids
        self._numEval += 1
        n = self._numCities
        table = self._distanceTable
        cost = 0
        for i in range(n-1):
            locFrom = current[i]
            locTo = current[i+1]
            cost += table[locFrom][locTo]
        return cost
    
    def mutants(self, current): ###
        n = self._numCities
        neighbors = []
        count = 0
        triedPairs = []
        while count <= n:  # Pick two random loci for inversion
            i, j = sorted([random.randrange(n) for _ in range(2)])
            if i < j and [i, j] not in triedPairs:
                triedPairs.append([i, j])
                mutant = self.inversion(current, i, j)
                count += 1
                neighbors.append(mutant)
        return neighbors

    def inversion(self,current, i, j):  # Perform inversion
        mutant = current[:]
        while i < j:
            mutant[i], mutant[j] = mutant[j], mutant[i]
            i += 1
            j -= 1
        return mutant

    def randomMutant(self,current): # Apply inversion
        while True:
            i, j = sorted([random.randrange(self._numCities)
                           for _ in range(2)])
            if i < j:
                curCopy = self.inversion(current, i, j)
                break
        return mutant

    def describe(self):
        print()
        n = self._numCities
        print("Number of cities:", n)
        print("City locations:")
        locations = self._locations
        for i in range(n):
            print("{0:>12}".format(str(locations[i])), end = '')
            if i % 5 == 4:
                print()

    def report(self):
        avgMinimum = round(self._avgMinimum)
        print()
        print("Average tour cost: {0:,}".format(avgMinimum))
        Problem.report(self)
        print("Best tour found:")
        self.tenPerRow()
        print("Best tour cost: {0:,}"
              .format(round(self._bestMinimum)))
        self.reportNumEvals()

    def tenPerRow(self):
        solution = self._bestSolution
        for i in range(len(solution)):
            print("{0:>5}".format(solution[i]), end='')
            if i % 10 == 9:
                print()

    # code below
    
    def initializePop(self, size): # Make a population of given size
        n = self._numCities        # n: number of cities
        pop = []
        for i in range(size):
            chromosome = self.randomInit()
            pop.append([0, chromosome])
        return pop

    def evalInd(self, ind):  # ind: [fitness, chromosome]
        ind[0] = self.evaluate(ind[1]) # Record fitness

    def crossover(self, ind1, ind2, XR): 
        # pC is interpreted as XR (crossover rate)
        if random.uniform(0, 1) <= XR:
            chr1, chr2 = self.oXover(ind1[1], ind2[1])
        else:
            chr1, chr2 = ind1[1][:], ind2[1][:]  # No change
        return [0, chr1], [0, chr2]

    def oXover(self, chrInd1, chrInd2):  # Ordered Crossover
        chr1 = chrInd1[:]
        chr2 = chrInd2[:]  # Make copies
        size = len(chr1)
        a, b = sorted([random.randrange(size) for _ in range(2)])
        holes1, holes2 = [True] * size, [True] * size
        for i in range(size):
            if i < a or i > b:
                holes1[chr2[i]] = False
                holes2[chr1[i]] = False
        # We must keep the original values somewhere
        # before scrambling everything
        temp1, temp2 = chr1, chr2
        k1, k2 = b + 1, b + 1
        for i in range(size):
            if not holes1[temp1[(i + b + 1) % size]]:
                chr1[k1 % size] = temp1[(i + b + 1) % size]
                k1 += 1
            if not holes2[temp2[(i + b + 1) % size]]:
                chr2[k2 % size] = temp2[(i + b + 1) % size]
                k2 += 1
        # Swap the content between a and b (included)
        for i in range(a, b + 1):
            chr1[i], chr2[i] = chr2[i], chr1[i]
        return chr1, chr2

    def mutation(self, ind, mR): # mutation by inversion
        # pM is interpreted as mR (mutation rate for inversion)
        child = ind[:]  # Make copy
        if random.uniform(0, 1) <= mR:
            i, j = sorted([random.randrange(self._numCities)
                           for _ in range(2)])
            child[1] = self.inversion(child[1], i, j)
        return child

    def indToSol(self, ind):
        return ind[1]
