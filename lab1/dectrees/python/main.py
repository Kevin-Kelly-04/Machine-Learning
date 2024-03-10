import dtree as dt
import monkdata as m
import drawtree_qt5 as qt
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def calcGain_allAttr(dataset):
	gain = []
	for attribute in m.attributes:
		gain.append((attribute, round(dt.averageGain(dataset, attribute), 4)))
	return gain


def assign1():
	print(dt.entropy(m.monk1))
	print(dt.entropy(m.monk2))
	print(dt.entropy(m.monk3))

def assign2(dataset):
	for attribute in m.attributes:
		print(round(dt.averageGain(dataset, attribute), 6))

def Mcommon(dataset):
	print()
	print()
	bestAttribute = dt.bestAttribute(dataset, m.attributes)
	subset = []
	for value in bestAttribute.values:
		subset.append(dt.select(dataset, bestAttribute, value))

	for i in range(len(subset)):
		print(bestAttribute, "= ", bestAttribute.values[i] )
		print("Most common:", dt.mostCommon(subset[i]), "\n")
		print(calcGain_allAttr(subset[i]), "\n")


def assign5a(dataset):
	bestAttribute = dt.bestAttribute(dataset, m.attributes)
	print(bestAttribute)
	print()
	subset = []

	for value in bestAttribute.values:
		subset.append(dt.select(dataset, bestAttribute, value))

	for i in range(len(subset)):
		print(bestAttribute, "= ", bestAttribute.values[i] )
		print(calcGain_allAttr(subset[i]), "\n")
	return subset

def assign5b(dataset):
	subset = assign5a(dataset)
	print("a5=1 mostCommon:", dt.mostCommon(subset[0]))

	for i in range(len(subset)):
		print("a5 = ", i)
		Mcommon(subset[i])

def assign5():
    tree1 = dt.buildTree(m.monk1, m.attributes)
    tree2 = dt.buildTree(m.monk2, m.attributes)
    tree3 = dt.buildTree(m.monk3, m.attributes)

    qt.drawTree(tree3)

    # Training error
    print("TRAINING ERROR")
    print("MONK-1:", 1 - dt.check(tree1, m.monk1))
    print("MONK-2:", 1 - dt.check(tree2, m.monk2))
    print("MONK-3:", 1 - dt.check(tree3, m.monk3))

    # Testing error
    print("TESTING ERROR")
    print("MONK-1:", 1 - dt.check(tree1, m.monk1test))
    print("MONK-2:", 1 - dt.check(tree2, m.monk2test))
    print("MONK-3:", 1 - dt.check(tree3, m.monk3test))

		
def partition(dataset, fraction):
    data = list(dataset)
    random.shuffle(data)
    breakPoint = int(len(data) * fraction)
    return data[:breakPoint], data[breakPoint:]


def prune_one_node(baseTree, valData):
    trees = dt.allPruned(baseTree)

    bestAcc = dt.check(baseTree, valData)
    bestIndex = -1

    for i in range(len(trees)):
        prunedAcc = dt.check(trees[i], valData)
        if prunedAcc >= bestAcc:
            bestAcc = prunedAcc
            bestIndex = i

    if bestIndex == -1:
        return None
    return trees[bestIndex]


def prune_tree(tree, valData):
    while True:
        prunedTree = prune_one_node(tree, valData)
        if prunedTree is None:
            break
        else:
            tree = prunedTree
    return tree


def assign7(dataset, testData, N, fraction):
    testAcc = []
    std = []

    for frac in fraction:
        fracAcc = []
        for iter in range(N):
            trData, valData = partition(dataset, frac)
            tree = dt.buildTree(trData, m.attributes)
            prunedTree = prune_tree(tree, valData)
            fracAcc.append(1 - dt.check(prunedTree, testData))
        testAcc.append(np.mean(fracAcc))
        std.append(np.std(fracAcc, ddof=1))

    assert(len(fraction) == len(testAcc) and len(fraction) == len(std))

    return testAcc, std


def assign7_plot():
    N = 500
    fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    testAcc_monk1, std_monk1 = assign7(m.monk1, m.monk1test, N, fraction)
    testAcc_monk3, std_monk3 = assign7(m.monk3, m.monk3test, N, fraction)

    matplotlib.rcParams.update({'font.size': 10})
    plt.errorbar(fraction, testAcc_monk1, yerr=std_monk1, fmt="--o", ecolor="k",
             elinewidth=0.4, capsize=2, capthick=0.4, linewidth=0.4)
    plt.errorbar(fraction, testAcc_monk3, yerr=std_monk3, fmt="--o", ecolor="k",
                 elinewidth=0.4, capsize=2, capthick=0.4, linewidth=0.4, color="r")
    plt.xlabel("Fraction of training data")
    plt.ylabel("Test error")
    plt.legend(("MONK-1", "MONK-3"))
    plt.title("Classification error on the test data using decision trees with reduced error pruning, %d trials. \n" % N)
    plt.show()


# Training data sets
#monk1 = m.monk1
#monk2 = m.monk2
#monk3 = m.monk3

assign7_plot()


#assign1()
#assign2(m.monk3)
#assign5b(m.monk1)
#assign5()
