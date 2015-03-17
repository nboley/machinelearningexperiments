
def load_data():
    samples = None
    genes = []
    N_genes = 1000 #103501
    expression = numpy.zeros((N_genes, 10), dtype=float)
    with open("all_quant.txt") as fp:
        for i, line in enumerate(fp):
            if i > N_genes: break
            data = line.split()
            if i == 0:
               samples = data[1:]
               continue
            genes.append(data[0])
            expression[i-1,:] = map(float, data[1:])
    
    s1 = expression[:,(3,0,5)]
    s2 = expression[:,(4,2,7)]
    return genes, s1, s2

def main():
    genes, sample1, sample2 = load_data()
    pdag, cond_independence_sets = estimate_pdag(sample1, sample2, genes)
    print pdag
    return

main()
