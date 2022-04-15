import json
import os
import sys 

def run_instance(counter, n_refinements, n_stages, n_ranks, scheme):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["NRefinements"]          = n_refinements
    datastore["IRKStages"]             = n_stages
    datastore["MaxRanks"]              = n_ranks
    datastore["TimeIntegrationScheme"] = scheme


    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():    
    counter = 0

    n_stages = int(sys.argv[1])
    n_ranks  = int(sys.argv[2]) * 48


    for n_refinements in range(3,20):
        # SPIRK
        run_instance(counter, n_refinements, n_stages, n_ranks, "spirk")
        counter = counter + 1;

        # IRK
        run_instance(counter, n_refinements, n_stages, n_ranks, "irk")
        counter = counter + 1;

        # IRK
        run_instance(counter, n_refinements, n_stages, n_ranks / n_stages, "irk")
        counter = counter + 1;


if __name__== "__main__":
  main()
