import json
import os
from argparse import ArgumentParser

def run_instance(counter, n_refinements, dt, k, n_stages, scheme):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["NRefinements"]          = n_refinements
    datastore["TimeStepSize"]          = dt
    datastore["FEDegree"]              = k
    datastore["IRKStages"]             = n_stages
    datastore["TimeIntegrationScheme"] = scheme

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")

    parser.add_argument('n_stages', type=int)
    parser.add_argument('k', type=int)
    
    arguments = parser.parse_args()
    return arguments

def main():
    options = parseArguments()

    n_stages = options.n_stages
    k        = options.k
    
    counter = 0

    for n_refinements in range(3,20):
        dt = 0.1 # TODO: select a more useful value

        # IRK
        run_instance(counter, n_refinements, dt, k, n_stages, "irk")
        counter = counter + 1;

        # SPIRK
        run_instance(counter, n_refinements, dt, k, n_stages, "spirk")
        counter = counter + 1;


if __name__== "__main__":
  main()
