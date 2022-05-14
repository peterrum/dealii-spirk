import json
import os

def run_instance(counter, n_refinements, scheme, tol):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["NRefinements"]          = n_refinements
    datastore["TimeIntegrationScheme"] = scheme
    datastore["InnerTolerance"]        = tol

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():    
    counter = 0

    for n_refinements in range(3,20):
        for tol in [0.0, 1e-1, 1e-2, 1e-3, 1e-4]:
            # IRK
            run_instance(counter, n_refinements, "irk", tol)
            counter = counter + 1;

            # SPIRK (row major)
            run_instance(counter, n_refinements, "spirk", tol)
            counter = counter + 1;


if __name__== "__main__":
  main()
