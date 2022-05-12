import json
import os
import sys 

def run_instance(counter, n_refinements, scheme, do_row_major):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["NRefinements"]          = n_refinements
    datastore["TimeIntegrationScheme"] = scheme
    datastore["IRKStages"]             = do_row_major

    if(scheme.startswith('complex_')):
        datastore["InnerTolerance"] = 0.0

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():    
    
    prefix = ""

    if(len(sys.argv) > 1):
      prefix = sys.argv[1]

    counter = 0

    for n_refinements in range(3,20):
        for n_stages in [2, 4, 6, 8]:
            # IRK
            run_instance(counter, n_refinements, prefix + "irk", n_stages)
            counter = counter + 1;

            # SPIRK (row major)
            run_instance(counter, n_refinements, prefix + "spirk", n_stages)
            counter = counter + 1;


if __name__== "__main__":
  main()
