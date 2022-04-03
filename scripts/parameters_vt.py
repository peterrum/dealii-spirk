import json
import os

def run_instance(counter, n_refinements, scheme, do_row_major, use_sm):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["NRefinements"]          = n_refinements
    datastore["TimeIntegrationScheme"] = scheme
    datastore["DoRowMajor"]            = do_row_major

    if use_sm:
        datastore["Padding"]         = 0
        datastore["UseSharedMemory"] = True


    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():    
    counter = 0

    for n_refinements in range(3,20):
        # IRK
        run_instance(counter, n_refinements, "irk", True, False)
        counter = counter + 1;

        # SPIRK (row major)
        run_instance(counter, n_refinements, "spirk", True, False)
        counter = counter + 1;

        # SPIRK (row major - sm)
        run_instance(counter, n_refinements, "spirk", True, True)
        counter = counter + 1;

        # SPIRK (column major)
        run_instance(counter, n_refinements, "spirk", False, False)
        counter = counter + 1;


if __name__== "__main__":
  main()
