import json
import os

def run_instance(scheme, degree, n_refinements, n_stages):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/generate_json_files.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["FEDegree"]              = degree
    datastore["NRefinements"]          = n_refinements
    datastore["TimeIntegrationScheme"] = scheme
    datastore["IRKStages"]             = n_stages

    # write data to output file
    with open("./generate_json_files_%s_%s_%d_%d.json" % (scheme, str(n_refinements).zfill(2), degree, n_stages), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    
    counter = 0;

    number_nodes     = 16
    number_processes = 48 * number_nodes

    #for degree in [1,2]:
    #    for n_stages in [2, 3, 9]:
    for degree in [1]:
        for n_stages in [3]:
            for scheme in ["irk", "spirk"]:

                print("array=($(ls generate_json_files_%s_*_%d_%d.json))" % (scheme, degree, n_stages))

                if scheme == "irk":
                    print("mpirun -np %d ./main \"${array[@]}\" | tee node-%s_%s-%d-%d.out" % (number_processes / n_stages, str(number_nodes).zfill(4), "irk0", degree, n_stages))
                    print("mpirun -np %d ./main \"${array[@]}\" | tee node-%s_%s-%d-%d.out" % (number_processes, str(number_nodes).zfill(4), "irk1", degree, n_stages))
                else:
                    print("mpirun -np %d ./main \"${array[@]}\" | tee node-%s_%s-%d-%d.out" % (number_processes, str(number_nodes).zfill(4), "spirk", degree, n_stages))

                print("")

                for n_refinements in range(2,20):
                    run_instance(scheme, degree, n_refinements, n_stages)


if __name__== "__main__":
  main()
