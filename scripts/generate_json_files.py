import json
import os

def run_instance(scheme, degree, n_refinements):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/generate_json_files.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["FEDegree"]              = degree
    datastore["NRefinements"]          = n_refinements
    datastore["TimeIntegrationScheme"] = scheme

    # write data to output file
    with open("./generate_json_files_%s_%s_%d.json" % (scheme, str(n_refinements).zfill(2), degree), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    
    counter = 0;

    for scheme in ["irk", "spirk"]:
        for degree in [1,2]:
            for n_refinements in range(2,20):

                run_instance(scheme, degree, n_refinements)


if __name__== "__main__":
  main()
