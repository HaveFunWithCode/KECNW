import argparse
import pandas as pd
from src import KECNW

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--datapath", required=True, help='dataset file as csv file')
    ap.add_argument("-t", "--textfield", required=True, help='text field name like "text"')
    ap.add_argument("-w", "--weighting_criteria", required=True,
                    help='list of weighting criterias with - as spliter (ex: DC-SC-Neigh_Imp-F-L-TF)')
    ap.add_argument("-n", "--numberofkeys", required=True, help='the number of best keywords')

    args = vars(ap.parse_args())

    print(args["datapath"])
    print(args["textfield"])
    print(list(args["weighting_criteria"].split('-')))

    dataset = pd.read_csv(args["datapath"])
    _text_field_name = args["textfield"]
    _n_best = args["numberofkeys"]
    _criterias = list(args["weighting_criteria"].split('-'))

    print('creating graph .......')
    g = KECNW(_dataset_df=dataset, _text_field_name=_text_field_name)
    g.set_weighting_criteria(_criterias)
    print('Extracted keywords .......')
    keys = g.keyword_extraction(n_best=int(_n_best))
    print(keys)

