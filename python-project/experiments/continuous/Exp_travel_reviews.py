from experiments.continuous import ContinuousExperiment
from spn.structure.StatisticalTypes import MetaType


class Exp_travel_reviews(ContinuousExperiment.ContinuousExperiment):

    # 33 attributes after filtering with 10 folds
    meta_types = [MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL, MetaType.REAL,
                  MetaType.REAL, MetaType.REAL,MetaType.REAL, MetaType.REAL, MetaType.REAL]
    var_types_string = "cccccccccc"

    def run(self, run: int, n_folds: int, fold_log: bool, kde_bw: str):
        print("\n------------------------------------------------------------------")
        print("------------------------------------------------------------------")
        print("------------------------- TRAVEL_REVIEWS -------------------------")
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------\n")

        super().run(run, n_folds, fold_log, kde_bw)


def main():
    run = 1
    n_folds = 10
    kde_bw = "normal_reference"
    data_name = "travel_reviews"
    fold_log = True
    exp = Exp_travel_reviews(data_name)
    exp.run(run, n_folds, fold_log, kde_bw)


if __name__ == "__main__":
    main()

