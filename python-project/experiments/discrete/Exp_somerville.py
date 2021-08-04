from experiments.discrete import DiscreteExperiment
from spn.structure.StatisticalTypes import MetaType


class Exp_somerville(DiscreteExperiment.DiscreteExperiment):

    # 7 attributes after filtering with 10 folds
    meta_types = [MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE,
                  MetaType.DISCRETE, MetaType.DISCRETE]
    var_types_string = "uuuuuuu"

    def run(self, run: int, n_folds: int, fold_log: bool, kde_bw: str):
        print("\n------------------------------------------------------------------")
        print("------------------------------------------------------------------")
        print("--------------------------- SOMERVILLE ---------------------------")
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------\n")

        super().run(run, n_folds, fold_log, kde_bw)


def main():
    run = 1
    n_folds = 10
    kde_bw = "normal_reference"
    data_name = "somerville"
    fold_log = True
    exp = Exp_somerville(data_name)
    exp.run(run, n_folds, fold_log, kde_bw)


if __name__ == "__main__":
    main()
