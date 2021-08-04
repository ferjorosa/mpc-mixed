from experiments.mixed import MixedExperiment
from spn.structure.StatisticalTypes import MetaType


class Exp_thoracic_surgery(MixedExperiment.MixedExperiment):

    # 14 attributes after filtering with 10 folds
    meta_types = [MetaType.DISCRETE, MetaType.REAL, MetaType.REAL, MetaType.DISCRETE, MetaType.DISCRETE,
                  MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE, MetaType.DISCRETE,
                  MetaType.DISCRETE, MetaType.DISCRETE, MetaType.REAL, MetaType.DISCRETE]
    var_types_string = "uccuuuuuuuuucu"

    def run(self, run: int, n_folds: int, fold_log: bool, kde_bw: str):
        print("\n------------------------------------------------------------------")
        print("------------------------------------------------------------------")
        print("------------------------- THORACIC_SURGERY -----------------------")
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------\n")

        super().run(run, n_folds, fold_log, kde_bw)


def main():
    run = 1
    n_folds = 10
    kde_bw = "normal_reference"
    data_name = "thoracic_surgery"
    fold_log = True
    exp = Exp_thoracic_surgery(data_name)
    exp.run(run, n_folds, fold_log, kde_bw)


if __name__ == "__main__":
    main()

