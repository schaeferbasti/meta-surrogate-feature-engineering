import datetime

from src.utils.get_data import get_openfe_data, get_openml_dataset_split
from src.utils.preprocess_data import factorize_data_split
from src.utils.run_models import run_default_lgbm, run_autogluon_lgbm


def main():
    f = open("results.txt", "w")
    f.write("Test different versions of LGBM with OpenFE \n" + str(datetime.datetime.now()) +"\n\n")
    f.close()

    """
            If running on a SLURM cluster, we need to initialize Ray with extra options and a unique tempr dir.
            Otherwise, given the shared filesystem, Ray will try to use the same temp dir for all workers and crash (semi-randomly).
            """
    import os
    import logging
    import ray
    import uuid
    import base64
    import time
    log = logging.getLogger(__name__)
    ray_mem_in_gb = 32
    log.info(f"Running on SLURM, initializing Ray with unique temp dir with {ray_mem_in_gb}GB.")
    ray_mem_in_b = int(ray_mem_in_gb * (1024.0 ** 3))
    tmp_dir_base_path = "tmp_dir_base_path"
    uuid_short = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b'=').decode('ascii')
    ray_dir = f"{tmp_dir_base_path}/{uuid_short}/ray"
    print(f"Start local ray instances. Using {os.environ.get('RAY_MEM_IN_GB')} GB for Ray.")
    ray.init(
        address="local",
        _memory=ray_mem_in_b,
        object_store_memory=int(ray_mem_in_b * 0.3),
        _temp_dir=ray_dir,
        include_dashboard=False,
        logging_level=logging.INFO,
        log_to_driver=True,
        num_gpus=0,
        num_cpus=8,
    )

    dataset_ids = [190411, 359983, 189354, 189356, 10090, 359979, 146818, 359955, 359960, 359968, 359959, 168757, 359954, 359969, 359970, 359984, 168911, 359981, 359962, 359965, 190392, 190137, 359958, 168350, 359956, 359975, 359963, 168784, 190146, 146820, 359974, 2073, 359944, 359950, 359942, 359951, 360945, 167210, 359930, 359948, 359931, 359932, 359933, 359934, 359939, 359945, 359935, 359940]
    for dataset_id in dataset_ids:
        f = open("results.txt", "a")
        f.write("Dataset: " + str(dataset_id) + "\n")
        f.close()

        X_train, y_train, X_test, y_test = get_openml_dataset_split(dataset_id)
        X_train, y_train, X_test, y_test = factorize_data_split(X_train, y_train, X_test, y_test)
        X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe = get_openfe_data(X_train, y_train, X_test, y_test)
        try:
            lgbm_results = run_default_lgbm(X_train, y_train, X_test, y_test)
            f = open("results.txt", "a")
            f.write("LGBM Results " + str(lgbm_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("LGBM Results " + str(e) + "\n")
            f.close()
        try:
            lgbm_openfe_results = run_default_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe)
            f = open("results.txt", "a")
            f.write("LGBM OpenFE Results " + str(lgbm_openfe_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("LGBM OpenFE Results " + str(e) + "\n")
            f.close()
        try:
            autogluon_lgbm_results = run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=False)
            f = open("results.txt", "a")
            f.write("Autogluon LGBM Results " + str(autogluon_lgbm_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("Autogluon LGBM Results " + str(e) + "\n")
            f.close()
        try:
            autogluon_lgbm_openfe_results = run_autogluon_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe, zeroshot=False)
            f = open("results.txt", "a")
            f.write("Autogluon LGBM OpenFE Results " + str(autogluon_lgbm_openfe_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("Autogluon LGBM OpenFE Results " + str(e) + "\n")
            f.close()
        try:
            tuned_autogluon_lgbm_results = run_autogluon_lgbm(X_train, y_train, X_test, y_test, zeroshot=True)
            f = open("results.txt", "a")
            f.write("Tuned Autogluon LGBM Results " + str(tuned_autogluon_lgbm_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("Tuned Autogluon LGBM Results " + str(e) + "\n")
            f.close()
        try:
            tuned_autogluon_lgbm_openfe_results = run_autogluon_lgbm(X_train_openfe, y_train_openfe, X_test_openfe, y_test_openfe, zeroshot=True)
            f = open("results.txt", "a")
            f.write("Tuned Autogluon LGBM OpenFE Results " + str(tuned_autogluon_lgbm_openfe_results) + "\n")
            f.close()
        except Exception as e:
            f = open("results.txt", "a")
            f.write("Tuned Autogluon LGBM OpenFE Results " + str(e) + "\n")
            f.close()


if __name__ == '__main__':
    main()
    