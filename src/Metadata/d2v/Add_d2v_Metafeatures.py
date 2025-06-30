import os
import json

import pandas as pd
import numpy as np
import tensorflow as tf

import argparse

from src.Metadata.d2v.dataset2vec.dummdataset import Dataset
from src.Metadata.d2v.dataset2vec.modules import FunctionF, PoolF, FunctionG, PoolG, FunctionH
from src.Metadata.d2v.dataset2vec.sampling import Batch, TestSampling
from src.utils.get_data import get_openml_dataset_split_and_metadata, get_name_and_split_and_save_dataset
from src.utils.get_matrix import get_additional_d2v_columns

def get_d2v_metafeatures(dataset_id):
    tf.random.set_seed(0)
    np.random.seed(42)
    dataset_name, datset_split = get_name_and_split_and_save_dataset(dataset_id)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--split',
                        help='Select metafeature extraction model (one can take the average of the metafeatures across all 5 splits)',
                        type=int, default=datset_split)
    parser.add_argument('--file', help='Select dataset name', type=str)

    args = parser.parse_args()
    args.file = dataset_name

    def Dataset2VecModel(configuration):
        nonlinearity_d2v = configuration['nonlinearity_d2v']
        # Function F
        units_f = configuration['units_f']
        nhidden_f = configuration['nhidden_f']
        architecture_f = configuration['architecture_f']
        resblocks_f = configuration['resblocks_f']

        # Function G
        units_g = configuration['units_g']
        nhidden_g = configuration['nhidden_g']
        architecture_g = configuration['architecture_g']

        # Function H
        units_h = configuration['units_h']
        nhidden_h = configuration['nhidden_h']
        architecture_h = configuration['architecture_h']
        resblocks_h = configuration['resblocks_h']
        #
        batch_size = configuration["batch_size"]
        trainable = False
        # input two dataset2vec shape = [None,2], i.e. flattened tabular batch
        x = tf.keras.Input(shape=[2], dtype=tf.float32)
        # Number of sampled classes from triplets
        nclasses = tf.keras.Input(shape=[batch_size], dtype=tf.int32, batch_size=1)
        # Number of sampled features from triplets
        nfeature = tf.keras.Input(shape=[batch_size], dtype=tf.int32, batch_size=1)
        # Number of sampled instances from triplets
        ninstanc = tf.keras.Input(shape=[batch_size], dtype=tf.int32, batch_size=1)
        # Encode the predictor target relationship across all instances
        layer = FunctionF(units=units_f, nhidden=nhidden_f, nonlinearity=nonlinearity_d2v, architecture=architecture_f,
                          resblocks=resblocks_f, trainable=trainable)(x)
        # Average over instances
        layer = PoolF(units=units_f)(layer, nclasses[0], nfeature[0], ninstanc[0])
        # Encode the interaction between features and classes across the latent space
        layer = FunctionG(units=units_g, nhidden=nhidden_g, nonlinearity=nonlinearity_d2v, architecture=architecture_g,
                          trainable=trainable)(layer)
        # Average across all instances
        layer = PoolG(units=units_g)(layer, nclasses[0], nfeature[0])
        # Extract the metafeatures
        metafeatures = FunctionH(units=units_h, nhidden=nhidden_h, nonlinearity=nonlinearity_d2v,
                                 architecture=architecture_h, trainable=trainable, resblocks=resblocks_h)(layer)
        # define hierarchical dataset representation model
        dataset2vec = tf.keras.Model(inputs=[x, nclasses, nfeature, ninstanc], outputs=metafeatures)
        return dataset2vec

    rootdir = os.path.dirname(os.path.realpath(__file__)) + "/dataset2vec"
    log_dir = os.path.join(rootdir, "checkpoints", f"searchspace-a/split-0/dataset2vec/vanilla/configuration-0/2025-06-05-18-47-03-578668")
    save_dir = os.path.join(rootdir, "extracted")
    configuration = json.load(open(os.path.join(log_dir, "configuration.txt"), "r"))
    os.makedirs(save_dir, exist_ok=True)

    metafeatures = pd.DataFrame(data=None)
    datasetmf = []

    batch = Batch(configuration['batch_size'])
    dataset = Dataset(args.file, rootdir)
    testsampler = TestSampling(dataset=dataset)

    model = Dataset2VecModel(configuration)

    model.load_weights(os.path.join(log_dir + "/iteration-50/.weights.h5/.weights.h5"))  # , by_name=False, skip_mismatch=False)

    for q in range(10):  # any number of samples
        batch = testsampler.sample_from_one_dataset(batch)
        batch.collect()
        datasetmf.append(model(batch.input).numpy())

    metafeatures = pd.DataFrame(np.vstack(datasetmf).mean(axis=0)[None], index=[args.file])
    return metafeatures

def add_d2v_metadata_columns(dataset_metadata, X_train, result_matrix):
    columns = get_additional_d2v_columns()
    metafeatures = get_d2v_metafeatures(dataset_metadata["task_id"])
    new_columns = pd.DataFrame(index=result_matrix.index, columns=columns)
    for row in result_matrix.iterrows():
        featurename = row[1][1]
        matching_indices = result_matrix[result_matrix["feature - name"] == str(featurename)].index
        for idx in matching_indices:
            new_columns.loc[idx] = metafeatures.iloc[0]
    insert_position = result_matrix.shape[1] - 2
    result_matrix = pd.concat([result_matrix.iloc[:, :insert_position], new_columns, result_matrix.iloc[:, insert_position:]], axis=1)
    return result_matrix


def main():
    result_matrix = pd.read_parquet("src/Metadata/core/Core_Matrix_Example.parquet")
    columns = get_additional_d2v_columns()
    result_matrix_pandas = pd.DataFrame(columns=columns)
    for dataset, _ in result_matrix.groupby('dataset - id'):
        print("Dataset: " + str(dataset))
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset)
        result_matrix_dataset = result_matrix[result_matrix['dataset - id'] == dataset]
        result_matrix_dataset = add_d2v_metadata_columns(dataset_metadata, X_train, result_matrix_dataset)
        result_matrix_pandas = pd.concat([result_matrix_pandas, result_matrix_dataset], axis=1)
        result_matrix.to_parquet("src/Metadata/d2v/D2V_Matrix_Complete" + str(dataset) + ".parquet")
    result_matrix.to_parquet("src/Metadata/d2v/D2V_Matrix_Complete.parquet")


if __name__ == '__main__':
    main()
