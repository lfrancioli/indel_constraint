import os
import sys
import vcf
import h5py
import argparse
import numpy as np
import logging
import random
from collections import defaultdict
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.preprocessing import LabelBinarizer
from pprint import pprint
from os import listdir, makedirs, path
import pickle

from Bio import Seq, SeqIO

from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Input, Dense, Dropout, SpatialDropout2D, Flatten, Reshape, merge, SpatialDropout1D
from keras.layers.convolutional import Conv1D, Convolution2D, MaxPooling1D, MaxPooling2D

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("indel")
logger.setLevel(logging.INFO)

BASES = ['A','C','G','T']
dna_encoder = LabelBinarizer(sparse_output=True).fit(BASES)

def overlaps_intervals(intervals, start, stop):
    interval_start = np.searchsorted(intervals[:, 0], start)
    interval_stop = np.searchsorted(intervals[:, 0], stop)

    return (
        interval_start != interval_stop or
        start < intervals[interval_start - 1][1] or intervals[interval_start][0] == start or
        stop < intervals[interval_stop - 1][1] or intervals[interval_stop][0] == stop
    )


def in_intervals(intervals, value):
    interval_index = np.searchsorted(intervals[:, 0], value)
    return value < intervals[interval_index - 1][1] or intervals[interval_index][0] == value


def rle_encode_ambiguous(seq_list):
    ambiguous_indices = np.invert(np.in1d(seq_list, np.array(BASES, dtype='|S1')))
    pos, = np.where(np.diff(ambiguous_indices) != 0)
    pos = np.concatenate(([0], pos + 1, [len(ambiguous_indices)]))
    return np.array([(a, b) for (a, b) in zip(pos[:-1], pos[1:]) if ambiguous_indices[a]])


def load_fasta_reference(in_path):
    reference = {}
    ambiguous_bases = {}
    logger.info("Reading fasta file {}".format(in_path))
    start = timer()
    fasta = SeqIO.to_dict(SeqIO.parse(in_path, "fasta"))
    logger.info("Reading fasata file loaded in {}.".format(timer() - start))
    for contig, sequence in fasta.iteritems():
        start = timer()
        logger.info("Encoding chromosome {}".format(contig))
        seq_list = list(str(sequence.seq))
        #one_hot = dna_encoder.transform(seq_list)
        #reference[contig] = one_hot.astype(np.int8).todense()
        ambiguous_bases[contig] = rle_encode_ambiguous(seq_list)
        logger.info("Chrom {} encoded in {}.".format(contig, timer() - start))
    return reference, ambiguous_bases


def write_bitpacked_reference(reference, ambiguous_bases, out_path):
    logger.info("Writing reference to {}".format(out_path))
    if not path.exists(out_path):
        makedirs(out_path)
    for contig, sequence in reference.iteritems():
        np.save(out_path + "/{}.npy".format(contig), np.packbits(sequence))
    pickle.dump(ambiguous_bases, open(out_path + "/ambiguous_bases.pickle", "w"))


def load_bitpacked_reference(in_path):
    start = timer()
    logger.info("Loading bitpacked reference from {}".format(in_path))
    reference = {}
    for f in listdir(in_path):
        if f.endswith(".npy"):
            packed = np.load(in_path + "/" + f)
            reference[f[:-4]] = np.unpackbits(packed).reshape(packed.shape[1]*2, 4)

    ambiguous_bases = pickle.load(open(in_path + "/ambiguous_bases.pickle", "r"))

    for contig in reference:
        if contig not in ambiguous_bases:
            logger.warn("No ambiguous bases indices found for contig {}.".format(contig))

    logger.info("Reference loaded in {}.".format(timer() - start))
    return reference, ambiguous_bases


def main(args):
    if args.ref_fasta:
        reference, ambiguous_bases = load_fasta_reference(args.ref_fasta)
        write_bitpacked_reference(reference, ambiguous_bases, args.ref_fasta + ".bp")
    else:
        reference, ambiguous_bases = load_bitpacked_reference(args.ref)

    indels = read_vcf(args.vcf, args.max_training)

    logger.info("Loading training data")
    training, labels, indel_indices = vcf_to_indel_tensors(indels, reference, ambiguous_bases, args.window_size, args.neg_train_window)
    logger.info("Loaded training examples with dimensions {}, labels with dimensions: {} and indel indices with dimesions: {}.".format(str(training.shape), str(labels.shape), str(indel_indices.shape)))

    logger.info("Splitting training data into training and testing sets.")
    train, test = split_data([training, labels, indel_indices], [0.8,0.2])

    logger.info("Training shape: {}, labels shape: {}".format(train[0].shape, train[1].shape))

    logger.info("Creating model")
    model = make_indel_model(args.window_size)

    if args.train_model:
        logger.info("Training model.")
        if args.callback:
            model.load_weights(args.callback, by_name=True)
        history = model.fit(train[0], train[1], batch_size=32, validation_split=0.2, shuffle=True, epochs=10, callbacks=get_callbacks(args.output+".callbacks"))

        logger.info("Plotting metrics.")
        plot_metric_history(history, args.output + ".metrics_history.pdf", "Indel model")
    else:
        model.load_weights(args.callback, by_name=True)

    logger.info("Evaluating model.")
    eval_values = model.evaluate(test[0], test[1])
    pprint(", ".join(["{}: {}".format(metric,value) for metric, value in zip(model.metrics_names, eval_values)]))

    if args.save_best_examples:
        logger.info("Saving best scoring training examples")
        positive_examples, negative_examples = get_best_scoring_training_examples(model, train)
        file = open(args.output + ".best_training_examples.tsv",'w')
        for seq, score, label, indel_index in positive_examples + negative_examples:
            ref = indels[indel_index].REF if indel_index > -1 else seq[30]
            alt = ",".join([str(a) for a in indels[indel_index].ALT]) if indel_index > -1 else "null"
            training_label = "indel" if label[0] == 1 else "no_indel"
            file.write("{}\t{}\t{}\t{}\t{}\t{:.3f}\t{}\n".format(seq[:30], seq[30], seq[31:], ref, alt, score, training_label))
        file.close()


def read_vcf(vcf_path, max_training):
    logger.info("Loadind indels from VCF {}.".format(vcf_path))
    file = open(vcf_path, 'r')
    vcf_reader = vcf.Reader(file)
    indels = []

    # Get positive training examples
    start = timer()
    n_indels = 0
    for variant in vcf_reader:
        if max_training >= 0 and n_indels >= max_training:
            break

        if not variant.is_snp:
            indels.append(variant)
            n_indels += 1

    logger.info("Loaded {} indels from VCF in {}.".format(len(indels), timer() - start))
    file.close()
    return indels

def vcf_to_indel_tensors(indels, reference, ambiguous_bases, window_size, neg_train_window, squash_multi_allelic = False):
    positive_training = []
    negative_training = []
    labels = np.empty((0,2),dtype=np.int8)
    indel_indices = []

    # Get positive training examples
    start = timer()
    positive_training_pos = {}
    contig = indels[0].CHROM
    for i, variant in enumerate(indels):
        #Get positive training examples
        if not variant.is_snp and not (squash_multi_allelic and variant.POS in positive_training_pos):
            pos = variant.POS
            positive_training.append(reference[contig][pos - window_size -1: pos + window_size])
            indel_indices.append(i)
            while pos in positive_training_pos: # A bit of a trick for multi-allelic indels, but should be fine
                pos+=1
            positive_training_pos[pos] = 1

        if i+1 == len(indels) or indels[i+1].CHROM != variant.CHROM:
            print i
            # Get negative training examples
            negative_training.extend(get_negative_training_tensors(positive_training_pos,
                                                     ambiguous_bases[contig],
                                                     neg_train_window,
                                                     reference[contig],
                                                     window_size
                                                     ))

            indel_indices.extend([-1]*len(positive_training_pos))
            labels = np.append(
                labels,
                np.concatenate(
                (
                    np.full((len(positive_training_pos), 2), np.array([1, 0])),  # positive labels
                    np.full((len(positive_training_pos), 2), np.array([0, 1]))  # negative labels
                ), axis=0),
                axis=0
            )

            positive_training_pos = {}
            contig = variant.CHROM

    logger.info("Training examples loaded in {}.".format(timer() - start))

    return np.asarray(positive_training + negative_training), labels, np.asarray(indel_indices)

def one_hot_encode(data, encoder):
    all_data = "".join(data)
    res = encoder.transform(list(all_data))
    return np.asarray(np.split(res, len(data)))


def get_training_tensor(chrom, pos, reference, window_size, encoder):
    contig = reference[chrom]
    record = contig[pos - window_size: pos + window_size + 1]
    return encoder.transform(list(record))


def get_negative_training_tensors(positions, ambiguous_bases, neg_train_window, reference, window_size, max_retry = 100):
    max_pos = len(reference) - window_size
    neg_positions = []
    if neg_train_window > 0:
        for pos in positions.keys():
            neg_pos = random.randint(pos - neg_train_window, pos + neg_train_window)
            i = 0
            while (neg_pos in positions or in_intervals(ambiguous_bases, neg_pos) or neg_pos > max_pos or neg_pos < window_size) and i < max_retry:
                neg_pos = random.randint(pos - neg_train_window, pos + neg_train_window)
                i+=1

            if i == max_retry:
                logger.warn("No suitable negative training position found for positive training example {}. Skipping.".format(pos))
            else:
                neg_positions.append(neg_pos)
    else:
        # Samples randomly on the chromosome. Note that this procedure can lead to duplicate positions
        while len(neg_positions) < len(positions):
            new_neg_positions = [x for x in
                                 random.sample(xrange(window_size, max_pos),
                                               len(positions) - len(neg_positions))
                                 if x not in positions and not in_intervals(ambiguous_bases, x)]
            neg_positions.extend(new_neg_positions)

    return [reference[neg_pos - window_size -1: neg_pos + window_size] for neg_pos in neg_positions]


def make_indel_model(window_size):
    indel = Input(shape=(2 * window_size + 1, len(BASES)), name="indel")
    x = Conv1D(filters=200, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(indel)
    x = SpatialDropout1D(0.2)(x)
    x = Conv1D(filters=200, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=100, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=64, kernel_initializer='normal', activation='relu')(x)
    prob_output = Dense(units=2, kernel_initializer='normal', activation='softmax')(x)

    model = Model(inputs=[indel], outputs=[prob_output])

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.5)
    adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)

    model.compile(optimizer=adamo, loss='binary_crossentropy', metrics=['accuracy'])

    logger.info('model summary:\n{}'.format(model.summary()))

    return model


def bed_file_to_dict(bed_file):
    ''' Create a dict to store intervals from a bed file.

    Arguments:
        bed_file: the file to load

    Returns:
        bed: dict where keys in the dict are contig ids
            values are a tuple of arrays the first array
            in the tuple contains the start positions
            the second array contains the end positions.
    '''
    bed = {}

    with open(bed_file)as f:
        for line in f:
            parts = line.split()
            contig = parts[0]
            lower = int(parts[1])
            upper = int(parts[2])

            if contig not in bed.keys():
                bed[contig] = ([], [])

            bed[contig][0].append(lower)
            bed[contig][1].append(upper)

    for k in bed.keys():
        bed[k] = (np.array(bed[k][0]), np.array(bed[k][1]))

    return bed


def in_bed_file(bed_dict, contig, pos):
    # Exclusive
    lows = bed_dict[contig][0]
    ups = bed_dict[contig][1]
    return np.any((lows < pos) & (pos < ups))


def split_data(datasets, subset_ratios, sequential=False):
    """

    Splits datasets in len(subset_ratios) subsets with ratios as specified in subset_ratios.
    If ratios don't sum up to 1, they are normalized first.

    :param list of array datasets: input datasets
    :param list of float subset_ratios: list of subsets with their respective ratios
    :param bool sequential: If set, the data is partitioned sequentially rather than shuffled
    :return: Subsets
    :rtype: list of array
    """

    samples = datasets[0].shape[0]
    sum_ratios = sum(subset_ratios)
    results_length = [int(x / sum_ratios * samples) for x in subset_ratios]

    indices = range(samples)

    if not sequential:
        random.shuffle(indices)

    results = []
    i = 0
    for n in results_length:
        results.append([x[indices[i:n+i]] for x in datasets])
        i += n

    return results


def plot_metric_history(history, output, title):
    """
    Plots metrics history as pdf.

    :param history:
    :param str output: output file name
    :param str title: plot title
    :return:
    """
    # list all data in history
    print(history.history.keys())

    row = 0
    col = 0
    num_plots = len(history.history) / 2.0  # valid and train plot together
    rows = 4
    cols = max(2, int(math.ceil(num_plots / float(rows))))

    f, axes = plt.subplots(rows, cols, sharex=True, figsize=(36, 24))
    for k in history.history.keys():

        if 'val' not in k:

            axes[row, col].plot(history.history[k])
            axes[row, col].plot(history.history['val_' + k])

            axes[row, col].set_ylabel(str(k))
            axes[row, col].legend(['train', 'valid'], loc='upper left')
            axes[row, col].set_xlabel('epoch')

            row += 1
            if row == rows:
                row = 0
                col += 1
                if row * col >= rows * cols:
                    break

    axes[0, 1].set_title(title)
    plt.savefig(output)


def get_callbacks(output_prefix, patience=2, batch_size=32):
    checkpointer = ModelCheckpoint(filepath=output_prefix + ".callbacks", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
    tensorboard = TensorBoard(log_dir=output_prefix + "/tensorboard", histogram_freq=1, batch_size=batch_size)
    return [checkpointer, earlystopper, tensorboard]


def tensor_to_str(tensor):
    return "".join(dna_encoder.inverse_transform(tensor))


def get_best_scoring_training_examples(model, training_data, n_best = 50):

    probs = model.predict(training_data[0])[:,0]

    # Best scoring positive training examples
    best_scoring_indices = np.argpartition(probs,-n_best)[-n_best:]
    positive_examples = zip(
        [tensor_to_str(t) for t in training_data[0][best_scoring_indices]],
        probs[best_scoring_indices].tolist(),
        training_data[1][best_scoring_indices],
        training_data[2][best_scoring_indices])

    # Best scoring negative training examples
    best_scoring_indices = np.argpartition(probs,n_best)[:n_best]
    negative_examples = zip(
        [tensor_to_str(t) for t in training_data[0][best_scoring_indices]],
        probs[best_scoring_indices].tolist(),
        training_data[1][best_scoring_indices],
        training_data[2][best_scoring_indices]
    )

    return positive_examples, negative_examples


def apply_model(indels, model, reference, ambiguous_bases, window_size = 1000):
    result = {}
    for chrom, positions in reference.iteritems():
        if not indels[chrom]:
            logger.info("Skipping chromosome {} as no indels found in VCF.".format(chrom))
        else:
            chrom_size = reference[chrom].shape(0)
            i = 0
            #TODO: Skip N's and ambiguous bases
            #Read from VCF
            while i < chrom_size - window_size:
                if overlaps_intervals(ambiguous_bases,i, i+window_size):
                    continue
                pred = model.predict(positions[i:i+window_size])
                i+= window_size



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', help='One-hot-encoded reference (required if not using --ref_fasta)', required=False)
    parser.add_argument('--ref_fasta', help='Reference fasta. If used, the reference will be one-hot-encoded and saved as a npy files. (required if not using --ref)', required=False)
    parser.add_argument('--vcf', help='VCF containing indel training examples', required=True)
    parser.add_argument('--output', help='Output prefix', required=True)
    parser.add_argument('--window_size', help='Size of the window to consider on each side of the indel (default 30).',
                        required=False, default=30, type=int)
    parser.add_argument('--max_training',
                        help='Maximum number of training examples to use for each class (default 100,000). Set to -1 to use all.',
                        required=False, default=100000, type=int)
    parser.add_argument('--neg_train_window',
                        help='Size of the window to consider on each side of the indel to sample a negative training example (default 1000). If set to a value < 1 => select a random position on the same chromosome.',
                        required=False, default=1000, type=int)
    parser.add_argument('--train_model', help='Trains a new model. Required if not using --callback.', required=False,
                        action='store_true')
    parser.add_argument('--callback', help='When specified, will initialize the model weights with the given callback.', required=False)
    parser.add_argument('--save_best_examples', help='When specified, saves the best training examples (for each class).',
                        action='store_true', required=False)
    args = parser.parse_args()

    if int(args.ref_fasta is not None) + int(args.ref is not None) != 1:
        sys.exit("One and only one of --ref_fasta or --ref is required.")

    if not args.train_model and not args.callback:
        sys.exit("Must specify --callback when not using --train_model.")

    main(args)
