"""It runs all the entry points of the project
"""
#Command line execution
from subprocess import check_call
from sys import stdout

#Commandline arguments
from argparse import ArgumentParser

commands = [
            #Siamese tuple generation
            'python siamese_input_tuples.py -i tests/store/label_df.csv -o input_tuples.csv -c Image Id --output_cols Anchor Sample Label -s 10 -f',

            #Siamese triplet generation
            'python siamese_input_tuples.py --triplets -i tests/store/label_df.csv -o input_tuples.csv -c Image Id --output_cols Anchor Positive Negative -s 10 -f',

            #Image augmentation
            'python augment.py -d dataset/train -o dataset/train_preprocessed -i dataset/train.csv -n 10 -c Image -s 224 224 --output_file input_data.batch.0.epoch.0.csv',

            #Convert labels to integer classes
            'python classify_labels.py --input_data input_data.batch.0.epoch.0.csv --label_col Id --mapping_keys classes.map',

            #Model generation
            'python model_generation.py -n cnn -b mobilenet -d 7',

            #Training
            'python train.py -m cnn_mobilenet -d "dataset/train_preprocessed" -c 128 -b 32 -r 0.0003 --batch_id 0 --epoch_id 0 -t samplewise_std_normalization=true --image_cols Image --label_col Id',

            #Prediction
            'python predict.py -m cnn_mobilenet.batch.0.epoch.0  -d "dataset/train_preprocessed" -i input_data.batch.0.epoch.0.csv --image_cols Image --label_col Id',

            #Compute input distribution
            'python evaluate_inputs.py --input_data input_data.batch.0.epoch.0.csv --label_col Id',

            #Rebalance the input
            'python rebalance.py --input_data input_data.batch.0.epoch.0.csv --label_col Id --output_file rebalanced_input_data.csv',

            #Consolidate results
            'python consolidate_result.py -e epoch_0 epoch_1'

            ]

def parse_args():
    parser = ArgumentParser(description = 'It runs the project entry points')

    parser.add_argument(
            '-i', '--resume_index',
            default = 0, type = int,
            help = 'It specifies the entry point from which to resume the execution.')

    args = parser.parse_args()

    return args.resume_index

if __name__ == '__main__':
    #Parse commandline arguments
    resume_index = parse_args()


    for index in range(resume_index, len(commands)):
        #Extract command
        command = commands[index]

        print('Index: ', index, command)

        check_call(command, shell = True)
