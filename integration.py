"""It runs all the entry points of the project
"""
#Command line execution
from subprocess import check_call
from sys import stdout

if __name__ == '__main__':
    commands = [
                    #Siamese tuple generation
                    'python siamese_input_tuples.py -i input_labels.csv -o input_tuples.csv -c Image Id --output_cols Anchor Sample Label -p 5 -n 5 -f',
                    
                    #Model generation
                    'python model_generation.py -n cnn -b mobilenet -a create -p "-u 155 -d 7"',
                    
                    #Training
                    'python siamese_train.py -m cnn_mobilenet -d "..\\Humpback Whale\\dataset\\train_preprocessed" -i input_data.csv  -c 128 -b 32 -r 0.0003 --input_data_training_set_id 1 -t samplewise_std_normalization=true --image_cols Image --label_col Id --session_id 1',
                    
                    #Prediction
                    'python predict.py -m cnn_mobilenet.session_id.0.set_id.0.epoch.1  -d "..\\Humpback Whale\\dataset\\train_preprocessed" -i input_data.csv --image_cols Image --label_col Id',
                    
                    #Compute input distribution
                    'python evaluate_inputs.py --input_data input_data.csv --label_col Id',

                    #Convert labels to integer classes
                    'python classify_labels.py --input_data input_data.csv --label_col Id --mapping_keys classes.map',
                   
                    #Rebalance the input
                    'python rebalance.py --input_data input_data.csv --label_col Id --output_file rebalanced_input_data.csv'
                ]

    for command in commands:
        print(command)

        check_call(command)

    