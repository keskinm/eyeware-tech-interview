# tech1

## Task1
    parser.add_argument('--work-dir',
                        default='./data/task1',
                        help='path of dir where data is generated and re-organized')
                        
Will generate data in work-dir and re-organize it accordingly to the assignment willing

ex: `python -m task1 --work-dir ./data/task1`
                        


## Task2

    parser = argparse.ArgumentParser(prog='task2')
    parser.add_argument('--files-input-dir',
                        default='./data/task2',
                        help='path of dir where text files are stored')

    parser.add_argument('--plot-output-dir',
                        default='./data',
                        help='path or dir where text files are stored')
                        
ex: `python -m task2 --files-input-dir ./data/task2 --plot-output-dir ./data`


## Task3

    parser.add_argument('--model-type',
                        choices=['two_conv', 'seven_conv'],
                        default='two_conv',
                        help='')

    parser.add_argument('-t',
                        '--test-model',
                        nargs='+',
                        help='model path and model name',
                        default=None)

    parser.add_argument('-r',
                        '--resume-model',
                        nargs='+',
                        help='model path and model name',
                        default=None)

    parser.add_argument('--train-batch-size', default=10, help='')

    parser.add_argument('--lr', default=0.005, type=float, help='')

    parser.add_argument('--train-epoch', default=20, type=int, help='')

    parser.add_argument('--seed', default=42, help='')

    parser.add_argument('--save-dir', default='./data/task3', help='')

    parser.add_argument('--optimizer',
                        choices=['adam', 'sgd'],
                        default='adam',
                        help='')

    parser.add_argument('--dump-metrics-frequency',
                        metavar='Batch_n',
                        default='10',
                        type=int,
                        help='Dump metrics every Batch_n batches')

    parser.add_argument(
        '--num-threads',
        default='0',
        type=int,
        help='Number of CPU to use for processing mini batches')

    parser.add_argument(
        '--simulated-dataset-size',
        default='80',
        type=int,
        help='Number of samples (train+val+test)')
        
ex:

Train:

`python -m --model-type two_conv`

`python -m --model-type seven_conv`

Validation mse metrics are stored by default in ./data/task3/metrics


Test : 

`python -m eval --test-model two_conv ./data/task3/models/two_conv.pth`

`python -m eval --test-model seven_conv ./data/task3/models/seven_conv.pth`



## Task4:

    parser.add_argument('--input-file-path',
                        default='./data/task4/chessboard_pattern_tech_interview.mp4',
                        help='path of input video')

    parser.add_argument('--output-dir',
                        default='./data',
                        help='')

    parser.add_argument('--board-dims',
                        default=(3, 3),
                        type=tuple,
                        help='')

    parser.add_argument('--track-method',
                        choices=['method_1', 'method_2'],
                        default='method_1',
                        help='track method in assignment')
                        
ex: `python -m task4 --input-file-path ./data/task4/chessboard_pattern_tech_interview.mp4 --track-method method_1`
