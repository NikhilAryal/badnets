import argparse
import pathlib
import time

import datetime
import pandas as pd
from dataset.dataset_builder import get_poisoned_dataset, build_testset, optimizer_picker
from torch.utils.data import DataLoader
from models import BadNet
from train_eval import *


def main():
    # Input -->  dataset, batch-size, epochs, rate, train
    parser = argparse.ArgumentParser(description='Accept and print user arguments')

    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Epoch num')
    parser.add_argument('--rate', type=float, default=0.1, help='Rate')
    parser.add_argument('--train', type=bool, default=True, help='Should train')
    parser.add_argument('--nb_classes', type=int, default=10, help='Num of classes')
    parser.add_argument('--trigger_label', type=int, default=1, help='Num of trigger label classes')
    parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
    parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')

    parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

    args = parser.parse_args()

    print("Input Information:")
    print(f"Dataset: {args.dataset}")
    print(f"Batch: {args.batch}")
    print(f"Epoch: {args.epochs}")
    print(f"Rate: {args.rate}")
    # print(f"Verbose mode: {'Enabled' if args.verbose else 'Disabled'}")
    
    
    # initial settings
    device = torch.device("cpu")
    
    # create related path
    pathlib.Path("./model_states/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True) 
    
    # Load dataset, model: poisoned
    print("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = get_poisoned_dataset(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)
    
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch, shuffle=True, num_workers=8)
    data_loader_clean_validation = DataLoader(dataset=dataset_val_clean, batch_size=args.batch, shuffle=True, num_workers=8)
    data_loader_poisoned_val = DataLoader(dataset=dataset_val_poisoned, batch_size=args.batch, shuffle=True, num_workers=8)
    
    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_picker(optimization='sgd', param=model.parameters(),  lr=0.01)
    
    basic_model_path = "./model_states/badnet-%s.pth" % args.dataset
    start_time = time.time()
    
    # Load / Train
    if args.load_local:
        print("## Load model from : %s" % basic_model_path)
        model.load_state_dict(torch.load(basic_model_path), strict=True)
        test_stats = evaluate_badnets(data_loader_clean_validation, data_loader_poisoned_val, model, device)
        print(f"Accuracy (Clean dataset): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate: {test_stats['asr']:.4f}")
    
    else:
        print(f"Start training for {args.epochs} epochs")
        stats = []
        for epoch in range(int(args.epochs)):
            train_stats = train_one_epoch(data_loader_train, model, loss, optimizer, device)
            test_stats = evaluate_badnets(data_loader_clean_validation, data_loader_poisoned_val, model, device)
            print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
            
            # save model 
            torch.save(model.state_dict(), basic_model_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
            }

            # save training stats
            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv("./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()