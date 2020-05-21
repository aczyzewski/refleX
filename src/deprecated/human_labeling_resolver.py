import os
import pandas
import argparse
from collections import defaultdict

choices = {
    1: 'all',               # Wszyscy oceniajacy sa zgodni
    2: 'majority',          # Wiekszosc oceniajacych sie zgadza
    3: 'one+',              # Co najmniej jeden oceniajacy sie zgadza
    4: 'fixed (0)'          # Nie ufamy nikomu (0)
}

comparators = {
    1: lambda x: x == 1.0,
    2: lambda x: x > 0.5,
    3: lambda x: x > 0,
    4: lambda x: False
}

def set_rules(file):
    
    anomalies = ["loop_scattering", "background_ring", "strong_background", "diffuse_scattering", "artifact", "ice_ring", "non_uniform_detector"]
    anomaly_rules = defaultdict(int)
    help_msg = ['[%s - %s]' % (int(key), value) for key, value in choices.items()]

    print('Choices: ' + ' '.join(help_msg))
    for anomaly in anomalies:
            anomaly_rules[anomaly] = comparators[int(input(anomaly + ": "))]

    return anomaly_rules

def apply_rules_to_file(file, output_file, rules, delimeter=','):
    
    file_content = open(file, 'r').read().split('\n')
    output_file = open(output_file, 'w')
    output_file.write(file_content[0])

    header = file_content[0].split(delimeter)[1:]

    for line in file_content[1:]:
        line = line.split(delimeter)
        results = [int(rules[header[idx]](float(value))) for idx, value in enumerate(line[1:])]
        output_file.write('\n' + line[0] + delimeter + delimeter.join(list(map(str, results))))

    return output_file      

def compare_files(file, gt, rules, delimeter=','):
    file_content = open(file, 'r').read().split('\n')
    gt_content = open(gt, 'r').read().split('\n')
    
    file_header = file_content[0].split(delimeter)[1:]
    gt_header = file_content[0].split(delimeter)[1:]
    
    for anomaly in file_header:
        if not anomaly in gt_header:
            print('[ERROR] Divergent headers! Exiting...')
            return

    removed_anomalies = []
    common_header = []

    for anomaly in file_header:
        if not rules[anomaly] is comparators[4]:
            common_header.append(anomaly)
        else:
            removed_anomalies.append(anomaly)
            print('[WARNING] Rule for %s was fixed! (omitting)' % anomaly)
        
    statistics, file_dict, gt_dict = {}, {}, {}
    
    for file_line in file_content[1:]:
        file_line = file_line.split(delimeter)
        file_dict[file_line[0]] = dict(zip(file_header, list(map(int, file_line[1:]))))

    for file_line in gt_content[1:]:
        file_line = file_line.split(delimeter)
        gt_dict[file_line[0]] = dict(zip(gt_header, list(map(int, file_line[1:]))))

    for anomaly in common_header:
        statistics[anomaly] = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    intersection_size = 0
    for file_name in file_dict.keys():
        if file_name in gt_dict.keys():
            for anomaly in common_header:
                if file_dict[file_name][anomaly]:
                    if gt_dict[file_name][anomaly]:
                        statistics[anomaly]['TP'] += 1
                    else:
                        statistics[anomaly]['FP'] += 1
                else:
                    if gt_dict[file_name][anomaly]:
                        statistics[anomaly]['FN'] += 1
                    else:
                        statistics[anomaly]['TN'] += 1
            intersection_size += 1

    print('Statistics:')
    print("  Intersection size: %d" % intersection_size)
    for anomaly in common_header:
        precision = round((statistics[anomaly]['TP'] / (statistics[anomaly]['TP'] + statistics[anomaly]['FP'])) * 100, 2)
        accuracy = round(((statistics[anomaly]['TP'] + statistics[anomaly]['TN']) / (statistics[anomaly]['TP'] + statistics[anomaly]['FP'] + statistics[anomaly]['TN'] + statistics[anomaly]['FN'])) * 100, 2)
        recall = round((statistics[anomaly]['TP'] / (statistics[anomaly]['TP'] + statistics[anomaly]['FN'])) * 100, 2)
        print("  %s (TP = %d | FP = %d | TN = %d | FN = %d | Acc = %s | Prec = %s | Rec = %s)" \
            % (anomaly, statistics[anomaly]['TP'], statistics[anomaly]['FP'], statistics[anomaly]['TN'], statistics[anomaly]['FN'], accuracy, precision, recall))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help="Input file.")
    parser.add_argument('-o', '--output', required=True, help="Output file.")
    parser.add_argument('-c', '--compare', default=None, help="Compare results to selected file")
    args = parser.parse_args()

    rules = set_rules(args.file)
    apply_rules_to_file(args.file, args.output, rules)

    if args.compare and os.path.exists(args.compare):
        print("[INFO] Comparing to %s..." % args.compare)
        compare_files(args.output, args.compare, rules)
    elif args.compare and not os.path.exists(args.compare):
        print("[WARINING] Argument -c (--compare) is incorrect! Omitting...")

    # + simple_grid_search?

    

