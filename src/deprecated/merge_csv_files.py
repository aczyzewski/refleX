import sys

    
def merge_files(gt, file, output="merged.csv", delimeter=','):

    anomalies = ["loop_scattering", "background_ring", "strong_background", "diffuse_scattering", "artifact", "ice_ring", "non_uniform_detector"]

    gt_content = open(gt, 'r').read().split('\n')
    file_content = open(file, 'r').read().split('\n')

    gt_header = gt_content[0].split(delimeter)[1:]
    file_header = file_content[0].split(delimeter)[1:]

    if not set(gt_header) == set(file_header):
        raise Exception

    file_dict, gt_dict = {}, {}

    for file_line in file_content[1:]:
        file_line = file_line.split(delimeter)

        #if file_line[0] in file_dict.keys():
            #print("DUPPLICATED: %s" % file_line[0])

        file_dict[file_line[0]] = dict(zip(file_header, list(map(int, file_line[1:]))))

    for file_line in gt_content[1:]:
        file_line = file_line.split(delimeter)
        gt_dict[file_line[0]] = dict(zip(gt_header, list(map(int, file_line[1:]))))

    intersection_size = 0
    
    print("GT size: %s" % (len(gt_dict.keys())))
    print("Input size: %s" % len(file_dict.keys()))

    for img_name in file_dict.keys():
        if not img_name in gt_dict.keys():
            gt_dict[img_name] = file_dict[img_name]
        else:
            intersection_size += 1

    print("Intersection size: %s" % intersection_size)
    print("Output file size: %s" % len(gt_dict.keys()))

    merged_files = open(output, 'w')
    merged_files.write("image," + ','.join(anomalies) + '\n')

    for img in gt_dict.keys():
        merged_files.write(img + "," + ','.join([str(gt_dict[img][anomaly]) for anomaly in anomalies]) + '\n')

    merged_files.close()

    print("--- DONE! ---")

    

if __name__ == "__main__":
    merge_files(*sys.argv[1:])