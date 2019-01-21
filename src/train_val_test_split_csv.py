import pandas as pd

def split_csv(filename, proportions=(0.8, 0.1, 0.1)):
    
    assert filename[filename.rfind('.'):] == '.csv'
    assert len(proportions) == 3
    assert sum(proportions) == 1
    
    orig_df = pd.read_csv(filename, index_col=0)
    shuffled_df = orig_df.sample(frac=1)
    assert len(orig_df) == len(shuffled_df)
    
    filename_core = filename[:filename.rfind('.')]
    train_filename = f'{filename_core}_train.csv'
    val_filename = f'{filename_core}_val.csv'
    test_filename = f'{filename_core}_test.csv'
    
    thresh1 = int(proportions[0]*len(shuffled_df))
    thresh2 = thresh1 + int(proportions[1]*len(shuffled_df))
    
    shuffled_df[:thresh1].to_csv(train_filename)
    shuffled_df[thresh1:thresh2].to_csv(val_filename)
    shuffled_df[thresh2:].to_csv(test_filename)