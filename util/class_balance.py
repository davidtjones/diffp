def class_balance(dataset):
    class_count = {'0':0, '1':0, '2':0, '3':0, '4':0}

    for idx, row in dataset.dr_frame.iterrows():
        class_count[str(row['level'])]+=1

    return class_count
    
