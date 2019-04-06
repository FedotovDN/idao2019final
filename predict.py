import datetime
import numpy as np
import pickle
import ast
import sys
import catboost


MODELS_FILE1 = 'models1.pkl'
MODELS_FILE2 = 'models2.pkl'
OUTPUT_HEADER = 'datetime,target_{},target_{},target_{},target_{},target_{}'

HOUR_IN_MINUTES = 60
DAY_IN_MINUTES = 24 * HOUR_IN_MINUTES
WEEK_IN_MINUTES = 7 * DAY_IN_MINUTES

SHIFTS = [
    HOUR_IN_MINUTES // 4,
    HOUR_IN_MINUTES // 2,
    HOUR_IN_MINUTES,
    DAY_IN_MINUTES,
    DAY_IN_MINUTES * 2,
    DAY_IN_MINUTES * 3,
    WEEK_IN_MINUTES,
    WEEK_IN_MINUTES * 2]
WINDOWS = [
    HOUR_IN_MINUTES // 4,
    HOUR_IN_MINUTES // 2,
    HOUR_IN_MINUTES,
    DAY_IN_MINUTES,
    DAY_IN_MINUTES * 2,
    WEEK_IN_MINUTES,
    WEEK_IN_MINUTES * 2]


def extractor(dt, history, parameters):
    features = []

    features.append(dt.weekday())
    features.append(dt.hour)
    features.append(dt.minute)
    features.append(dt.month)
    if dt.hour < 6:
        features.append(1.0)
    else:
        features.append(0.0)
    if ((dt.weekday() >= 5) or ((dt.day == 23) and (dt.month == 2))
                              or ((dt.day == 8) and (dt.month == 3))
                              or ((dt.day == 9) and (dt.month == 3))
                              or ((dt.day == 30) and (dt.month == 4))
                              or ((dt.day == 1) and (dt.month == 5))
                              or ((dt.day == 2) and (dt.month == 5))
                              or ((dt.day == 9) and (dt.month == 5))
                              or ((dt.day == 11) and (dt.month == 6))
                              or ((dt.day == 12) and (dt.month == 6)) ):
        features.append(1.0)
    else:
        features.append(0.0)

    for shift in SHIFTS:
        for window in WINDOWS:
            if window > shift:
                continue
            if window == shift:
                features.append(sum(history[-shift:-1]))
            else:
                features.append(sum(history[-shift:-shift + window]))

    return np.array(features)


if __name__ == '__main__':
    models1 = pickle.load(open(MODELS_FILE1, 'rb'))
    models2 = pickle.load(open(MODELS_FILE2, 'rb'))

    input_header = input()
    output_header = OUTPUT_HEADER.format(*sorted(list(models1['models'].keys())))
    print(output_header)
    
    all_features = []
    all_queries = []
    while True:
        # read data, calculate features line by line for memory efficient
        try:
            raw_line = input()
        except EOFError:
            break
                    
        line = raw_line.split(',', 1)
        dt = datetime.datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
        history = list(map(int, line[1][2:-2].split(', ')))
        features = extractor(dt, history, None)
        
        all_features.append(features)
        all_queries.append(line[0])
    
    # predict all objects for time efficient
    predictions1 = []
    predictions2 = []
    for position, model in models1['models'].items():
        predictions1.append(model.predict(all_features))
    for position, model in models2['models'].items():
        predictions2.append(model.predict(all_features))
    
    for i in range(len(predictions[0])):
        print(','.join([all_queries[i]] + list(map(lambda x: str(x[i]), (predictions1 + predictions2)/2.0))))
