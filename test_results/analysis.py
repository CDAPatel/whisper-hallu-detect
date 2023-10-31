import pandas as pd

results = "C:\\dev\\whisper-hallu-detect\\test_results\\init_param_tests\\mean_85_height_85_with_std.csv"

def main():

    data = pd.read_csv(results)

    tot_hallu = 0
    tot_success = 0
    tot_error = 0
    # Not going to worry about generatitive vs rep for now.
    hallu_hallu = 0
    hallu_none = 0
    hallu_error = 0
    hallu_Derror = 0

    success_none = 0
    success_hallu = 0 
    success_error = 0
    success_Derror = 0

    error_none = 0
    error_error = 0
    error_hallu = 0
    error_Derror = 0

    tot = 0

    for idx, row in data.iterrows():
        if not pd.notna(row['detector']):
            break
        tot += 1

        if row['annotation'] == 'H':
            tot_hallu += 1
            if row['detector'] == 'H':
                hallu_hallu += 1
            elif row['detector'] == 'E':
                hallu_error += 1
            elif row['detector'] == 'N':
                hallu_none += 1
            else:
                hallu_Derror += 1

        elif row['annotation'] == 'E':
            tot_error += 1
            if row['detector'] == 'H':
                error_hallu += 1
            elif row['detector'] == 'E':
                error_error += 1
            elif row['detector'] == 'N':
                error_none += 1
            else:
                error_Derror += 1

        else:
            tot_success += 1
            if row['detector'] == 'H':
                success_hallu += 1
            elif row['detector'] == 'E':
                success_error += 1
            elif row['detector'] == 'N':
                success_none += 1
            else:            
                success_Derror +1

    print('Total Transcriptions: ', tot)
    print('Total Hallucinations: ', tot_hallu)
    print('Total Successes: ', tot_success)
    print('Total Errors: ', tot_error)

    print('For hallucinations:')
    print('Correctly identified as hallucination: ', hallu_hallu)
    print('Incorrectly identified as none: ',hallu_none)
    print('Incorrectly identified as errors: ', hallu_error)
    print('Detector error: ', hallu_Derror)

    print('For successes:')
    print('Correctly identified as none: ', success_none)
    print('Incorrectly identified as hallucination: ', success_hallu)
    print('Incorrectly identified as errors: ', success_error)
    print('Detector error: ', success_Derror)

    print('For errors:')
    print('Correctly identified as errors: ', error_error)
    print('Correctly identified as none: ',error_none)
    print('Incorrectly identified as hallucinations: ', error_hallu)
    print('Detector error: ', error_Derror)

if __name__ == "__main__":
    main()