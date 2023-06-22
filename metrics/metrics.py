import sys
import json
import numpy as np

def evaluate_results(json_file):
    with open(seen_json_file, 'r') as file:
        data = json.load(file)


    def evaluate(ground_truth, prediction):
        accu_1 = 0.
        accu_10 = 0.
        accu_100 = 0.
        length = len(ground_truth)
        for i in range(length):
            if ground_truth[i] in prediction[i][:100]:
                accu_100 += 1
                if ground_truth[i] in prediction[i][:10]:
                    accu_10 += 1
                    if ground_truth[i] == prediction[i][0]:
                        accu_1 += 1
        return accu_1/length*100, accu_10/length*100, accu_100/length*100

    def evaluate_test(ground_truth, prediction):
        accu_1 = 0.
        accu_10 = 0.
        accu_100 = 0.
        length = len(ground_truth)
        pred_rank = []
        for i in range(length):
            try:
                pred_rank.append(prediction[i][:].index(ground_truth[i]))
            except:
                pred_rank.append(1000)
            if ground_truth[i] in prediction[i][:100]:
                accu_100 += 1
                if ground_truth[i] in prediction[i][:10]:
                    accu_10 += 1
                    if ground_truth[i] == prediction[i][0]:
                        accu_1 += 1
        return accu_1/length*100, accu_10/length*100, accu_100/length*100, np.median(pred_rank), np.sqrt(np.var(pred_rank))
        
    
    return "Evaluation completed successfully"

# Usage example
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the JSON output file as a command line argument.")
        sys.exit(1)

    output_file = sys.argv[1]
    evaluation_result = evaluate_results(output_file)
    print(evaluation_result)
