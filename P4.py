import pandas as pd
data = pd.read_csv(r'prog4.csv')

print(data)

def find_s_algorithm(data):
    attributes = data.iloc[:, :-1].values 
    target = data.iloc[:, -1].values 

    # Step 1: Initialize hypothesis with first positive example
    for i in range(len(target)):
        if target[i] == "Yes": 
            hypothesis = attributes[i].copy()
            break


    # Step 2: Update hypothesis based on other positive examples
    for i in range(len(target)):
        if target[i] == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != attributes[i][j]:
                    hypothesis[j] = '?' 
    return hypothesis


# Run Find-S Algorithm
final_hypothesis = find_s_algorithm(data)


# Print the learned hypothesis
print("Most Specific Hypothesis:", final_hypothesis)
