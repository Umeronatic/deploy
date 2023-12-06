import pickle

import numpy as np

vec = pickle.load(open('tfidf.pkl','rb'))  # Initialize the vocabulary
# Load your trained model
deploy = pickle.load(open('/Users/macvision/PycharmProjects/pythonProject/spam.pkl', 'rb'))

user_input = input("Enter a message: ")
print()
input_feature = vec.transform([user_input])

decision_scores = deploy.decision_function(input_feature)

threshold = 0.25

if decision_scores > threshold:
    print("Your Message","--->",user_input)
    print()
    print("___________________________")
    print("Result","--> ","This message is spam :(.")
    print("___________________________")
else:
    print("Your Message","--->",user_input)
    print()
    print("___________________________")
    print("Result","--> ","This message is not spam :).")
    print("___________________________")
