import pickle
import streamlit as st

deploy = pickle.load(open('/Users/macvision/PycharmProjects/pythonProject/spam.pkl', 'rb'))
vec = pickle.load(open('tfidf.pkl','rb'))  # Initialize the vocabulary


def predictions_system(user_input):
    input_feature = vec.transform([user_input])

    decision_scores = deploy.decision_function(input_feature)

    threshold = 0.25

    if decision_scores > threshold:
        return "Result  -->  This message is spam :("
    else:
        return "Result --> This message is not spam :)"


def main():
    st.title("Spam prediction")

    msg = st.text_input('Enter Your message')

    predict = ''

    if st.button("prediction"):
        predict = predictions_system(msg)

    st.success(predict)


if __name__ == '__main__':
    main()