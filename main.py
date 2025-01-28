from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
review = "The product is good."
result = classifier(review)
print("Sentiment Analysis Result:", result)

# Text Summarization
summarizer = pipeline("summarization", model="t5-small", min_length=10, max_length=10)
text = """Mental health is just as important as physical health, yet it has often been neglected or stigmatized. 
In recent years, however, there has been a significant shift in the conversation surrounding mental health. 
People are starting to recognize the importance of mental well-being and how it impacts various aspects of life, 
including work, relationships, and overall quality of life.

Many individuals experience mental health challenges at some point in their lives, whether it's dealing with stress, 
anxiety, depression, or other mental illnesses. Unfortunately, due to the stigma, many people do not seek help, 
which can lead to long-term negative effects on their health.

Raising awareness about mental health is crucial in breaking down these barriers. By fostering open conversations, 
providing education, and offering support, society can work together to create an environment where people feel 
safe seeking help. Mental health awareness campaigns can also play a key role in spreading knowledge about available 
resources and treatments.

The goal should be to create a society where mental health is prioritized, and individuals are empowered to take care 
of their mental well-being without fear of judgment or discrimination. It's important to remember that taking care 
of one's mental health is not a sign of weakness but a necessary part of leading a healthy, fulfilling life."""

summary = summarizer(text)
print("Summary Result:", summary[0]['summary_text'])
