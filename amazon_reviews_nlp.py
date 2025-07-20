# amazon_reviews_nlp.py

import spacy

# 1. Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# 2. Sample product reviews
reviews = [
    "I absolutely love the Apple AirPods. The sound quality is amazing!",
    "The Samsung Galaxy S21 is overpriced and underperforms. Very disappointed.",
    "Sony headphones are great for the price. Good bass and noise cancellation.",
    "Terrible experience with the Lenovo ThinkPad. It crashed on day one.",
    "I’m impressed with the JBL Flip 5 speaker. It’s loud and waterproof!"
]

# 3. Define simple rule-based sentiment analyzer
positive_words = ["love", "great", "amazing", "impressed", "good", "excellent", "loud"]
negative_words = ["terrible", "disappointed", "bad", "crashed", "overpriced", "underperforms"]

# 4. Process reviews
for i, review in enumerate(reviews):
    doc = nlp(review)
    
    # Extract named entities
    print(f"\nReview {i+1}: {review}")
    print("Named Entities (Product/Brand):")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")

    # Simple rule-based sentiment
    review_lower = review.lower()
    pos_score = sum(word in review_lower for word in positive_words)
    neg_score = sum(word in review_lower for word in negative_words)
    
    if pos_score > neg_score:
        sentiment = "Positive"
    elif neg_score > pos_score:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"Sentiment: {sentiment}")
