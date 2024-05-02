import pickle
from train_model_v2 import text_to_vector
from sklearn.feature_extraction.text import TfidfVectorizer
import neat
import numpy as np
from train_model_v2 import get_vectorizer

model_path = 'TrainedModels/best_genome.pkl'

# Load the trained genome
with open(model_path, 'rb') as f:
    winner = pickle.load(f)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config/test_config.txt')

# Create a neural network from the winner genome
net = neat.nn.FeedForwardNetwork.create(winner, config)

def generate_response(input_text, vectorizer):
    input_vec = text_to_vector(input_text)
    output_vec = net.activate(input_vec)

    # Convert the output vector to text
    response_text = vector_to_text(output_vec, vectorizer)

    return response_text


def vector_to_text(vector, vectorizer):
    # Get feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Sort vector indices by value to identify important words
    sorted_indices = np.array(vector).argsort()[::-1]

    # Extract top words from the vector
    top_words = [feature_names[i] for i in sorted_indices if vector[i] > 0]

    # Join top words to form the text representation
    text = ' '.join(top_words)

    return text

def test():
    vectorizer = get_vectorizer()
    input_text = input("Say something to TweetBot: ")
    print()
    response = generate_response(input_text, vectorizer)
    print("TweetBot:", response)

if __name__ == "__main__":
    test()

