import neat
import numpy as np
import pickle
from data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
import TensorFlow as tf
import os

# Model Generations
GENERATIONS = 1000

curr_path = os.getcwd()
file_path = os.path.join(curr_path, "dataset\\Tweets.csv")
dataset = Data(file_path).data_file
# Load tweets from Data
tweets = dataset.tolist()
vectorizer = TfidfVectorizer()
# Fit and transform the corpus
corpus = ["This is an example.", "Another example example."]
vectorizer.fit(corpus)

# Initialize TensorFlow to utilize GPU (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_vectorizer():
    return vectorizer

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        for tweet in tweets:
            input_vec = text_to_vector(tweet)
            output_vec = text_to_vector(tweet)  # Output is the same as input for this example

            output = net.activate(input_vec)
            genome.fitness += np.linalg.norm(np.array(output) - np.array(output_vec))

def text_to_vector(text):
    vector = vectorizer.transform([text]).toarray()
    return vector.flatten()

def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config/neat_config.txt')

    p = neat.Population(config)

    # Add reporters for statistics
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    winner = p.run(eval_genomes, GENERATIONS)

    # Save the winner
    with open('TrainedModels/best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    main()
