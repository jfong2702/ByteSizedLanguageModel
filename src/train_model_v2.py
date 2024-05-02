import neat
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from data import Data
import warnings
import graphviz
import matplotlib.pyplot as plt
import visualize
import networkx as nx

# Model Generations
GENERATIONS = 5

curr_path = os.getcwd()
file_path = os.path.join(curr_path, "dataset\\Tweetsv2.csv")
dataset = Data(file_path).data_file
# Load tweets from Data
tweets = dataset.tolist()

# Preprocess tweets
tweets = [tweet.split() for tweet in tweets]

# Split data into training and validation sets
train_tweets, val_tweets = train_test_split(tweets, test_size=0.2, random_state=42)

# Fit and transform the corpus
vectorizer = TfidfVectorizer()
vectorizer.fit([' '.join(tweet) for tweet in train_tweets])

def get_vectorizer():
    return vectorizer

def text_to_vector(text):
    vector = vectorizer.transform([' '.join(text)]).toarray()
    return vector.flatten()


def vector_to_text(vector):
    return vectorizer.inverse_transform(np.array([vector]))[0]



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        for tweet in train_tweets:
            input_vec = text_to_vector(tweet[:-1])  # Input is all words except the last
            target_vec = text_to_vector(tweet[1:])  # Target is all words except the first

            output = net.activate(input_vec)

            # Calculate cosine similarity with handling for zero denominator
            dot_product = np.dot(output, target_vec)
            output_norm = np.linalg.norm(output)
            target_norm = np.linalg.norm(target_vec)
            if output_norm == 0 or target_norm == 0:
                cosine_similarity = 0
            else:
                cosine_similarity = dot_product / (output_norm * target_norm)

            genome.fitness += cosine_similarity

        # Normalize fitness by the number of tweets
        genome.fitness /= len(train_tweets)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to GENERATIONS generations.
    winner = p.run(eval_genomes, GENERATIONS)
    # visualize_network(winner)

    node_names = {-1: 'A', -2: 'B', 0: 'C'}
    draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # Save the winner
    with open('TrainedModels/best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against validation data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for tweet in val_tweets:
        input_vec = text_to_vector(tweet[:-1])
        output = winner_net.activate(input_vec)
        output_text = vector_to_text(output)
        print("input {!r}, expected {!r}, got {!r}".format(tweet[:-1], tweet[1:], output_text))

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot

def visualize_network(winner):
    # Create a directed graph from the network
    G = nx.DiGraph()

    # Add nodes to the graph
    for node in winner.nodes.values():
        G.add_node(node.key)

    # Add edges to the graph
    for connection in winner.connections.values():
        G.add_edge(connection.input_node.key, connection.output_node.key)

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = 'config/neat_config.txt'
    run(config_path)