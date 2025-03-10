#!/usr/bin/env python3

import numpy as np
import random
import matplotlib.pyplot as plt

# Generate random coordinates for cities
def generate_cities(n):
    return np.random.rand(n, 2) * 100  # Cities in a 100x100 grid

# Compute the total distance of a given route
def route_distance(route, cities):
    dist = 0
    for i in range(len(route) - 1):
        dist += np.linalg.norm(cities[route[i]] - cities[route[i + 1]])
    dist += np.linalg.norm(cities[route[-1]] - cities[route[0]])  # Return to start
    return dist

# Generate an initial population of random routes
def initial_population(size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size)]

# Tournament selection (chooses the best of k random individuals)
def tournament_selection(population, fitnesses, k=5):
    selected = random.sample(range(len(population)), k)
    best_index = min(selected, key=lambda i: fitnesses[i])
    return population[best_index]

# Ordered Crossover (OX) to maintain valid paths
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    fill_values = [city for city in parent2 if city not in child]
    idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = fill_values[idx]
            idx += 1
    return child

# Swap mutation (swaps two cities in a route)
def swap_mutation(route, mutation_rate=0.2):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Evolve population
def evolve_population(population, cities, elite_size=2, mutation_rate=0.2):
    fitnesses = [route_distance(route, cities) for route in population]
    new_population = sorted(population, key=lambda x: route_distance(x, cities))[:elite_size]
    
    while len(new_population) < len(population):
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        child = ordered_crossover(parent1, parent2)
        child = swap_mutation(child, mutation_rate)
        new_population.append(child)
    
    return new_population

# Genetic Algorithm Execution
def genetic_algorithm(num_cities=20, pop_size=100, generations=500):
    cities = generate_cities(num_cities)
    population = initial_population(pop_size, num_cities)
    best_distance = float('inf')
    best_route = None
    
    for gen in range(generations):
        population = evolve_population(population, cities)
        current_best = min(population, key=lambda x: route_distance(x, cities))
        current_distance = route_distance(current_best, cities)
        
        if current_distance < best_distance:
            best_distance = current_distance
            best_route = current_best
        
        if gen % 50 == 0:
            print(f"Generation {gen}: Best Distance = {best_distance:.2f}")
    
    # Plot result
    plot_route(best_route, cities, best_distance)
    return best_route, best_distance

# Function to plot the best route
def plot_route(route, cities, distance):
    plt.figure(figsize=(8, 6))
    ordered_cities = np.array([cities[i] for i in route] + [cities[route[0]]])
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], marker='o', linestyle='-')
    plt.title(f"Best Route - Distance: {distance:.2f}")
    plt.savefig('genetic_algo.png')
    plt.show()

# Run the Genetic Algorithm
best_route, best_distance = genetic_algorithm()
