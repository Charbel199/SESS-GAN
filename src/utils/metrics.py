import os
import json
import matplotlib.pyplot as plt

# Function to parse all generations and compute average metrics
def parse_and_plot_metrics(top_level_folder):
    generations = sorted(
        [d for d in os.listdir(top_level_folder) if d.startswith("generation")],
        key=lambda x: int(x.replace("generation", ""))
    )

    gen_numbers = []
    avg_fitness = []
    avg_collision = []
    avg_proximity = []
    avg_simulation = []

    for gen in generations:
        gen_path = os.path.join(top_level_folder, gen)
        metrics_path = os.path.join(gen_path, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        
        with open(metrics_path, "r") as f:
            data = json.load(f)
        
        fitness_vals = data["fitness"]
        add_metrics = data["additional_metrics"]

        #avg_fitness.append(sum(fitness_vals) / len(fitness_vals))
        avg_collision.append(sum(m["collisionCount"] for m in add_metrics) / len(add_metrics))
        avg_proximity.append(sum(m["proximityTime"] for m in add_metrics) / len(add_metrics))
        avg_simulation.append(sum(m["simulationTime"] for m in add_metrics) / len(add_metrics))

        # avg_fitness.append(max(fitness_vals)) 
        # avg_collision.append(max(m["collisionCount"] for m in add_metrics))
        # avg_proximity.append(max(m["proximityTime"] for m in add_metrics))
        # avg_simulation.append(max(m["simulationTime"] for m in add_metrics))


        gen_numbers.append(int(gen.replace("generation", "")))

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # axs[0, 0].plot(gen_numbers, avg_fitness, marker='o')
    # axs[0, 0].set_title("Average Fitness")
    # axs[0, 0].set_xlabel("Generation")
    # axs[0, 0].set_ylabel("Fitness")

    axs[0, 1].plot(gen_numbers, avg_collision, marker='o')
    axs[0, 1].set_title("Average Collision Count")
    axs[0, 1].set_xlabel("Generation")
    axs[0, 1].set_ylabel("Collision Count")

    axs[1, 0].plot(gen_numbers, avg_proximity, marker='o')
    axs[1, 0].set_title("Average Proximity Time")
    axs[1, 0].set_xlabel("Generation")
    axs[1, 0].set_ylabel("Proximity Time")

    axs[1, 1].plot(gen_numbers, avg_simulation, marker='o')
    axs[1, 1].set_title("Average Simulation Time")
    axs[1, 1].set_xlabel("Generation")
    axs[1, 1].set_ylabel("Simulation Time")

    plt.tight_layout()
    plt.show()
parse_and_plot_metrics("./src/assets/results15/ga")
