import os
import cv2
import random
import numpy as np


# ============================================================
# Image Parameters
# ============================================================

WIDTH = 300
HEIGHT = 400
NUM_TRIANGLES = 100


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Image Loading
# ============================================================

def load_target_image(filepath):
    """Loads the target image, resizes it and converts it from BGR to RGB."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image {filepath} not found.")

    img = cv2.imread(filepath)

    if img is None:
        raise ValueError(f"Could not read image file: {filepath}")

    img = cv2.resize(img, (WIDTH, HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# ============================================================
# Representation
# ============================================================

def get_local_color(target_img, cx, cy, patch_size=5):
    """Gets the average RGB color around a given pixel."""
    half = patch_size // 2

    x_min = max(0, cx - half)
    x_max = min(WIDTH, cx + half + 1)

    y_min = max(0, cy - half)
    y_max = min(HEIGHT, cy + half + 1)

    patch = target_img[y_min:y_max, x_min:x_max]
    color = np.mean(patch.reshape(-1, 3), axis=0)

    return tuple(color.astype(int))


def create_random_triangle(target_img=None):
    """Creates one random triangle."""
    cx = random.randint(0, WIDTH - 1)
    cy = random.randint(0, HEIGHT - 1)

    size = random.randint(10, 150)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, 3))

    points = []

    for angle in angles:
        radius = random.randint(max(1, size // 3), size)

        x = int(cx + radius * np.cos(angle))
        y = int(cy + radius * np.sin(angle))

        x = int(np.clip(x, 0, WIDTH - 1))
        y = int(np.clip(y, 0, HEIGHT - 1))

        points.append((x, y))

    if target_img is not None:
        color = get_local_color(target_img, cx, cy)
    else:
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    return {
        "p1": points[0],
        "p2": points[1],
        "p3": points[2],
        "color": color,
        "alpha": random.uniform(0.02, 0.80),
        "z": random.random()
    }


def create_random_individual(target_img=None):
    """Creates an individual composed of NUM_TRIANGLES triangles."""
    return [
        create_random_triangle(target_img)
        for _ in range(NUM_TRIANGLES)
    ]


# ============================================================
# Rendering
# ============================================================

def get_background_color(target_img):
    """Gets the average RGB color of the target image."""
    return tuple(
        np.mean(target_img.reshape(-1, 3), axis=0).astype(int)
    )


def render_individual(individual, background_color=(0, 0, 0)):
    """Renders an individual into an RGB image."""
    canvas = np.full(
        (HEIGHT, WIDTH, 3),
        background_color,
        dtype=np.uint8
    )

    ordered_individual = sorted(
        individual,
        key=lambda tri: tri["z"]
    )

    for tri in ordered_individual:
        pts = np.array(
            [tri["p1"], tri["p2"], tri["p3"]],
            dtype=np.int32
        )

        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        color = np.array(tri["color"], dtype=np.float32)
        alpha = tri["alpha"]

        canvas_float = canvas.astype(np.float32)

        canvas_float[mask == 255] = (
            alpha * color +
            (1 - alpha) * canvas_float[mask == 255]
        )

        canvas = np.clip(canvas_float, 0, 255).astype(np.uint8)

    return canvas


# ============================================================
# Fitness
# ============================================================

def calculate_fitness(rendered, target):
    """Calculates RMSE between generated image and target image."""
    diff = rendered.astype(np.float32) - target.astype(np.float32)
    mse = np.mean(diff ** 2)

    return np.sqrt(mse)


# ============================================================
# Mutation
# ============================================================

def mutate(individual, target_img=None, mutation_rate=0.04):
    """Applies mutation to an individual."""
    new_individual = []

    for tri in individual:
        new_tri = tri.copy()

        step_pos = 5 if random.random() < 0.80 else 30
        step_col = 10 if random.random() < 0.80 else 60

        for point_key in ["p1", "p2", "p3"]:
            if random.random() < mutation_rate:
                x, y = new_tri[point_key]

                x = int(np.clip(
                    x + random.randint(-step_pos, step_pos),
                    0,
                    WIDTH - 1
                ))

                y = int(np.clip(
                    y + random.randint(-step_pos, step_pos),
                    0,
                    HEIGHT - 1
                ))

                new_tri[point_key] = (x, y)

        if random.random() < mutation_rate:
            r, g, b = new_tri["color"]

            r = int(np.clip(r + random.randint(-step_col, step_col), 0, 255))
            g = int(np.clip(g + random.randint(-step_col, step_col), 0, 255))
            b = int(np.clip(b + random.randint(-step_col, step_col), 0, 255))

            new_tri["color"] = (r, g, b)

        if random.random() < mutation_rate * 0.5:
            new_tri["alpha"] = float(np.clip(
                new_tri["alpha"] + random.uniform(-0.10, 0.10),
                0.02,
                0.80
            ))

        if "z" in new_tri and random.random() < mutation_rate * 0.3:
            new_tri["z"] = float(np.clip(
                new_tri["z"] + random.uniform(-0.10, 0.10),
                0.0,
                1.0
            ))

        if random.random() < mutation_rate * 0.4:
            cx = (
                new_tri["p1"][0] +
                new_tri["p2"][0] +
                new_tri["p3"][0]
            ) / 3

            cy = (
                new_tri["p1"][1] +
                new_tri["p2"][1] +
                new_tri["p3"][1]
            ) / 3

            for point_key in ["p1", "p2", "p3"]:
                x, y = new_tri[point_key]

                new_x = int(cx + (x - cx) * 0.8)
                new_y = int(cy + (y - cy) * 0.8)

                new_tri[point_key] = (
                    int(np.clip(new_x, 0, WIDTH - 1)),
                    int(np.clip(new_y, 0, HEIGHT - 1))
                )

        if random.random() < mutation_rate * 0.05:
            new_tri = create_random_triangle(target_img)

        new_individual.append(new_tri)

    return new_individual


# ============================================================
# Selection
# ============================================================

def tournament_selection(population, fitnesses, tournament_size=5):
    """Selects one parent using tournament selection."""
    candidate_indices = random.sample(
        range(len(population)),
        tournament_size
    )

    best_index = min(
        candidate_indices,
        key=lambda idx: fitnesses[idx]
    )

    return population[best_index]


def roulette_selection(population, fitnesses):
    """Roulette selection adapted for minimization."""
    fitnesses = np.array(fitnesses, dtype=np.float64)

    adjusted = 1 / (fitnesses + 1e-8)
    probabilities = adjusted / adjusted.sum()

    selected_index = np.random.choice(
        len(population),
        p=probabilities
    )

    return population[selected_index]


def select_parent(
    population,
    fitnesses,
    selection_type="tournament",
    tournament_size=5
):
    """Selects one parent according to the selected strategy."""
    if selection_type == "tournament":
        return tournament_selection(
            population,
            fitnesses,
            tournament_size
        )

    if selection_type == "roulette":
        return roulette_selection(
            population,
            fitnesses
        )

    raise ValueError(f"Unknown selection type: {selection_type}")


# ============================================================
# Crossover
# ============================================================

def one_point_crossover(parent_1, parent_2):
    """Performs one-point crossover."""
    split = random.randint(1, NUM_TRIANGLES - 1)

    child = parent_1[:split] + parent_2[split:]

    return [triangle.copy() for triangle in child]


def uniform_crossover(parent_1, parent_2):
    """Performs uniform crossover."""
    child = []

    for tri_1, tri_2 in zip(parent_1, parent_2):
        if random.random() < 0.5:
            child.append(tri_1.copy())
        else:
            child.append(tri_2.copy())

    return child

def spatial_crossover(parent_1, parent_2):
    """Crossover by spatial zone of the image.
    Triangles whose center is in the upper half of the image come from parent 1.
    Triangles whose center is in the lower half come from parent 2.
    Exploits the fact that triangles cover distinct spatial zones."""
    child = []

    for tri_1, tri_2 in zip(parent_1, parent_2):
        # Calculate center of triangle from parent 1
        cx = (tri_1["p1"][0] + tri_1["p2"][0] + tri_1["p3"][0]) / 3
        cy = (tri_1["p1"][1] + tri_1["p2"][1] + tri_1["p3"][1]) / 3
        
        # Upper half from parent 1, lower half from parent 2
        if cy < HEIGHT / 2:
            child.append(tri_1.copy())
        else:
            child.append(tri_2.copy())
    return child

def no_crossover(parent_1, parent_2):
    """No crossover - child is a copy of parent 1, only mutation applies.
    Used to evaluate the isolated contribution of crossover."""
    return [triangle.copy() for triangle in parent_1]

def apply_crossover(
    parent_1,
    parent_2,
    crossover_type="one_point"
):
    """Applies the selected crossover strategy."""
    if crossover_type == "one_point":
        return one_point_crossover(parent_1, parent_2)

    if crossover_type == "uniform":
        return uniform_crossover(parent_1, parent_2)
    
    if crossover_type == "none":
        return no_crossover(parent_1, parent_2)
    
    if crossover_type == "spatial":
        return spatial_crossover(parent_1, parent_2)

    raise ValueError(f"Unknown crossover type: {crossover_type}")


# ============================================================
# Diversity Metrics
# ============================================================

def individual_to_vector(individual):
    genes = []
    for tri in individual:
        (x1, y1), (x2, y2), (x3, y3) = tri['p1'], tri['p2'], tri['p3']
        r, g, b = tri['color']
        genes.extend([
            x1 / WIDTH, x2 / WIDTH, x3 / WIDTH,
            y1 / HEIGHT, y2 / HEIGHT, y3 / HEIGHT,
            r / 255.0, g / 255.0, b / 255.0,
            float(np.clip(tri.get('alpha', 0.0), 0.0, 1.0)),
            float(np.clip(tri.get('z', 0.0), 0.0, 1.0)),
        ])
    return np.asarray(genes, dtype=np.float32)


def genotypic_variance(population):
    """Variância média por locus na população."""
    M = np.vstack([individual_to_vector(ind) for ind in population])
    return float(np.mean(np.var(M, axis=0)))


# ============================================================
# Configurable Genetic Algorithm
# ============================================================

def evolve_configurable(
    target_img,
    pop_size=30,
    generations=300,
    mutation_rate=0.05,
    elite_size=3,
    selection_type="tournament",
    tournament_size=5,
    crossover_type="one_point",
    crossover_rate=0.9,
    use_mutation_decay=False,
    min_mutation_rate=0.01,
    snapshot_generations=None,
    print_every=None,
    track_diversity=False
):
    """
    Runs a configurable Genetic Algorithm.
    Supports different selection, crossover and mutation configurations.
    """
    background_color = get_background_color(target_img)

    population = [
        create_random_individual(target_img)
        for _ in range(pop_size)
    ]

    best_fitness_history = []
    mutation_rate_history = []
    snapshots = {}

    if snapshot_generations is None:
        snapshot_generations = []

    snapshot_generations = set(
        gen for gen in snapshot_generations
        if 0 <= gen < generations
    )

    best_individual = None
    diversity_history = []

    for gen in range(generations):
        rendered_images = [
            render_individual(ind, background_color)
            for ind in population
        ]

        fitnesses = [
            calculate_fitness(rendered, target_img)
            for rendered in rendered_images
        ]

        sorted_indices = np.argsort(fitnesses)

        population = [
            population[i]
            for i in sorted_indices
        ]

        fitnesses = [
            fitnesses[i]
            for i in sorted_indices
        ]

        current_best_fit = fitnesses[0]
        best_individual = population[0]

        best_fitness_history.append(current_best_fit)

        if track_diversity:
            diversity_history.append(genotypic_variance(population))

        
        if use_mutation_decay:
            current_mutation_rate = max(
                min_mutation_rate,
                mutation_rate * (1 - gen / generations)
            )
        else:
            current_mutation_rate = mutation_rate

        mutation_rate_history.append(current_mutation_rate)

        if gen in snapshot_generations:
            best_rendered = render_individual(
                best_individual,
                background_color
            )

            snapshots[gen] = {
                "image": best_rendered.copy(),
                "rmse": current_best_fit,
                "mutation_rate": current_mutation_rate
            }

        new_population = [
            individual.copy()
            for individual in population[:elite_size]
        ]

        while len(new_population) < pop_size:
            parent_1 = select_parent(
                population,
                fitnesses,
                selection_type=selection_type,
                tournament_size=tournament_size
            )

            parent_2 = select_parent(
                population,
                fitnesses,
                selection_type=selection_type,
                tournament_size=tournament_size
            )

            if random.random() < crossover_rate:
                child = apply_crossover(
                    parent_1,
                    parent_2,
                    crossover_type=crossover_type
                )
            else:
                child = [
                    triangle.copy()
                    for triangle in parent_1
                ]

            child = mutate(
                child,
                target_img,
                current_mutation_rate
            )

            new_population.append(child)

        population = new_population

        if print_every is not None:
            if gen % print_every == 0 or gen == generations - 1:
                print(
                    f"Generation {gen:05d} | "
                    f"Best RMSE: {current_best_fit:.4f} | "
                    f"Mutation: {current_mutation_rate:.4f}"
                )

    return {
        "best_individual": best_individual,
        "history": best_fitness_history,
        "background_color": background_color,
        "snapshots": snapshots,
        "mutation_rate_history": mutation_rate_history,
        "best_rmse": best_fitness_history[-1],
        "diversity_history": diversity_history if track_diversity else None
    }