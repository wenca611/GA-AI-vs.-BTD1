"""
Project Name: Bloons TD 1 vs. AI
Description: GA AI playing BTD 1 using screen and mouse
Author: Bc. Václav Pastušek
Date: 30.07.2023
Python: 3.9
Version: 1.3
"""

# Import necessary libraries
from typing import Optional
import pygetwindow as gw  # Library for managing windows on the system
import pytesseract  # Library for using Tesseract OCR engine
import pyautogui  # Library for GUI automation
import cv2  # OpenCV's library for computer vision tasks
import numpy as np  # NumPy's library for numerical operations
from fuzzywuzzy import fuzz  # Library for fuzzy string matching
from collections import defaultdict  # Defaultdict for handling default values in dictionaries
import time  # Library for working with time and timing functions
import random  # Library for generating random numbers and selections
import pyperclip  # Library for interacting with the clipboard (copy/paste operations)
import copy  # Library for shallow and deep copy operations


# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'D:/GAMES/Tesseract/tesseract.exe'


def save_data_to_file(tower_map, tower_price, sell_price, file_path):
    """
    Save data to a file in a custom simple format.

    Args:
        tower_map (list): 2D array of values for tower_map.
        tower_price (list): 1D array of values for tower_price.
        sell_price (list): 1D array of values for sell_price.
        file_path (str): Path to the file where data will be saved.
    """
    with open(file_path, "w") as file:
        # Writing tower_map
        file.write("tower_map\n")
        for row in tower_map:
            file.write(" ".join(str(val) for val in row))
            file.write("\n")

        # Writing tower_price
        file.write("tower_price\n")
        file.write(" ".join(str(val) for val in tower_price))
        file.write("\n")

        # Writing sell_price
        file.write("sell_price\n")
        file.write(" ".join(str(val) for val in sell_price))
        file.write("\n")


def load_data_from_file(file_path):
    """
    Load data from a file in a custom simple format.

    Args:
        file_path (str): Path to the file from which you want to load data.

    Returns:
        tuple: Tuple containing the loaded data: (tower_map, tower_price, sell_price).
    """
    try:
        with open(file_path, "r") as file:
            data_lines = file.readlines()
    except FileNotFoundError:
        # If the file doesn't exist, return empty data
        return [], [], []

    tower_map = []
    tower_price = []
    sell_price = []

    flags = [False]*3
    for line in data_lines:
        print("Got line:", line[:150])
        line = line.strip()
        if line == "tower_map":
            if any(flags):
                print("Problem 1")
                exit(1)
            else:
                flags[0] = True
        elif line == "tower_price":
            if not flags[0] or flags[1] or flags[2]:
                print("Problem 2")
                exit(1)
            else:
                flags[:2] = [False, True]
        elif line == "sell_price":
            if flags[0] or not flags[1] or flags[2]:
                print("Problem 3")
                exit(1)
            else:
                flags[1:3] = [False, True]
        else:
            if flags[0]:
                rows = [val.replace("[", "").replace("]", "") for val in line.split("] [")]
                int_rows = []
                for row in rows:
                    int_rows.append([int(val) for val in row.split(", ")])
                tower_map.append(int_rows)
            if flags[1]:
                tower_price.extend([int(val) for val in line.split()])
            if flags[2]:
                sell_price.extend([int(val) for val in line.split()])

    # print("Tower map:", tower_map)
    # print("Buy:", tower_price)
    # print("Sell:", sell_price)

    return tower_map, tower_price, sell_price


def get_text(screenshot, a, b, c, d, number_flag=False) -> str:
    """
    Extracts text from a specific region of the screenshot using OCR (Optical Character Recognition).

    Parameters:
        screenshot (PIL.Image): The full screenshot from which text needs to be extracted.
        a, b, c, d (int): The coordinates defining the region to be cropped from the screenshot.
        number_flag (bool, optional): Specifies whether the extracted text should be limited to digits only.
                                      Default is False.

    Returns:
        str: The recognized text extracted from the specified region.
    """
    # Crop the area containing the digits
    region = screenshot.crop((a, b, c, d))

    # Move the mouse to the edges of the cropped area with a 0.1 or 0.2-second delay
    if 0:  # for debuging
        pyautogui.moveTo(a, b, duration=0.1)
        pyautogui.moveTo(c, b, duration=0.2)
        pyautogui.moveTo(c, d, duration=0.1)
        pyautogui.moveTo(a, d, duration=0.2)

    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a black and white image
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use pytesseract to recognize the text with English alphabet and digits only
    config = "--psm 6"
    if number_flag:
        config = "--psm 6 digits"

    return pytesseract.image_to_string(threshold, config=config)


def get_number(a, b, c, d) -> Optional[int]:
    """
    Extracts and returns a number from the specified area of the screen using OCR (Optical Character Recognition).

    Parameters:
        a, b, c, d (int): The coordinates defining the area containing the numbers.
                         The numbers are assumed to be located at position x=a, y=b, width=c-a, height=d-b.

    Returns:
        Optional[int]: The extracted number as an integer, or None if no valid number is found.
    """
    # Hover the mouse over the area with numbers (assuming numbers are at position x=a, y=b, width=c-a, height=d-b)
    pyautogui.moveTo(a, b, duration=0.02)
    pyautogui.dragTo(c, d, duration=0.02, button='left')

    # Simulate Ctrl+C hotkey to copy the text to clipboard
    pyautogui.hotkey('ctrl', 'c')

    # Wait for the simulated text copy
    time.sleep(0.05)

    # Get the content of the clipboard
    copied_text = pyperclip.paste()

    # Click outside to reset the selection
    pyautogui.click(1200, 300)

    # Convert the copied text to an integer (if possible)
    if copied_text.isdigit():
        return int(copied_text)
    else:
        return None


def gen_xy_pos(tower_type, tower_map) -> (int, int):
    """
    Generates random (x, y) coordinates within the map area, avoiding specific restricted regions.

    Args:
        tower_type (int): Index of the tower type.
        tower_map (list): Tower map containing the weight values.

    Returns:
        (int, int): A tuple containing the randomly generated x and y coordinates.
    """
    # Map area
    a, b, c, d = 561, 230, 1139, 825
    original_shape = (len(tower_map[tower_type]), len(tower_map[tower_type][0]))

    weights = [num for row in tower_map[tower_type] for num in row]
    positions = [num for num in range(len(weights))]

    # Start measuring time
    # start_time = time.time()

    # Generate random position based on weights  TODO optimalization
    gen_pos = random.choices(positions, weights, k=1)[0]

    # End measuring time
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.6f} seconds")

    # 1D -> 2D index conversion TODO control
    gen_y, gen_x = divmod(gen_pos, original_shape[1])
    gen_x += a
    gen_y += b

    return gen_x, gen_y


def fitness_function(round, money, lives) -> int:
    """
    Calculates the fitness score of an individual based on the provided parameters.

    Args:
        round (int): The round reached in the game.
        money (int): The amount of money the player has.
        lives (int): The number of lives remaining.


    Returns:
        int: The fitness score of the individual, calculated as the sum of the given parameters.
    """
    return int(round * 5e7 + lives * 1e6 + money)


def play_population(population) -> list:
    """
    Simulates the game for each individual in the population and calculates their scores.

    Args:
        population (list): A list of individuals representing the population.

    Returns:
        list: A list containing the scores of each individual.
    """
    score = []
    for idx, genotype in enumerate(population):
        print("\nTrying individual:", idx + 1)
        print("genotype", genotype)
        """Bring the game window to the foreground at each level
        When 'Start Round' is visible, click on it
        Detect game over with 'Lives:0' and start a new game"""
        round = 1
        money = 650
        lives = 40
        gen = []
        if genotype:
            gen = genotype.pop(0)

        while lives > 0:
            # Bring the game window to the foreground
            window.activate()

            # Get a screenshot of the game using pyautogui
            time.sleep(0.05)
            screenshot = pyautogui.screenshot()

            start = get_text(screenshot, 1170, 740, 1320, 780)
            # print("ratio for Start round:", fuzz.ratio(start[:-1], "Start Round"))
            if fuzz.ratio(start[:-1], "Start Round") > 30:
                while True:
                    old_money = get_number(1335, 292, 1240, 292)

                    if gen:
                        tower_type = gen[0]
                        tower_xy = gen[1]
                        if tower_type == 0:
                            pyautogui.click(1184, 400)
                        elif tower_type == 1:
                            pyautogui.click(1214, 400)
                        elif tower_type == 2:
                            pyautogui.click(1250, 400)
                        elif tower_type == 3:
                            pyautogui.click(1286, 400)
                        elif tower_type == 4:
                            pyautogui.click(1324, 400)

                        pyautogui.click(tower_xy)
                        pyautogui.click(tower_xy)

                        new_money = get_number(1335, 292, 1240, 292)

                        # print("money:", old_money, new_money)

                        if old_money - new_money > 0:
                            print("Placing tower type", tower_type, "at position", tower_xy)
                            if genotype:
                                gen = genotype.pop(0)
                        else:
                            break

                print("clicking on start")
                pyautogui.click(1245, 760)

            time.sleep(5)  # pause before next control

            lives = get_number(1335, 320, 1240, 320)
            # print("Lives remaining:", lives)
            if lives == 0:
                round = get_number(1335, 260, 1240, 260)
                money = get_number(1335, 292, 1240, 292)
                score += [fitness_function(round, money, lives)]
                print("Game over click")
                pyautogui.click(900, 520)
                break
            else:
                screenshot = pyautogui.screenshot()
                win = get_text(screenshot, 620, 420, 1120, 470)
                win_ratio = fuzz.ratio(win[:-1], "CONGRATULATIONS!")
                # print("WIN ratio:", win_ratio, win[:-1])

                if win_ratio > 35:
                    round = get_number(1335, 260, 1240, 260)
                    money = get_number(1335, 292, 1240, 292)
                    score += [fitness_function(round, money, lives)]
                    print("Win click")
                    pyautogui.click(850, 540)
                    break

    print("Best score index:", score.index(max(score)))
    print("Best individual:", population[score.index(max(score))])

    return score


def selection(population, scores) -> list:
    """
    Selects the best individuals from the population based on their scores.

    Args:
        population (list): A list of individuals representing the population.
        scores (list): A list of scores corresponding to each individual in the population.

    Returns:
        list: A list containing the selected best individuals without repetition.
    """
    # Get indices of individuals sorted by score from highest to lowest
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

    # Calculate the number of the best individuals to keep
    elite_size = int(0.2 * len(population))

    # Select best individuals without repetition
    selected_indices = set()
    selected_population = []
    for index in sorted_indices:
        if len(selected_population) >= 2:
            break

        if index not in selected_indices:
            selected_population.append(population[index])
            selected_indices.add(index)

    # If there are still not enough selected individuals, choose random individuals until we have 2
    while len(selected_population) < 2:
        random_index = random.choice(range(len(population)))
        if random_index not in selected_indices:
            selected_population.append(population[random_index])
            selected_indices.add(random_index)

    return selected_population


def crossover(parent1, parent2):
    """
    Performs crossover of genes between two parents to create offspring.

    Args:
        parent1 (list): A list representing the genes of the first parent.
        parent2 (list): A list representing the genes of the second parent.

    Returns:
        tuple: A tuple containing two children (offspring) created through gene crossover.
    """
    # Get the lengths of genes from parents
    len_parent1 = len(parent1)
    len_parent2 = len(parent2)

    # Determine the crossover point
    crossover_point = random.randint(0, min(len_parent1, len_parent2))

    # Create offspring from gene crossover
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2


def mutation(child, mutation_prob=0.1) -> list:
    """
    Performs mutation on a child's genes with a given probability.

    Args:
        child (list): A list representing the genes of the child.
        mutation_prob (float, optional): Probability of mutation. Defaults to 0.1.

    Returns:
        list: The mutated child's genes.
    """
    a, b, c, d = 561, 230, 1139, 825
    a1, b1, c1, d1 = 561, 230, 645, 304
    a2, b2, c2, d2 = 561, 786, 599, 825

    for i, gene in enumerate(child):
        if random.random() < mutation_prob:
            x = random.randint(0, 2)

            if x == 0:
                # Mutation of the first part of the gene to a random number from 0 to 5
                child[i][0] = random.randint(0, 4)
            elif x == 1 and len(gene) >= 2:
                # Mutation of the second part of the gene to a new x-coordinate
                while True:
                    gen_x = random.randint(a, c)
                    gen_y = gene[1][1]  # Preserve the original y-coordinate

                    if a1 < gen_x < c1 and b1 < gen_y < d1:
                        continue

                    if a2 < gen_x < c2 and b2 < gen_y < d2:
                        continue

                    child[i][1] = (gen_x, gen_y)
                    break
            elif x == 2 and len(gene) >= 2:
                # Mutation of the second part of the gene to a new y-coordinate
                while True:
                    gen_x = gene[1][0]  # Preserve the original x-coordinate
                    gen_y = random.randint(b, d)

                    if a1 < gen_x < c1 and b1 < gen_y < d1:
                        continue

                    if a2 < gen_x < c2 and b2 < gen_y < d2:
                        continue

                    child[i][1] = (gen_x, gen_y)
                    break

    return child


if __name__ == '__main__':
    # Inform the user about the purpose of the code
    print("AI vs. BTD1")

    # Check if the game window is open, otherwise exit
    game_title = "Bloons TD"
    windowsNames = gw.getWindowsWithTitle(game_title)

    # Filter out any other windows that might have similar names
    for name in windowsNames:
        if not name.title.endswith(game_title) and not name.title.startswith(game_title):
            windowsNames.remove(name)

    game_title_count = len(windowsNames)
    window = None

    # Handle different scenarios with the game window
    if game_title_count < 1:
        print("The game is not running.")
        exit(1)
    elif game_title_count > 1:
        print("Multiple instances of the game are running.")  # TODO
        exit(1)
    else:
        window = windowsNames[0]

    # Bring the game window to the foreground
    print("Bringing the game to the foreground")
    window.activate()

    # Check if the text 'New game' is displayed and click on it if found
    print("If 'New game' text is displayed => click on it")
    print("Reading text using screenshots and comparing")

    # Get a screenshot of the game using pyautogui
    time.sleep(0.01)
    screenshot = pyautogui.screenshot()

    # Area of NEW GAME
    a, b, c, d = 745, 355, 1030, 425

    print("What I see:", get_text(screenshot, a, b, c, d)[:-1], end=" ")

    # Initialize an empty set to store unique text options
    text_options = set()

    # Measure time before running the code
    start_time: float = time.time()
    text = get_text(screenshot, a, b, c, d)
    print(fuzz.ratio(text[:-1], "NEW GAME"))

    results = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if abs(i) != abs(j):
                results += [get_text(screenshot, a + i, b + j, c + i, d + j)]

    # Measure time after running the code
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")

    # Initialize a dictionary to store the frequency count of each unique text option
    frequency_count = defaultdict(int)
    fuzz_ratios = []

    # Update text frequencies and calculate fuzz ratio
    for text in results:
        frequency_count[text] += 1
        fuzz_ratios.append(fuzz.ratio(text, "NEW GAME"))

    # Calculate the average fuzz ratio
    mean_fuzz_ratio = sum(fuzz_ratios) / len(fuzz_ratios)

    # Print unique texts with frequencies and fuzz ratio
    print("Unique text options with fuzz ratio:")
    for text, count in frequency_count.items():
        ratio = fuzz.ratio(text, "NEW GAME")
        print(text[:-1], count, ratio)

    # Print the mean fuzz ratio
    print("Mean Fuzz Ratio:", mean_fuzz_ratio)

    print("Click on NEW GAME if the ratio is above 35%, otherwise someone else has already done it")

    if mean_fuzz_ratio > 35:
        print("Clicked on NEW GAME")
        pyautogui.click(874, 401)  # new game click

    print("Identifying the cost of buildings, upgrades, and sales that can be purchased right at the beginning - TODO")

    filepath = "tower_data.txt"

    tower_map, tower_price, sell_price = load_data_from_file(filepath)

    if not tower_map:
        # Map area
        a, b, c, d = 561, 230, 1139, 825
        # Areas where clicking on the map is not allowed
        a1, b1, c1, d1 = 561, 230, 645, 304
        a2, b2, c2, d2 = 561, 786, 599, 825

        map_weights = [[100]*(c+1-a) for _ in range(d+1-b)]

        # like A[y][x] = A[[x1],[x2],...]
        for j in range(0, d + 1 - b):
            for i in range(0, c + 1 - a):
                if 0 <= i <= c1-a1 and 0 <= j <= d1-b1:
                    map_weights[j][i] = 0
                elif 0 <= i <= c2-a2 and 0 <= j <= d2-b2:
                    map_weights[j][i] = 0

        tower_map = [copy.deepcopy(map_weights) for _ in range(5)]
        tower_price = [int(1e9) for _ in range(5)]
        sell_price = [0 for _ in range(5)]
        cells_to_reset = []
        tower_bought_flag = [False]*5

        for tower_type in range(len(tower_map)):
            for j in range(0, d + 1 - b):  # 0-595
                for i in range(0, c + 1 - a):  # 0-578
                    if i % 10 == 0 and j % 10 == 0 and tower_map[tower_type][j][i] != 0:  # grid 10x10px
                        if i == 0 or j == 0:
                            continue

                        pyautogui.press('esc')

                        old_money = get_number(1335, 292, 1240, 292)

                        if tower_type == 0:
                            pyautogui.click(1184, 400)
                        elif tower_type == 1:
                            pyautogui.click(1214, 400)
                        elif tower_type == 2:
                            pyautogui.click(1250, 400)
                        elif tower_type == 3:
                            pyautogui.click(1286, 400)
                        elif tower_type == 4:
                            pyautogui.click(1324, 400)

                        pyautogui.click(i+a, j+b)
                        pyautogui.click(i+a, j+b)

                        new_money = get_number(1335, 292, 1240, 292)

                        if new_money < old_money:
                            tower_bought_flag[tower_type] = True
                            tower_map[tower_type][j][i] *= int(1e6)
                            # buy
                            if tower_price[tower_type] == 1e9:
                                tower_price[tower_type] = old_money-new_money

                            # sell
                            pyautogui.click(i + a, j + b)
                            pyautogui.click(1270, 700)

                            if sell_price[tower_type] == 0:
                                after_sell_money = get_number(1335, 292, 1240, 292)
                                sell_price[tower_type] = after_sell_money-new_money

                            # change value for other towers
                            for other_tower_type in range(len(tower_map)):
                                if other_tower_type != tower_type:
                                    tower_map[other_tower_type][j][i] *= 5

                            if new_money+sell_price[tower_type] < tower_price[tower_type]:
                                # reset
                                pyautogui.click(1310, 811)

                        else:
                            tower_map[tower_type][j][i] //= 10
                            # change value for other towers
                            cells_to_reset.append((tower_type, j, i))
                            for other_tower_type in range(len(tower_map)):
                                if other_tower_type != tower_type:
                                    tower_map[other_tower_type][j][i] //= 2

        for idx, elem in enumerate(tower_bought_flag):
            if not elem:  # not enough money -> reverse values
                for tower_type, j, i in cells_to_reset:
                    tower_map[tower_type][j][i] *= 10

    save_data_to_file(tower_map, tower_price, sell_price, filepath)

    print("The first generation will have 20 individuals and genes from 0 to 50")

    population = []
    for _ in range(20):  # number of individuals in the population 10+
        # number of possible genes in the genotype
        genes_size = random.randint(0, 50)
        genotype = []
        for _ in range(genes_size):
            # tower type
            tower_type = random.randint(0, 4)
            # tower position
            tower_position = gen_xy_pos(tower_type, tower_map)
            genotype += [[tower_type, tower_position]]

        population += [genotype]

    print("Population length:", len(population))
    print(population)
    print("Population test")
    scores = play_population(population)

    # number of generations 10+
    for i in range(10):
        print("\nGeneration", i + 1)
        # Select the best individuals for the next generation
        selected_population = selection(population, scores)

        # Create a new generation
        new_generation = []

        # Crossover and mutation
        while len(new_generation) < len(population):
            parent1 = random.choice(selected_population)
            # print("parent1", parent1, "from pop:", selected_population)
            selected_population.remove(parent1)
            # print("selected population - parent1:", selected_population)
            parent2 = random.choice(selected_population)
            selected_population.append(parent1)

            child1, child2 = crossover(parent1, parent2)

            # Mutation probability for the first child
            if random.random() < 0.02:
                child1 = mutation(child1)

            # Mutation probability for the second child
            if random.random() < 0.02:
                child2 = mutation(child2)

            # Add both offspring to the new generation
            new_generation.append(child1)
            new_generation.append(child2)

        # Set the new generation as the current population for the next iteration
        population = new_generation

        # Calculate the score of the new generation
        scores = play_population(population)

        print("Population:", population)
        print("Score:", scores)
        print("Best in this generation:", scores.index(max(scores)), population[scores.index(max(scores))])

    # Return the individual with the best score from all generations
    best_individual = population[scores.index(max(scores))]
    print("Best individual ever:", scores.index(max(scores)), best_individual)


