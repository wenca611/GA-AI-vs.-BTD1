# Bloons TD 1 vs. AI

## Description
This project aims to create an AI that plays the game "Bloons TD 1" using screen capture and mouse control. The AI is developed using a Genetic Algorithm (GA) to find the best strategy for purchasing and placing towers in the game.

## Author
- Name: Bc. Václav Pastušek
- Date: 30.07.2023

## Python Version
Python 3.9

## Version
1.1

## Required Libraries
The following libraries are required to run the project:
- pygetwindow: For managing windows on the system.
- pytesseract: For using Tesseract OCR engine to extract text from screenshots.
- pyautogui: For GUI automation, including mouse control.
- cv2 (OpenCV): For computer vision tasks, used for image processing.
- numpy: For numerical operations.
- fuzzywuzzy: For fuzzy string matching.
- collections: For handling default values in dictionaries.
- time: For working with time and timing functions.
- random: For generating random numbers and selections.
- pyperclip: For interacting with the clipboard (copy/paste operations).

## Execution
1. Ensure that the required libraries are installed using pip: `pip install pygetwindow pytesseract pyautogui opencv-python-headless numpy fuzzywuzzy pyperclip`
2. Set the Tesseract executable path in the `pytesseract.pytesseract.tesseract_cmd` variable. For instructions on how to install pytesseract and set the executable path, follow this guide: [How to Install Pytesseract Python Library](https://www.projectpro.io/recipes/what-is-pytesseract-python-library-and-do-you-install-it)
3. Run the script using Python.
4. The script will check if the game window of "Bloons TD 1" is open. If not, it will exit.
5. If multiple instances of the game are running, it will also exit.
6. The game window will be brought to the foreground, and the script will check if the "New game" text is displayed. If found, it will click on it.
7. The script will then identify the cost of buildings, upgrades, and sales that can be purchased at the beginning of the game.
8. It will create the first generation of individuals for the AI, where each individual represents a sequence of building purchases without upgrades.
9. The AI will play through multiple generations using the Genetic Algorithm, selecting the best individuals, applying crossover and mutation, and calculating scores.
10. The individual with the best score from all generations will be returned as the most successful strategy.

For video demonstrations and updates on this project, you can visit my YouTube channel: [LevelUpGA YouTube Channel](https://www.youtube.com/@LevelUpGA)

## TODO list for Project Bloons TD 1 vs. AI

- Ensure that buildings can't be placed on top of each other (except for nails and glue).
- Before running the first generation, test various building types to determine where they can and cannot be placed on the map.
- Consider alternative approaches to the genetic algorithm, such as using a different method instead of crossover.
- Investigate the cost of buildings and upgrades (Note: Costs may vary based on different map layouts and building positioning, especially for the village building).
- Evaluate the buildings and attempt to devise a strategy for optimal placement (Consider using various metrics to assess building effectiveness).
- Experiment with creating a neural network that takes into account different stats like money, current levels, and the buildings' positions to make recommendations on what buildings to purchase and where to place them.
