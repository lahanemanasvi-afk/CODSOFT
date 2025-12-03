Task 1: Rule-Based Chatbot in Python

Project Overview

This project implements a simple, rule-based chatbot using Python. It serves as an introductory demonstration of Natural Language Processing (NLP) concepts, focusing on conditional logic and pattern matching to maintain a basic conversation flow.

Key Features

Context Management: The chatbot remembers and uses the user's name (pre-set to 'Mansvi') for personalized greetings and fallbacks.

Pattern Matching (Regex): Utilizes Python's re module for advanced input detection, such as parsing simple arithmetic expressions (e.g., 10 * 5).

AI Specialization: Includes specific rules to handle queries related to AI, Machine Learning, and ideal programming languages.

Rule-Based Responses: Uses a clear if/elif/else structure for predefined responses.

Prerequisites

Python 3.x installed on your system.

How to Run

Save the Code: Ensure the Python script is saved as chatbot.py.

Open Terminal: Navigate to the directory where you saved the file.

Execute the Script: Run the following command in your terminal:

py chatbot.py



(Note: If py doesn't work, use the full path to your Python executable or just python chatbot.py.)

Interact: Type your queries and conversation will begin. Type exit or quit to end the session.

Example Interactions

User Input

Expected Behavior

Hello

Greets you by name (Mansvi).

what is 10 plus 5

Performs the calculation.

what is machine learning

Responds with AI specialization context.

who are you

Identifies itself as a rule-based chatbot.

thank you

Gives a personalized thank you.