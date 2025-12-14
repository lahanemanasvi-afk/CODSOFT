Unbeatable Tic-Tac-Toe AI

Project Overview

This project implements the classic game of Tic-Tac-Toe (Noughts and Crosses) featuring an Unbeatable Artificial Intelligence (AI) opponent. The AI uses the Minimax Algorithm to calculate the optimal move for every possible game state, ensuring that the human player can never win; the game will always result in a win for the AI or a draw.

This task was completed as part of the CodSoft Web Development Internship.

Features

Unbeatable AI: The opponent uses the Minimax algorithm to play perfectly.

Two Players: Human (X) vs. AI (O).

Real-time Status Updates: Clear messages indicate the current player, game status (Win/Draw), and when the AI is 'thinking'.

Responsive Design: Styled using Tailwind CSS for a modern look that works well on both desktop and mobile devices.

Reset Functionality: A dedicated button to quickly start a new game.

Technologies Used

HTML5: For the basic structure of the game.

CSS3 (Tailwind CSS): Used for styling and ensuring a modern, responsive layout.

JavaScript (ES6): Used for all game logic, state management, and the core Minimax AI implementation.

How to Run Locally

You do not need any special server to run this game.

Save the File: Save the content of tictactoe.html to a file on your local machine.

Open in Browser: Locate the saved file (tictactoe.html) and double-click it.

Play: The game will open directly in your default web browser, and you can start playing immediately.

Minimax Algorithm

The AI's intelligence is built on the Minimax algorithm, a recursive, decision-making algorithm used in game theory.

Maximizing Player (AI / 'O'): The AI always tries to achieve the maximum possible score (leading to a win: +10).

Minimizing Player (Human / 'X'): The AI assumes the human player will always try to minimize the AI's score (leading to a loss: -10).

Depth: The algorithm calculates the score based on how quickly a win or loss is achieved (a faster win gets a higher score).

By exploring all future moves, the AI chooses the path that guarantees the best outcome for itself, regardless of the opponent's moves.