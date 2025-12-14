import time
import re

# Global variable to store context (e.g., the user's name)
user_data = {
    # Set the user's name for personalized responses
    "name": "Mansvi", 
    # Added specialization context for relevant responses
    "internship_specialization": "AI"
}

def get_response(user_input):
    """
    Determines the chatbot's response based on the user's input using
    predefined rules, now incorporating basic context and pattern matching.
    """
    global user_data
    text = user_input.lower().strip()
    
    # --- CONTEXT/PATTERN-BASED RESPONSES (More Complex Rules) ---

    # Rule 1: Greetings & Name Check
    if "hello" in text or "hi" in text or "hey" in text:
        if user_data["name"]:
            return f"Hello, {user_data['name']}! How can I assist you further today?"
        else:
            return "Hello! I am a rule-based AI assistant. What is your name?"

    # Rule 2: Storing the User's Name 
    # Uses a regular expression to look for phrases like "my name is [name]", "i am [name]", or "call me [name]"
    name_match = re.search(r"(?:my name is|i am|call me|you can call me)\s+([a-zA-Z]+)", text)
    if name_match:
        name = name_match.group(1).capitalize()
        user_data["name"] = name
        return f"Pleased to meet you, {name}! How can I assist you with your tasks?"
    
    # Rule 3: Math Calculation
    # Looks for simple calculations like "5 + 3" or "10 times 5"
    calc_match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", text)
    if calc_match:
        try:
            num1 = float(calc_match.group(1))
            operator = calc_match.group(2)
            num2 = float(calc_match.group(3))
            
            if operator == '+': result = num1 + num2
            elif operator == '-': result = num1 - num2
            elif operator == '*': result = num1 * num2
            elif operator == '/': 
                if num2 == 0:
                    return "I cannot divide by zero!"
                result = num1 / num2
            
            return f"The result of {num1} {operator} {num2} is {result:.2f}"
        except Exception:
            # Fallback for complex math expressions not covered
            return "I can handle simple arithmetic like '5 + 3', but more complex calculations are difficult for me right now."


    # --- KEYWORD-BASED RESPONSES (Existing Simple Rules) ---

    # Rule 4: Asking for the bot's name
    elif "your name" in text or "who are you" in text:
        return "I am a simple Python chatbot operating based on predefined rules."

    # Rule 5: Asking about capabilities/purpose
    elif "what can you do" in text or "what is your purpose" in text:
        return "I can respond to simple keywords, remember your name, and even perform basic math! Try asking me a calculation like '10 * 5'."
    
    # Rule 9: AI/ML Specialization Inquiry (Custom rule based on specialization)
    elif "ai" in text or "machine learning" in text or "data science" in text:
        if user_data["internship_specialization"] == "AI":
            return f"That's right. Your specialization is in {user_data['internship_specialization']}! Focus on understanding Machine Learning algorithms."
        return "I can talk about AI and Machine Learning, but remember I am currently rule-based."
        
    # Rule 10: Best Programming Language for AI
    elif "best language" in text or "ai language" in text or "best programming" in text:
        return "For Artificial Intelligence, Python is the best programming language due to its extensive support for libraries and frameworks."
        
    # Rule 6: Expressions of gratitude
    elif "thank you" in text or "thanks" in text:
        # Personalized thank you
        if user_data["name"]:
            return f"You are most welcome, {user_data['name']}! I am glad I could help."
        return "You are welcome! I'm pleased I was able to assist you."
    
    # Rule 7: Project reference
    elif "project" in text or "task" in text:
        return "I was built as a rule-based NLP project."
        
    # Rule 8: Inquiry about time
    elif "time" in text or "what is the time" in text:
        current_time = time.strftime("%H:%M:%S")
        return f"The current time is approximately {current_time}."

    # --- FALLBACK RESPONSE ---
    
    else:
        # Personalized fallback response
        if user_data["name"]:
            return f"Apologies, {user_data['name']}, I didn't understand that. Try asking for my name or a simple calculation (e.g., '10 * 5')."
        return "I apologize, I did not understand that. Can you please rephrase your query?"

def main():
    """
    The main function to run the chatbot loop.
    """
    # Main title
    print("🤖 Context-Aware Chatbot (Task 1) 🤖")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60) 
    
    # Main conversational loop
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                print("\nChatbot: Goodbye! Have a productive day!")
                break
                
            # Get and display the chatbot's response
            response = get_response(user_input)
            print(f"Chatbot: {response}")
            
        except EOFError:
            print("\nChatbot: Session ended. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

# Execute the main function when the script is run
if __name__ == "__main__":
    main()