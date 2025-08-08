import faiss
import numpy as np
import pandas as pd
import requests
import json
from groq import Groq
import re
import time
import hashlib

client = Groq(api_key="GROQ_API_KEY")

df = pd.read_csv("C:/Users/sanja/Desktop/GENAI/filtered_recipenlg_50k.csv")
embeddings = np.load("C:/Users/sanja/Desktop/GENAI/recipe_embeddings.npy", allow_pickle=True)
recipe_ids = np.load("C:/Users/sanja/Desktop/GENAI/recipe_ids.npy", allow_pickle=True)

df = df[df["id"].isin(recipe_ids)]

sorted_indices = np.argsort(recipe_ids)
recipe_ids = recipe_ids[sorted_indices]
embeddings = embeddings[sorted_indices]
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


JINA_API_URL = "JINA_API_KEY"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer jina_ca33f621ce834d449167883e66e98172Ux7fhR58FXoSD3u-usCd8eUcdZKS"
}


def get_query_embedding(text):
    data = {
        "model": "jina-clip-v2",
        "dimensions": 1024,
        "normalized": True,
        "embedding_type": "float",
        "input": [{"text": text}]
    }

    response = requests.post(JINA_API_URL, headers=HEADERS, json=data)

    if response.status_code == 200:
        response_data = json.loads(response.text)

        embedding_vector = np.array(response_data["data"][0]["embedding"], dtype=np.float32)

        return embedding_vector
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def retrieve_similar_recipes(user_query, k=5):
    query_embedding = get_query_embedding(user_query)

    if query_embedding is None:
        print("❌ Error generating query embedding. Try again.")
        return None

    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    matched_ids = [recipe_ids[i] for i in indices[0] if i < len(recipe_ids)]

    results = df[df["id"].isin(matched_ids)]
    return results[["title", "ingredients", "directions"]]

# Track previously generated recipes to avoid duplication
recipe_history = {}

def generate_recipe(user_query, retrieved_recipes, feedback=None):
    global recipe_history
    
    # Create a hash of the user query to track recipe history
    query_hash = hashlib.md5(user_query.lower().strip().encode()).hexdigest()
    
    # Get history count for this query - how many times have we generated recipes for it
    history_count = recipe_history.get(query_hash, 0)
    recipe_history[query_hash] = history_count + 1
    
    # Increase temperature based on history to encourage variation
    # Start at 0.7 and increase up to 1.0 for repeated queries
    temperature = min(0.7 + (history_count * 0.1), 1.0)
    
    retrieved_text = "\n\n".join(
        f"Title: {row['title']}\nIngredients: {row['ingredients']}\nDirections: {row['directions']}"
        for _, row in retrieved_recipes.iterrows()
    )

    # Add feedback to prompt if available
    feedback_text = ""
    if feedback:
        feedback_text = f"""
        IMPORTANT - Previous generation attempt had these issues:
        {feedback}
        
        Please address these specific issues in your new recipe and create a VALID food combination.
        Focus on creating harmonious flavor profiles and logical ingredient combinations.
        Consider completely changing the approach if necessary - don't just make minor adjustments.
        """
    
    # Add variation encouragement if this is a repeated query
    variation_text = ""
    if history_count > 0:
        variation_text = f"""
        IMPORTANT: This is request #{history_count+1} for these same ingredients. 
        Please generate a COMPLETELY DIFFERENT recipe than before.
        Be creative and explore different cooking styles, cuisines, or preparation methods.
        Consider:
        - Different cooking methods (baking, frying, steaming, etc.)
        - Different cuisine inspirations (Italian, Asian, Mexican, etc.)
        - Different textures and presentations
        - Different flavor profiles (spicy, sweet, savory, etc.)
        """

    prompt = f"""
    The user wants a recipe with the following ingredients: {user_query}.

    Here are some similar recipes:
    {retrieved_text}
    
    {feedback_text}
    
    {variation_text}

    Based on these, generate a **new recipe** that:
    - Uses the user-provided ingredients in a LOGICAL and HARMONIOUS way
    - Creates VALID flavor combinations that would actually taste good
    - Follows a structured format: **Title, Ingredients, Directions**
    - Ensures the recipe makes culinary sense (appropriate cooking techniques, temperatures, etc.)
    - The last line should be the "name of recipe"
    
    IMPORTANT: Focus on creating a recipe that will pass a validity check for logical food combinations!
    """

    # Add a timestamp seed to increase randomness
    seed = int(time.time()) % 10000

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert chef AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=0.95,
        seed=seed,
        stream=True,
        stop=None,
    )

    generated_text = ""
    for chunk in completion:
        generated_text += chunk.choices[0].delta.content or ""

    return generated_text

def perform_validity_check(generated_recipe):
    new_prompt = f"""
    This is the generated recipe: {generated_recipe}.

    Perform a detailed validity check on the generated recipe to determine if the food combination is valid or not.
    Evaluate against these specific metrics for food combinations:
    
    1. Flavor Profile Compatibility
    - Are there compatible flavor compounds between ingredients?
    - Do the flavors work well together or clash?
    
    2. Taste Balance
    - Is there a good balance of the five basic tastes: sweet, salty, sour, bitter, and umami?
    - Does any single taste overwhelm or conflict with others?
    
    3. Texture Harmony
    - Do the textures complement or contrast pleasantly?
    - Are there any conflicting textures that create an unpleasant mouthfeel?
    
    4. Cultural & Contextual Expectations
    - Does the combination make sense within some culinary tradition?
    - Is the combination logical in the context of how we normally eat food?
    
    5. Temperature & Serving Logic
    - Do the hot/cold elements work well together?
    - Is the serving method appropriate for the ingredients?
    
    6. Ingredient Function
    - Are ingredients being used in appropriate ways (bases, condiments, highlight flavors)?
    - Is there a clear main ingredient and supporting elements?
    
    IMPORTANT: Start your response with either "VALID:" or "INVALID:" followed by your detailed assessment.
    If invalid, clearly explain which aspects need to be fixed and suggest specific improvements.
    If valid, explain what makes this a successful recipe.
    
    Be decisive and direct in your evaluation.
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert chef AI and food critic with deep knowledge of culinary science and flavor combinations."},
            {"role": "user", "content": new_prompt}
        ],
        temperature=0.7,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    generated_new_text = ""
    for chunk in completion:
        generated_new_text += chunk.choices[0].delta.content or ""

    return generated_new_text

def is_recipe_valid(validity_result):
    """
    Check if the recipe is valid based on the validity check result.
    Returns True if valid, False if invalid.
    """
    # Check if the response begins with the expected format
    if validity_result.strip().upper().startswith("VALID:"):
        return True
    if validity_result.strip().upper().startswith("INVALID:"):
        return False
    
    # If the new format isn't used, fall back to the previous method
    # Convert to lowercase for easier matching
    lower_result = validity_result.lower()
    
    # Look for clear indicators of invalidity
    invalid_indicators = [
        "invalid", "not valid", "illogical", "doesn't work", "does not work", 
        "poor combination", "bad combination", "incompatible", "clash", "disjointed",
        "lacks balance", "disrupts the harmony", "unconventional", "may not appeal"
    ]
    
    # Look for clear indicators of validity
    valid_indicators = [
        "valid", "well-balanced", "harmonious", "complementary", "works well",
        "good combination", "flavorful", "tasty", "delicious", "balanced"
    ]
    
    # First check for explicit invalidity statements
    for indicator in invalid_indicators:
        if indicator in lower_result:
            return False
    
    # Then check for explicit validity statements
    for indicator in valid_indicators:
        if indicator in lower_result:
            return True
    
    # If we reach here, the validity is ambiguous
    # Count the number of positive vs negative phrases to make a decision
    positive_count = sum(1 for word in ["good", "nice", "great", "excellent", "perfect", "balanced", "complementary"] 
                         if word in lower_result)
    negative_count = sum(1 for word in ["not", "doesn't", "don't", "clash", "odd", "unusual", "strange"] 
                         if word in lower_result)
    
    # If there are significantly more positive words than negative ones, consider it valid
    return positive_count > negative_count

def parse_recipe_sections(recipe_text):
    """
    Parse a recipe text to extract title, ingredients, and directions sections.
    Returns a dictionary with these sections.
    """
    # Define patterns to match each section - handle both markdown formats
    title_pattern = r'(?:\*\*Title:\*\*|^[*#]+\s*Title:\s*|\*\*)(.*?)(?=\*\*Ingredients|\n\s*\*\*Ingredients|\n\s*###\s*Ingredients|Ingredients:|\Z)'
    ingredients_pattern = r'(?:\*\*Ingredients:\*\*|Ingredients:|###\s*Ingredients:)(.*?)(?=\*\*Directions|Directions:|###\s*Directions:|\Z)'
    directions_pattern = r'(?:\*\*Directions:\*\*|Directions:|###\s*Directions:)(.*)'
    
    # Find matches
    title_match = re.search(title_pattern, recipe_text, re.DOTALL | re.IGNORECASE)
    ingredients_match = re.search(ingredients_pattern, recipe_text, re.DOTALL | re.IGNORECASE)
    directions_match = re.search(directions_pattern, recipe_text, re.DOTALL | re.IGNORECASE)
    
    # Extract content or use empty string if not found
    title = title_match.group(1).strip() if title_match else ""
    ingredients = ingredients_match.group(1).strip() if ingredients_match else ""
    directions = directions_match.group(1).strip() if directions_match else ""
    
    # Clean up directions - remove any "name of recipe" line that might be at the end
    if directions:
        lines = directions.split('\n')
        # Remove empty lines and possible "name of recipe" at the end
        clean_lines = [line for line in lines if line.strip()]
        if len(clean_lines) > 1 and not clean_lines[-1].startswith('1') and not clean_lines[-1].startswith('-'):
            # Last line might be the recipe name, remove it
            directions = '\n'.join(clean_lines[:-1])
        else:
            directions = '\n'.join(clean_lines)
    
    return {
        "title": title,
        "ingredients": ingredients,
        "directions": directions
    }

# Main program loop
def main():
    global recipe_history
    print("🍳 Welcome to the RAG Recipe Generator! 🍳")
    print("Enter ingredients, and I'll generate a recipe for you.")
    print("Type 'exit' to quit the program.")
    print("Type 'clear' to clear recipe history and start fresh.")
    
    last_ingredients = None
    
    while True:
        # If we have last ingredients and user wants to regenerate, use those
        if last_ingredients is not None:
            print(f"\nCurrent ingredients: {last_ingredients}")
            user_input = input("🔸 Enter ingredients, 'more' for another recipe with same ingredients, 'exit' to quit, 'clear' to reset: ")
        else:
            # Otherwise ask for new ingredients
            user_input = input("\n🔸 Enter ingredients (or 'exit' to quit, 'clear' to reset): ")
        
        # Check if user wants to exit
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("👋 Thanks for using the RAG Recipe Generator. Goodbye!")
            break
        
        # Check if user wants to clear history
        if user_input.lower() == 'clear':
            recipe_history = {}
            last_ingredients = None
            print("📋 Recipe history cleared. Next recipe will be fresh!")
            continue
            
        # Check if user wants another recipe with same ingredients
        if user_input.lower() in ['more', 'another', 'new'] and last_ingredients is not None:
            user_input = last_ingredients
            print(f"\n🔄 Generating another recipe with: {user_input}")
        else:
            # New ingredients entered, save them
            last_ingredients = user_input
        
        # Retrieve similar recipes
        retrieved_recipes = retrieve_similar_recipes(user_input, k=5)
        
        if retrieved_recipes is not None:
            print("\n🔹 Top Matching Recipes:")
            print(retrieved_recipes)
            
            # First attempt at recipe generation
            generated_recipe = generate_recipe(user_input, retrieved_recipes)
            print("\n=== FIRST GENERATED RECIPE ===")
            print(generated_recipe)
            
            # Perform validity check
            validity_result = perform_validity_check(generated_recipe)
            print("\n=== VALIDITY CHECK RESULT ===")
            print(validity_result)
            
            # Keep regenerating until we get a valid recipe or hit max attempts
            max_attempts = 5
            attempt_count = 1
            
            while not is_recipe_valid(validity_result) and attempt_count < max_attempts:
                print(f"\n=== RECIPE INVALID, REGENERATING (ATTEMPT {attempt_count+1}/{max_attempts})... ===")
                
                # Prepare detailed feedback based on validity result
                specific_feedback = f"""
                {validity_result}
                
                To fix this recipe, you need to:
                1. Reconsider flavor compatibility between ingredients
                2. Ensure taste balance (sweet, salty, sour, bitter, umami)
                3. Create logical texture combinations
                4. Respect cultural and contextual food expectations
                5. Consider completely changing the approach with these ingredients
                """
                
                # Regenerate recipe with feedback
                regenerated_recipe = generate_recipe(user_input, retrieved_recipes, feedback=specific_feedback)
                print(f"\n=== REGENERATED RECIPE (ATTEMPT {attempt_count+1}) ===")
                print(regenerated_recipe)
                
                # Update the generated recipe to use the regenerated one
                generated_recipe = regenerated_recipe
                
                # Perform a new validity check on the regenerated recipe
                validity_result = perform_validity_check(generated_recipe)
                print(f"\n=== VALIDITY CHECK RESULT (ATTEMPT {attempt_count+1}) ===")
                print(validity_result)
                
                attempt_count += 1
            
            if not is_recipe_valid(validity_result):
                print("\n❌ Sorry, we couldn't generate a valid recipe after multiple attempts.")
                print("Please try with different ingredients or combinations.")
            else:
                print("\n✅ Valid recipe generated successfully!")
            
            # Parse and print just the directions from the final recipe
            try:
                recipe_sections = parse_recipe_sections(generated_recipe)
                print("\n=== JUST THE DIRECTIONS ===")
                if recipe_sections["directions"]:
                    print(recipe_sections["directions"])
                else:
                    print("No directions found in the recipe. Here's the full recipe:")
                    print(generated_recipe)
            except Exception as e:
                print(f"\n⚠️ Error parsing the recipe: {e}")
                print("Here's the full recipe:")
                print(generated_recipe)
                
            # After generating a recipe, remind the user they can ask for another one
            if is_recipe_valid(validity_result):
                print("\n💡 Tip: Type 'more' to generate another recipe with the same ingredients.")
        else:
            print("❌ No recipes retrieved. Try different ingredients.")

if __name__ == "__main__":
    main()



