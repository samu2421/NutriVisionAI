"""
NutriVision AI - GenAI Integration Module
Handles AI-powered explanations using Groq/OpenAI APIs
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path


class GenAIHelper:
    """Handles GenAI API calls for food explanations and suggestions"""
    
    def __init__(self, api_provider='groq', api_key=None):
        """
        Initialize GenAI helper
        
        Args:
            api_provider: 'groq' or 'openai'
            api_key: API key (if None, loads from environment)
        """
        self.api_provider = api_provider.lower()
        self.api_key = api_key or self._load_api_key()
        self.client = None
        
        if self.api_key:
            self._initialize_client()
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment variables"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        if self.api_provider == 'groq':
            return os.getenv('GROQ_API_KEY')
        elif self.api_provider == 'openai':
            return os.getenv('OPENAI_API_KEY')
        else:
            return None
    
    def _initialize_client(self):
        """Initialize the API client"""
        try:
            if self.api_provider == 'groq':
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                self.model = 'llama-3.1-70b-versatile'
            elif self.api_provider == 'openai':
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.model = 'gpt-3.5-turbo'
        except ImportError as e:
            print(f"Warning: Could not import {self.api_provider} library: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if GenAI is available"""
        return self.client is not None and self.api_key is not None
    
    def generate_cultural_explanation(self, food_name: str, base_description: str) -> str:
        """
        Generate cultural/contextual explanation (≤50 words)
        
        Args:
            food_name: Name of the food
            base_description: Base description from JSON
            
        Returns:
            Cultural explanation string
        """
        if not self.is_available():
            return base_description
        
        system_prompt = """You are a food culture expert specializing in Indian cuisine and Ayurveda.
Provide concise, culturally rich explanations about Indian foods.
Keep responses under 50 words. Be factual and educational."""
        
        user_prompt = f"""Explain the cultural significance of {food_name} in Indian cuisine.
Base context: {base_description}

Provide a brief explanation (max 50 words) covering cultural importance or Ayurvedic perspective."""
        
        try:
            response = self._call_api(system_prompt, user_prompt, max_tokens=100)
            return response.strip()
        except Exception as e:
            print(f"GenAI error: {e}")
            return base_description
    
    def generate_sustainability_explanation(self, food_name: str, 
                                           sustainability_level: str,
                                           eco_score: int) -> str:
        """
        Generate sustainability explanation
        
        Args:
            food_name: Name of the food
            sustainability_level: Low/Medium/High
            eco_score: Numeric eco score (1-10)
            
        Returns:
            Sustainability explanation string
        """
        if not self.is_available():
            return f"This food has {sustainability_level.lower()} environmental impact (Eco Score: {eco_score}/10)."
        
        system_prompt = """You are an environmental sustainability expert focusing on food systems.
Explain the environmental impact of foods clearly and concisely.
Keep responses under 60 words. Be factual and educational."""
        
        user_prompt = f"""Explain why {food_name} has a {sustainability_level.lower()} environmental impact.
Eco Score: {eco_score}/10

Provide a brief explanation (max 60 words) about its carbon footprint, water usage, or production impact."""
        
        try:
            response = self._call_api(system_prompt, user_prompt, max_tokens=120)
            return response.strip()
        except Exception as e:
            print(f"GenAI error: {e}")
            return f"This food has {sustainability_level.lower()} environmental impact (Eco Score: {eco_score}/10)."
    
    def generate_leftover_ideas(self, food_name: str, base_ideas: str) -> list:
        """
        Generate creative leftover remix ideas (2 ideas)
        
        Args:
            food_name: Name of the food
            base_ideas: Base leftover ideas from JSON
            
        Returns:
            List of 2 leftover ideas
        """
        if not self.is_available():
            # Parse base ideas or return default
            if base_ideas and base_ideas != "Data not available":
                return [base_ideas, "Store in airtight container and reheat within 24 hours."]
            else:
                return [
                    "Store in airtight container and refrigerate.",
                    "Reheat thoroughly before consuming."
                ]
        
        system_prompt = """You are a creative chef specializing in Indian cuisine and food waste reduction.
Provide practical, creative ideas for using leftover Indian foods.
Give exactly 2 ideas. Keep each idea under 20 words."""

        user_prompt = f"""Give me 2 creative ways to use leftover {food_name}.
Base idea: {base_ideas}

Provide exactly 2 practical ideas, one per line. Each idea should be under 20 words."""

        try:
            response = self._call_api(system_prompt, user_prompt, max_tokens=100)
            # Parse response into list
            ideas = [line.strip() for line in response.strip().split('\n') if line.strip()]
            # Remove numbering if present (e.g., "1. ", "2. ")
            ideas = [idea.lstrip('0123456789.-) ') for idea in ideas]
            # Return first 2 ideas
            if len(ideas) >= 2:
                return ideas[:2]
            elif len(ideas) == 1:
                return [ideas[0], "Store in airtight container and reheat within 24 hours."]
            else:
                return [base_ideas, "Store in airtight container and reheat within 24 hours."]
        except Exception as e:
            print(f"GenAI error: {e}")
            if base_ideas and base_ideas != "Data not available":
                return [base_ideas, "Store in airtight container and reheat within 24 hours."]
            else:
                return [
                    "Store in airtight container and refrigerate.",
                    "Reheat thoroughly before consuming."
                ]

    def generate_ai_insights(self, food_name: str, nutrition_data: Dict[str, Any]) -> str:
        """
        Generate structured AI insights grounded in food_data.json

        Args:
            food_name: Name of the food
            nutrition_data: Nutrition data from FoodDataManager

        Returns:
            Structured insights (max 60 words)
        """
        if not self.client:
            # Fallback to structured static insights
            description = nutrition_data.get('description', '')
            macros = nutrition_data.get('macros', {})
            sustainability = nutrition_data.get('sustainability', 'Medium')

            protein = macros.get('protein', '0g')
            carbs = macros.get('carbs', '0g')

            return f"{food_name.replace('_', ' ').title()}: {description[:80]}... Macros: {protein} protein, {carbs} carbs. Sustainability: {sustainability} impact."

        try:
            # Structured prompt grounded in JSON data
            description = nutrition_data.get('description', 'Traditional Indian food')
            macros = nutrition_data.get('macros', {})
            sustainability = nutrition_data.get('sustainability', 'Medium')
            flavor_profile = nutrition_data.get('flavor_profile', 'Savory')

            prompt = f"""Provide 3 bullet points about {food_name.replace('_', ' ')} based ONLY on this data:
- Description: {description}
- Macros: {macros}
- Sustainability: {sustainability}
- Flavor: {flavor_profile}

Format:
• [Health aspect]
• [Cultural aspect]
• [Sustainability aspect]

Max 60 words total. Be specific and factual."""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a nutrition expert. Provide concise, factual insights based only on given data. No hallucination."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )

            insights = response.choices[0].message.content.strip()
            return insights if insights else f"{food_name.replace('_', ' ').title()} is a traditional Indian food with unique nutritional properties."

        except Exception as e:
            print(f"Error generating AI insights: {e}")
            # Fallback to structured static insights
            description = nutrition_data.get('description', '')
            return f"{food_name.replace('_', ' ').title()}: {description[:100]}"

    def generate_complete_analysis(self, food_name: str, nutrition_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete AI-powered analysis

        Args:
            food_name: Name of the food
            nutrition_data: Nutrition data from FoodDataManager

        Returns:
            Dictionary with AI-generated content
        """
        if nutrition_data.get('status') == 'not_found':
            return {
                'cultural_explanation': 'Data not available',
                'sustainability_explanation': 'Data not available',
                'leftover_ideas': ['Data not available', 'Data not available'],
                'flavor_pairing': 'Data not available',
                'ai_insights': 'Data not available'
            }

        # Generate each component
        cultural_exp = self.generate_cultural_explanation(
            food_name,
            nutrition_data.get('description', 'Data not available')
        )

        sustainability_exp = self.generate_sustainability_explanation(
            food_name,
            nutrition_data.get('sustainability', 'Medium'),
            nutrition_data.get('eco_score', 5)
        )

        leftover_ideas = self.generate_leftover_ideas(
            food_name,
            nutrition_data.get('leftover_ideas', 'Data not available')
        )

        # Get flavor pairing from JSON (with fallback)
        flavor_pairing = nutrition_data.get('pairing', 'Pairs well with traditional accompaniments')

        # Generate structured AI insights
        ai_insights = self.generate_ai_insights(food_name, nutrition_data)

        return {
            'cultural_explanation': cultural_exp,
            'sustainability_explanation': sustainability_exp,
            'leftover_ideas': leftover_ideas,
            'flavor_pairing': flavor_pairing,
            'ai_insights': ai_insights
        }

    def _call_api(self, system_prompt: str, user_prompt: str, max_tokens: int = 150) -> str:
        """
        Call the GenAI API

        Args:
            system_prompt: System instruction
            user_prompt: User query
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        if not self.is_available():
            raise Exception("GenAI client not available")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if self.api_provider == 'groq':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
        elif self.api_provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
        else:
            raise Exception(f"Unsupported API provider: {self.api_provider}")

        return response.choices[0].message.content


def load_genai_helper(api_provider='groq', api_key=None) -> GenAIHelper:
    """
    Convenience function to create GenAIHelper instance

    Args:
        api_provider: 'groq' or 'openai'
        api_key: API key (optional)

    Returns:
        GenAIHelper instance
    """
    return GenAIHelper(api_provider=api_provider, api_key=api_key)


if __name__ == "__main__":
    # Test the GenAI helper
    print("="*60)
    print("GENAI HELPER TEST")
    print("="*60)

    # Initialize helper
    helper = load_genai_helper(api_provider='groq')

    print(f"\nAPI Provider: {helper.api_provider}")
    print(f"API Available: {helper.is_available()}")

    if helper.is_available():
        print("\n" + "-"*60)
        print("Testing with 'samosa'")
        print("-"*60)

        # Test cultural explanation
        print("\n1. Cultural Explanation:")
        cultural = helper.generate_cultural_explanation(
            'samosa',
            'Triangular fried pastry with potato filling.'
        )
        print(f"   {cultural}")

        # Test sustainability explanation
        print("\n2. Sustainability Explanation:")
        sustainability = helper.generate_sustainability_explanation(
            'samosa',
            'Low',
            9
        )
        print(f"   {sustainability}")

        # Test leftover ideas
        print("\n3. Leftover Ideas:")
        ideas = helper.generate_leftover_ideas(
            'samosa',
            'Smash and top with chickpea curry to make Samosa Chaat.'
        )
        for i, idea in enumerate(ideas, 1):
            print(f"   {i}. {idea}")

        # Test complete analysis
        print("\n" + "-"*60)
        print("Testing complete analysis")
        print("-"*60)

        nutrition_data = {
            'status': 'success',
            'description': 'Triangular fried pastry with potato filling.',
            'sustainability': 'Low',
            'eco_score': 9,
            'leftover_ideas': 'Smash and top with chickpea curry to make Samosa Chaat.'
        }

        analysis = helper.generate_complete_analysis('samosa', nutrition_data)
        print(f"\nCultural: {analysis['cultural_explanation']}")
        print(f"Sustainability: {analysis['sustainability_explanation']}")
        print(f"Leftover Ideas: {analysis['leftover_ideas']}")
    else:
        print("\n⚠️  GenAI not available (API key not found)")
        print("Set GROQ_API_KEY or OPENAI_API_KEY in .env file")

        # Test fallback behavior
        print("\n" + "-"*60)
        print("Testing fallback behavior (no API)")
        print("-"*60)

        cultural = helper.generate_cultural_explanation(
            'samosa',
            'Triangular fried pastry with potato filling.'
        )
        print(f"\nCultural (fallback): {cultural}")

        ideas = helper.generate_leftover_ideas(
            'samosa',
            'Smash and top with chickpea curry.'
        )
        print(f"Leftover Ideas (fallback): {ideas}")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

