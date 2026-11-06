"""
NutriVision AI - Utility Functions
Handles food data loading, nutrition calculations, and portion scaling
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List


class FoodDataManager:
    """Manages food nutrition data and calculations"""
    
    def __init__(self, data_path='data/food_data.json'):
        """
        Initialize the food data manager
        
        Args:
            data_path: Path to food_data.json
        """
        self.data_path = data_path
        self.food_data = self._load_food_data()
    
    def _load_food_data(self) -> Dict[str, Any]:
        """Load food data from JSON file"""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Food data not found at {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def get_food_info(self, food_name: str) -> Optional[Dict[str, Any]]:
        """
        Get complete food information
        
        Args:
            food_name: Name of the food item
            
        Returns:
            Dictionary with food data or None if not found
        """
        # Normalize food name (lowercase, replace spaces with underscores)
        normalized_name = food_name.lower().replace(' ', '_')
        
        if normalized_name in self.food_data:
            return self.food_data[normalized_name]
        else:
            return None
    
    def get_nutrition_data(self, food_name: str, portion_multiplier: float = 1.0) -> Dict[str, Any]:
        """
        Get nutrition data with portion scaling
        
        Args:
            food_name: Name of the food item
            portion_multiplier: Multiplier for portion size (e.g., 1.5 for 1.5x serving)
            
        Returns:
            Dictionary with scaled nutrition data
        """
        food_info = self.get_food_info(food_name)
        
        if food_info is None:
            return {
                'status': 'not_found',
                'message': 'Data not available',
                'food_name': food_name
            }
        
        # Scale calories
        base_calories = food_info.get('calories', 0)
        scaled_calories = int(base_calories * portion_multiplier)
        
        # Scale macros (extract numeric values)
        base_macros = food_info.get('macros', {})
        scaled_macros = {}
        for macro, value in base_macros.items():
            # Extract number from string like "17g"
            numeric_value = float(value.replace('g', '').strip())
            scaled_value = numeric_value * portion_multiplier
            scaled_macros[macro] = f"{scaled_value:.1f}g"
        
        # Calculate burn-off times based on scaled calories
        burn_off = self.calculate_burn_off(scaled_calories)
        
        # Get other data (not scaled)
        serving_size = food_info.get('serving_size', 'Data not available')
        sustainability = food_info.get('sustainability', 'Data not available')
        eco_badge_color = food_info.get('eco_badge_color', 'gray')
        description = food_info.get('description', 'Data not available')
        leftover_ideas = food_info.get('leftover_ideas', 'Data not available')
        flavor_profile = food_info.get('flavor_profile', 'Data not available')
        pairing = food_info.get('pairing', 'Data not available')
        
        return {
            'status': 'success',
            'food_name': food_name,
            'calories': scaled_calories,
            'base_calories': base_calories,
            'serving_size': serving_size,
            'portion_multiplier': portion_multiplier,
            'macros': scaled_macros,
            'burn_off': burn_off,
            'sustainability': sustainability,
            'eco_badge_color': eco_badge_color,
            'eco_score': self.get_eco_score(sustainability),
            'description': description,
            'leftover_ideas': leftover_ideas,
            'flavor_profile': flavor_profile,
            'pairing': pairing,
            'dietary_tags': self.get_dietary_filters(food_name)
        }
    
    def calculate_burn_off(self, calories: int) -> Dict[str, int]:
        """
        Calculate burn-off times for running and yoga
        
        Formula:
        - Running: calories / 11.4 (kcal/min)
        - Yoga: calories / 13.9 (kcal/round, ~1 Surya Namaskar)
        
        Args:
            calories: Number of calories
            
        Returns:
            Dictionary with run_mins and yoga_rounds
        """
        run_mins = int(round(calories / 11.4))
        yoga_rounds = int(round(calories / 13.9))
        
        return {
            'run_mins': run_mins,
            'yoga_rounds': yoga_rounds
        }
    
    def get_eco_score(self, sustainability: str) -> int:
        """
        Map sustainability level to numeric score

        Args:
            sustainability: Sustainability level (Low/Medium/High)

        Returns:
            Numeric score (1-10)
        """
        mapping = {
            'Low': 9,      # Low impact = high score (eco-friendly)
            'Medium': 5,   # Medium impact = medium score
            'High': 2      # High impact = low score (not eco-friendly)
        }
        return mapping.get(sustainability, 5)  # Default to 5 if unknown

    def get_dietary_filters(self, food_name: str) -> List[str]:
        """
        Get dietary classification tags for a food item

        Args:
            food_name: Name of the food item

        Returns:
            List of dietary tags (e.g., ['high_protein', 'low_fat'])
        """
        food_info = self.get_food_info(food_name)

        if food_info is None:
            return []

        tags = []

        # Check macros for classification
        macros = food_info.get('macros', {})

        # Extract numeric values
        protein = float(macros.get('protein', '0g').replace('g', '').strip())
        carbs = float(macros.get('carbs', '0g').replace('g', '').strip())
        fat = float(macros.get('fat', '0g').replace('g', '').strip())

        # High protein (>10g per serving)
        if protein > 10:
            tags.append('high_protein')

        # Low calorie (<150 kcal)
        calories = food_info.get('calories', 0)
        if calories < 150:
            tags.append('low_calorie')

        # High calorie (>350 kcal)
        if calories > 350:
            tags.append('high_calorie')

        # Low fat (<5g)
        if fat < 5:
            tags.append('low_fat')

        # Eco-friendly (Low sustainability impact)
        sustainability = food_info.get('sustainability', '')
        if sustainability == 'Low':
            tags.append('eco_friendly')

        return tags

    def get_all_food_names(self) -> List[str]:
        """
        Get list of all available food names

        Returns:
            List of food names
        """
        return list(self.food_data.keys())

    def search_foods_by_criteria(self, max_calories: Optional[int] = None,
                                  min_protein: Optional[float] = None,
                                  sustainability: Optional[str] = None) -> List[str]:
        """
        Search foods by nutritional criteria

        Args:
            max_calories: Maximum calories
            min_protein: Minimum protein in grams
            sustainability: Sustainability level (Low/Medium/High)

        Returns:
            List of matching food names
        """
        matching_foods = []

        for food_name, food_info in self.food_data.items():
            # Check calories
            if max_calories is not None:
                if food_info.get('calories', 0) > max_calories:
                    continue

            # Check protein
            if min_protein is not None:
                protein_str = food_info.get('macros', {}).get('protein', '0g')
                protein = float(protein_str.replace('g', '').strip())
                if protein < min_protein:
                    continue

            # Check sustainability
            if sustainability is not None:
                if food_info.get('sustainability', '') != sustainability:
                    continue

            matching_foods.append(food_name)

        return matching_foods


def load_food_data_manager(data_path='data/food_data.json') -> FoodDataManager:
    """
    Convenience function to create FoodDataManager instance

    Args:
        data_path: Path to food_data.json

    Returns:
        FoodDataManager instance
    """
    return FoodDataManager(data_path=data_path)


def format_calories(calories: int) -> str:
    """Format calories with comma separator"""
    return f"{calories:,} kcal"


def format_burn_off_text(burn_off: Dict[str, int]) -> Dict[str, str]:
    """
    Format burn-off data into readable text

    Args:
        burn_off: Dictionary with run_mins and yoga_rounds

    Returns:
        Dictionary with formatted text
    """
    return {
        'running': f"{burn_off['run_mins']} minutes of running",
        'yoga': f"{burn_off['yoga_rounds']} rounds of Surya Namaskar (yoga)"
    }


if __name__ == "__main__":
    # Test the food data manager
    manager = load_food_data_manager()

    print("="*60)
    print("FOOD DATA MANAGER TEST")
    print("="*60)

    # Test 1: Get food info
    print("\n1. Testing get_nutrition_data for 'samosa':")
    data = manager.get_nutrition_data('samosa')
    print(f"   Status: {data['status']}")
    print(f"   Calories: {data['calories']} kcal")
    print(f"   Macros: {data['macros']}")
    print(f"   Burn-off: {data['burn_off']}")
    print(f"   Eco Score: {data['eco_score']}/10")

    # Test 2: Portion scaling
    print("\n2. Testing portion scaling (1.5x) for 'idli':")
    data = manager.get_nutrition_data('idli', portion_multiplier=1.5)
    print(f"   Base Calories: {data['base_calories']} kcal")
    print(f"   Scaled Calories: {data['calories']} kcal")
    print(f"   Scaled Macros: {data['macros']}")

    # Test 3: Dietary filters
    print("\n3. Testing dietary filters for 'kadai_paneer':")
    tags = manager.get_dietary_filters('kadai_paneer')
    print(f"   Tags: {tags}")

    # Test 4: Search by criteria
    print("\n4. Searching low-calorie foods (<150 kcal):")
    low_cal = manager.search_foods_by_criteria(max_calories=150)
    print(f"   Found: {low_cal}")

    # Test 5: Missing data
    print("\n5. Testing missing food item:")
    data = manager.get_nutrition_data('nonexistent_food')
    print(f"   Status: {data['status']}")
    print(f"   Message: {data['message']}")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

