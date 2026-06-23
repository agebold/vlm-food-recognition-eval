# Food ingredient synonym groups — based on common naming variations in Nutrition5k
# and standard culinary equivalences. Each group is a set of equivalent names;
# any two names in the same group match with SemMatch = 1.0 (PMC13092701 Eq 6).
#
# Extend this list as you discover mismatches during evaluation.

_RAW_GROUPS: list[list[str]] = [
    # Alliums
    ["scallion", "spring onion", "green onion"],
    ["onion", "yellow onion", "white onion", "brown onion"],
    ["red onion", "purple onion"],
    ["shallot", "eschalot"],

    # Leafy greens
    ["spinach", "baby spinach"],
    ["lettuce", "romaine", "romaine lettuce", "cos lettuce"],
    ["kale", "curly kale"],
    ["arugula", "rocket", "rucola"],
    ["bok choy", "pak choi", "bok choi"],
    ["cabbage", "green cabbage", "white cabbage"],

    # Peppers
    ["bell pepper", "capsicum", "sweet pepper"],
    ["red bell pepper", "red pepper", "red capsicum"],
    ["green bell pepper", "green pepper", "green capsicum"],
    ["yellow bell pepper", "yellow pepper", "yellow capsicum"],
    ["chili", "chilli", "chile", "chili pepper"],
    ["jalapeno", "jalapeño"],

    # Potatoes / starchy veg
    ["potato", "white potato", "russet potato"],
    ["sweet potato", "yam"],
    ["corn", "maize", "sweet corn", "sweetcorn"],
    ["peas", "green peas", "garden peas"],

    # Proteins — beef
    ["beef", "ground beef", "minced beef"],
    ["steak", "beef steak"],

    # Proteins — chicken
    ["chicken", "chicken breast", "grilled chicken"],
    ["chicken thigh", "chicken leg"],

    # Proteins — pork
    ["pork", "pork loin"],
    ["bacon", "streaky bacon"],
    ["ham", "cooked ham"],

    # Proteins — fish & seafood
    ["salmon", "atlantic salmon"],
    ["tuna", "canned tuna", "yellowfin tuna"],
    ["shrimp", "prawn", "prawns", "shrimps"],

    # Proteins — other
    ["egg", "eggs", "whole egg"],
    ["tofu", "bean curd", "firm tofu"],
    ["tempeh", "fermented soybean"],

    # Legumes
    ["chickpeas", "garbanzo beans", "garbanzo"],
    ["black beans", "black bean"],
    ["kidney beans", "red kidney beans"],
    ["lentils", "lentil", "red lentils"],
    ["edamame", "edamame beans", "soybeans"],

    # Grains
    ["rice", "white rice", "steamed rice", "cooked rice"],
    ["brown rice", "whole grain rice"],
    ["pasta", "spaghetti", "penne", "fettuccine", "linguine"],
    ["bread", "white bread", "sandwich bread"],
    ["noodles", "egg noodles", "wheat noodles"],
    ["quinoa", "cooked quinoa"],
    ["oats", "oatmeal", "rolled oats"],

    # Dairy
    ["cheese", "cheddar", "cheddar cheese"],
    ["mozzarella", "mozzarella cheese"],
    ["parmesan", "parmigiano", "parmigiano reggiano", "parmesan cheese"],
    ["feta", "feta cheese"],
    ["cream cheese", "soft cheese"],
    ["sour cream", "creme fraiche"],
    ["butter", "unsalted butter"],
    ["milk", "whole milk", "cow milk"],
    ["yogurt", "yoghurt", "greek yogurt", "plain yogurt"],

    # Sauces & condiments
    ["ketchup", "tomato ketchup", "catsup"],
    ["mayo", "mayonnaise"],
    ["mustard", "yellow mustard", "dijon mustard"],
    ["soy sauce", "shoyu", "tamari"],
    ["hot sauce", "chili sauce", "sriracha"],
    ["ranch dressing", "ranch", "ranch sauce"],
    ["balsamic", "balsamic vinegar", "balsamic dressing"],
    ["olive oil", "extra virgin olive oil", "evoo"],

    # Tomatoes
    ["tomato", "tomatoes", "fresh tomato"],
    ["cherry tomato", "cherry tomatoes", "grape tomatoes"],
    ["sun-dried tomato", "sundried tomato"],

    # Cucumbers / squash
    ["cucumber", "english cucumber"],
    ["zucchini", "courgette"],
    ["eggplant", "aubergine", "brinjal"],

    # Mushrooms
    ["mushroom", "mushrooms", "white mushroom", "button mushroom"],
    ["portobello", "portobello mushroom"],
    ["shiitake", "shiitake mushroom"],

    # Fruits used as food
    ["avocado", "hass avocado"],
    ["lemon", "lemon juice"],
    ["lime", "lime juice"],

    # Nuts & seeds
    ["almonds", "almond", "sliced almonds"],
    ["peanuts", "peanut"],
    ["sesame seeds", "sesame", "sesame seed"],
    ["sunflower seeds", "sunflower seed"],
    ["walnuts", "walnut"],

    # Herbs
    ["cilantro", "coriander", "fresh coriander"],
    ["parsley", "fresh parsley"],
    ["basil", "fresh basil"],
    ["mint", "fresh mint"],

    # Spices (less common to confuse but included)
    ["garlic", "minced garlic", "garlic clove"],
    ["ginger", "fresh ginger", "ginger root"],
    ["turmeric", "turmeric powder"],
    ["cumin", "cumin seeds", "ground cumin"],
]

# Build bidirectional lookup: name -> set of all equivalent names
SYNONYMS: dict[str, set[str]] = {}

for group in _RAW_GROUPS:
    normalized = [g.lower().strip() for g in group]
    for name in normalized:
        SYNONYMS.setdefault(name, set()).update(normalized)
        SYNONYMS[name].discard(name)  # don't include self


def get_variants(name: str) -> set[str]:
    return SYNONYMS.get(name.lower().strip(), set())
