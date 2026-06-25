"""
Hand-labeled visibility of every ingredient in the Nutrition5k test split.

VISIBLE   = can realistically be identified from an overhead photo
INVISIBLE = physically impossible to see (liquids, dissolved seasonings, spices)
AMBIGUOUS = sometimes visible depending on preparation (e.g. onions — rings=yes, diced/cooked=no)
"""

# Each entry: ingredient_name -> ('visible'|'invisible'|'ambiguous', reason)
LABELS: dict[str, tuple[str, str]] = {
    # ── Oils / fats ─────────────────────────────────────────────────────────
    "olive oil":        ("invisible",  "liquid, absorbs into food"),
    "butter":           ("invisible",  "melted and absorbed"),

    # ── Salts / minerals ────────────────────────────────────────────────────
    "salt":             ("invisible",  "dissolved, no visual presence"),
    "sugar":            ("invisible",  "dissolved or invisible granules"),

    # ── Acids / liquids ─────────────────────────────────────────────────────
    "vinegar":          ("invisible",  "clear liquid, no visual presence"),
    "lemon juice":      ("invisible",  "clear liquid, squeezed on"),
    "orange juice":     ("invisible",  "absorbed or poured"),
    "soy sauce":        ("invisible",  "dark liquid, soaks in"),
    "white wine":       ("invisible",  "clear liquid, cooked off"),
    "wine":             ("invisible",  "liquid, cooked off"),

    # ── Dry spices / ground seasonings ──────────────────────────────────────
    "pepper":           ("invisible",  "ground black pepper, specks too small"),
    "mustard":          ("ambiguous",  "as condiment dollop=visible; as coating=invisible"),
    "thyme":            ("invisible",  "dried, tiny flakes invisible at overhead scale"),
    "rosemary":         ("ambiguous",  "fresh sprigs=visible; dried=invisible"),
    "basil":            ("ambiguous",  "fresh leaves=visible; dried=invisible"),
    "chive":            ("visible",    "green garnish, clearly visible"),
    "cilantro":         ("visible",    "leafy green garnish, clearly visible"),
    "parsley":          ("ambiguous",  "fresh garnish=visible; mixed in=invisible"),
    "chili":            ("invisible",  "ground spice, invisible"),
    "oregano":          ("invisible",  "dried, tiny flakes invisible"),
    "ginger":           ("invisible",  "usually grated/powdered into dish"),
    "garlic":           ("invisible",  "usually minced/cooked in"),
    "cumin":            ("invisible",  "ground spice, invisible"),
    "paprika":          ("invisible",  "ground spice, dusted on"),
    "cinnamon":         ("invisible",  "powder, invisible"),
    "flour":            ("invisible",  "coating or binder, invisible when cooked"),

    # ── Sauces / dressings ──────────────────────────────────────────────────
    "vinaigrette":      ("invisible",  "liquid dressing soaks into salad"),
    "caesar dressing":  ("invisible",  "liquid coating"),
    "caesar salad":     ("visible",    "the full prepared salad is visible"),
    "ketchup":          ("visible",    "red dollop clearly visible"),
    "mayonnaise":       ("ambiguous",  "as spread=visible; mixed in=invisible"),
    "pesto":            ("ambiguous",  "green sauce coating=visible; mixed in=invisible"),
    "salsa":            ("visible",    "chunky topping clearly visible"),
    "cream":            ("ambiguous",  "poured over=visible; mixed in=invisible"),
    "sour cream":       ("visible",    "white dollop clearly visible"),
    "tuna salad":       ("visible",    "prepared mixture clearly visible"),
    "garden salad":     ("visible",    "full salad visible"),
    "tomato sauce":     ("visible",    "red sauce visible on dish"),

    # ── Onion family ────────────────────────────────────────────────────────
    "onions":           ("ambiguous",  "raw rings/chunks=visible; diced/cooked=invisible"),
    "green onions":     ("visible",    "green garnish strips, clearly visible"),
    "shallots":         ("ambiguous",  "whole=visible; sliced/cooked=invisible"),

    # ── Greens / leaves ─────────────────────────────────────────────────────
    "arugula":          ("visible",    "leafy greens visible on plate"),
    "spinach (raw)":    ("visible",    "dark green leaves visible"),
    "spinach (cooked)": ("visible",    "wilted greens visible"),
    "chard":            ("visible",    "large green/red leaves visible"),
    "mixed greens":     ("visible",    "leafy salad base visible"),
    "kale":             ("visible",    "dark curly leaves visible"),
    "lettuce":          ("visible",    "pale green leaves visible"),
    "bok choy":         ("visible",    "white stems + green leaves visible"),
    "mustard greens":   ("visible",    "leafy greens visible"),
    "tatsoi":           ("visible",    "small dark leaves visible"),
    "cabbage":          ("visible",    "pale shredded/chopped visible"),

    # ── Proteins ────────────────────────────────────────────────────────────
    "chicken":          ("visible",    "cooked piece clearly visible"),
    "chicken breast":   ("visible",    "large piece clearly visible"),
    "grilled chicken":  ("visible",    "charred/browned piece clearly visible"),
    "chicken thighs":   ("visible",    "skin-on piece clearly visible"),
    "chicken apple sausage": ("visible", "sausage slices clearly visible"),
    "pork":             ("visible",    "cooked meat clearly visible"),
    "beef":             ("visible",    "cooked meat clearly visible"),
    "steak":            ("visible",    "large piece clearly visible"),
    "ground turkey":    ("visible",    "browned crumbles visible"),
    "fish":             ("visible",    "fillet/piece clearly visible"),
    "salmon":           ("visible",    "orange/pink fillet clearly visible"),
    "bacon":            ("visible",    "strips clearly visible"),
    "turkey bacon":     ("visible",    "strips clearly visible"),
    "sausage":          ("visible",    "links/slices clearly visible"),
    "tofu":             ("visible",    "white cubes/block visible"),
    "eggs":             ("visible",    "fried/boiled form visible"),
    "egg whites":       ("ambiguous",  "fried/baked=visible; mixed in batter=invisible"),
    "scrambled eggs":   ("visible",    "yellow fluffy mass visible"),

    # ── Dairy ───────────────────────────────────────────────────────────────
    "cheese":           ("visible",    "melted/shredded/sliced visible"),
    "parmesan cheese":  ("visible",    "grated white flakes visible"),
    "feta cheese":      ("visible",    "white crumbles clearly visible"),
    "goat cheese":      ("visible",    "white dollop/crumbles visible"),
    "cream cheese":     ("visible",    "white spread visible"),
    "greek yogurt":     ("visible",    "white dollop clearly visible"),
    "milk":             ("invisible",  "liquid, invisible in dish"),
    "cottage cheese":   ("visible",    "white curds visible"),
    "frozen yogurt":    ("visible",    "ice cream-like serving visible"),

    # ── Grains / carbs ──────────────────────────────────────────────────────
    "white rice":       ("visible",    "white granules clearly visible"),
    "brown rice":       ("visible",    "brown granules clearly visible"),
    "wheat berry":      ("visible",    "whole grain kernels visible"),
    "millet":           ("visible",    "small yellow granules visible"),
    "quinoa":           ("visible",    "small cooked granules visible"),
    "country rice":     ("visible",    "rice visible"),
    "wild rice":        ("visible",    "dark long-grain visible"),
    "bulgur":           ("visible",    "coarse wheat granules visible"),
    "fried rice":       ("visible",    "rice + mix-ins clearly visible"),
    "oatmeal":          ("visible",    "porridge visible"),
    "couscous":         ("visible",    "fine grain visible"),
    "pilaf":            ("visible",    "rice/grain dish visible"),
    "hominy":           ("visible",    "large corn kernels visible"),
    "lentils":          ("visible",    "small legume visible"),
    "chickpeas":        ("visible",    "round beige legumes visible"),
    "black beans":      ("visible",    "dark oval legumes visible"),

    # ── Bread / baked ───────────────────────────────────────────────────────
    "bread":            ("visible",    "slice/roll clearly visible"),
    "bagels":           ("visible",    "ring shape clearly visible"),
    "pizza":            ("visible",    "flat round base clearly visible"),
    "cheese pizza":     ("visible",    "topped pizza clearly visible"),
    "pepperoni pizza":  ("visible",    "red toppings on pizza visible"),
    "croutons":         ("visible",    "golden cubes visible on salad"),
    "tortilla chips":   ("visible",    "triangular chips visible"),
    "cookies":          ("visible",    "round baked goods visible"),
    "brownies":         ("visible",    "dark square pieces visible"),

    # ── Vegetables ──────────────────────────────────────────────────────────
    "carrot":           ("visible",    "orange sticks/rounds visible"),
    "cherry tomatoes":  ("visible",    "small red spheres clearly visible"),
    "bell peppers":     ("visible",    "colorful chunks clearly visible"),
    "broccoli":         ("visible",    "green florets clearly visible"),
    "cucumbers":        ("visible",    "green slices visible"),
    "cauliflower":      ("visible",    "white florets clearly visible"),
    "jalapenos":        ("visible",    "green slices visible"),
    "green beans":      ("visible",    "long green pods visible"),
    "mushroom":         ("visible",    "brown caps visible"),
    "zucchini":         ("visible",    "green slices visible"),
    "tomatoes":         ("visible",    "red chunks/slices visible"),
    "olives":           ("visible",    "dark/green oval pieces visible"),
    "roasted potatoes": ("visible",    "golden chunks clearly visible"),
    "potatoes":         ("visible",    "chunks/slices visible"),
    "hash browns":      ("visible",    "shredded patty visible"),
    "sweet potato":     ("visible",    "orange flesh visible"),
    "yam":              ("visible",    "orange flesh visible"),
    "squash":           ("visible",    "yellow/orange chunks visible"),
    "eggplant":         ("visible",    "purple/dark chunks visible"),
    "corn on the cob":  ("visible",    "yellow cob clearly visible"),
    "corn":             ("visible",    "yellow kernels visible"),
    "celery":           ("visible",    "green stalks visible"),
    "celery root":      ("visible",    "pale bulb chunk visible"),
    "radishes":         ("visible",    "red/pink slices visible"),
    "pumpkin seeds":    ("visible",    "flat green seeds visible on dish"),
    "sun dried tomatoes":("visible",   "dark red chewy pieces visible"),
    "pickles":          ("visible",    "green slices visible"),
    "avocado":          ("visible",    "green flesh/slices visible"),
    "artichokes":       ("visible",    "layered vegetable visible"),
    "snow peas":        ("visible",    "flat green pods visible"),
    "chayote squash":   ("visible",    "pale green chunks visible"),
    "asparagus":        ("visible",    "green spears visible"),
    "beets":            ("visible",    "dark red chunks visible"),
    "brussels sprouts": ("visible",    "small green spheres visible"),

    # ── Fruit ───────────────────────────────────────────────────────────────
    "blueberries":      ("visible",    "small blue spheres visible"),
    "raspberries":      ("visible",    "red drupelets visible"),
    "blackberries":     ("visible",    "dark purple drupelets visible"),
    "strawberries":     ("visible",    "red heart-shaped fruit visible"),
    "berries":          ("visible",    "small colorful fruit visible"),
    "pineapple":        ("visible",    "yellow chunks clearly visible"),
    "apple":            ("visible",    "red/green slices visible"),
    "cantaloupe":       ("visible",    "orange flesh visible"),
    "honeydew melons":  ("visible",    "pale green flesh visible"),
    "watermelon":       ("visible",    "red flesh + black seeds visible"),
    "grapes":           ("visible",    "round purple/green clusters visible"),
    "pears":            ("visible",    "green/yellow fruit visible"),
    "lime":             ("visible",    "green wedge visible"),
    "orange":           ("visible",    "orange segments visible"),
    "mandarin oranges": ("visible",    "small orange segments visible"),
    "cranberries":      ("visible",    "small red berries visible"),
    "figs":             ("visible",    "purple/brown fruit visible"),

    # ── Nuts / seeds ────────────────────────────────────────────────────────
    "almonds":          ("visible",    "whole/sliced nuts visible"),
    "walnuts":          ("visible",    "brown nut pieces visible"),
    "pecans":           ("visible",    "brown nut halves visible"),
    "chia seeds":       ("visible",    "tiny dark specks visible on surface"),
    "granola":          ("visible",    "golden clusters visible"),

    # ── Other / prepared ────────────────────────────────────────────────────
    "lemon":            ("visible",    "yellow wedge clearly visible"),
    "pasta":            ("visible",    "noodles/shapes clearly visible"),
    "wine":             ("invisible",  "liquid in glass or cooked off"),
    "deprecated":       ("invisible",  "dataset artifact, ignore"),
    "plate only":       ("visible",    "empty plate is visible"),
}


def visibility(ingredient: str) -> str:
    """Return 'visible', 'invisible', or 'ambiguous' for an ingredient."""
    return LABELS.get(ingredient.lower().strip(), ("ambiguous", "unlabeled"))[0]


def reason(ingredient: str) -> str:
    return LABELS.get(ingredient.lower().strip(), ("ambiguous", "unlabeled"))[1]


if __name__ == "__main__":
    import json, pathlib, sys
    sys.path.insert(0, ".")
    d = json.loads(pathlib.Path("results/claude_opus_overhead_full.json").read_text())
    import collections
    all_gt = collections.Counter()
    for s in d["samples"]:
        for g in s["ground_truth"]:
            all_gt[g] += 1

    unlabeled = [ing for ing in all_gt if ing not in LABELS]
    print(f"Labeled: {len(LABELS)}  |  Unlabeled in dataset: {len(unlabeled)}")
    if unlabeled:
        print("UNLABELED:", unlabeled)

    vis = sum(cnt for ing, cnt in all_gt.items() if visibility(ing) == "visible")
    inv = sum(cnt for ing, cnt in all_gt.items() if visibility(ing) == "invisible")
    amb = sum(cnt for ing, cnt in all_gt.items() if visibility(ing) == "ambiguous")
    tot = sum(all_gt.values())
    print(f"\nDataset breakdown ({tot} ingredient instances):")
    print(f"  Visible:   {vis:4d} ({100*vis/tot:.0f}%)")
    print(f"  Invisible: {inv:4d} ({100*inv/tot:.0f}%)")
    print(f"  Ambiguous: {amb:4d} ({100*amb/tot:.0f}%)")
