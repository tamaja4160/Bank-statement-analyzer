"""
Comprehensive German banking transaction data for statement generation.
Contains realistic recurring and one-off payment patterns for ML training.
"""

# Recurring Payments - Essential services that Check24 would compare
RECURRING_PAYMENTS = [
    # Insurance - Multiple providers per category (essential services)
    {
        "partner": "Allianz SE", "category": "Liability Insurance",
        "description_template": "HAFTPFLICHT {partner} K-{contract_id}",
        "base_amount": 55.20, "variation": (-2.5, 2.5)
    },
    {
        "partner": "HUK-Coburg", "category": "Car Insurance",
        "description_template": "Kfz-VERSICHERUNG {partner} {contract_id}",
        "base_amount": 89.50, "variation": (-5.0, 5.0)
    },
    {
        "partner": "Generali", "category": "Home Insurance",
        "description_template": "WOHNGEBAEUDEVERSICHERUNG {partner} {contract_id}",
        "base_amount": 45.80, "variation": (-3.0, 3.0)
    },
    {
        "partner": "AXA", "category": "Legal Protection",
        "description_template": "RECHTSSCHUTZ {partner} {contract_id}",
        "base_amount": 29.90, "variation": (-2.0, 2.0)
    },
    {
        "partner": "DKV Deutsche Krankenversicherung", "category": "Health Insurance",
        "description_template": "GESUNDHEITSVERSICHERUNG {partner} {contract_id}",
        "base_amount": 89.50, "variation": (-5.0, 5.0)
    },

    # Telecom & Internet - Multiple providers (essential services)
    {
        "partner": "Deutsche Telekom AG", "category": "Internet",
        "description_template": "TELEKOM RECHNUNG {contract_id}",
        "base_amount": 49.95, "variation": (0.0, 0.0)
    },
    {
        "partner": "Vodafone", "category": "Internet",
        "description_template": "VODAFONE INTERNET {contract_id}",
        "base_amount": 39.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "O2 Germany", "category": "Mobile",
        "description_template": "O2 MOBILFUNK {contract_id}",
        "base_amount": 29.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "1&1", "category": "Mobile",
        "description_template": "1&1 MOBILE {contract_id}",
        "base_amount": 24.99, "variation": (0.0, 0.0)
    },

    # Utilities - Multiple providers per category (essential services)
    {
        "partner": "E.ON Energie", "category": "Electricity",
        "description_template": "E.ON STROM {contract_id}",
        "base_amount": 78.40, "variation": (-4.0, 4.0)
    },
    {
        "partner": "RWE", "category": "Electricity",
        "description_template": "RWE STROMABSCHLAG {contract_id}",
        "base_amount": 72.50, "variation": (-3.5, 3.5)
    },
    {
        "partner": "EnBW", "category": "Electricity",
        "description_template": "ENBW ENERGIE {contract_id}",
        "base_amount": 69.90, "variation": (-3.0, 3.0)
    },
    {
        "partner": "Stadtwerke Berlin", "category": "Gas",
        "description_template": "GASABSCHLAG {partner} {contract_id}",
        "base_amount": 65.80, "variation": (-3.0, 3.0)
    },
    {
        "partner": "Stadtwerke München", "category": "Gas",
        "description_template": "GASRECHNUNG {partner} {contract_id}",
        "base_amount": 71.20, "variation": (-3.5, 3.5)
    },
    {
        "partner": "Stadtwerke Hamburg", "category": "Water",
        "description_template": "WASSERABSCHLAG {partner} {contract_id}",
        "base_amount": 25.90, "variation": (-2.0, 2.0)
    },
    {
        "partner": "Stadtwerke Köln", "category": "Water",
        "description_template": "WASSERRECHNUNG {partner} {contract_id}",
        "base_amount": 28.40, "variation": (-2.5, 2.5)
    },

    # Streaming & Entertainment (multiple allowed)
    {
        "partner": "Netflix", "category": "Entertainment",
        "description_template": "NETFLIX ABONNEMENT {contract_id}",
        "base_amount": 15.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "Amazon Prime", "category": "Entertainment",
        "description_template": "AMAZON PRIME {contract_id}",
        "base_amount": 8.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "Disney+", "category": "Entertainment",
        "description_template": "DISNEY PLUS {contract_id}",
        "base_amount": 8.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "Spotify", "category": "Music",
        "description_template": "SPOTIFY PREMIUM {contract_id}",
        "base_amount": 9.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "DAZN", "category": "Sports",
        "description_template": "DAZN SPORT {contract_id}",
        "base_amount": 14.99, "variation": (0.0, 0.0)
    },

    # Software & Cloud Services (multiple allowed)
    {
        "partner": "Microsoft", "category": "Software",
        "description_template": "MICROSOFT 365 {contract_id}",
        "base_amount": 69.00, "variation": (0.0, 0.0)
    },
    {
        "partner": "Adobe", "category": "Software",
        "description_template": "ADOBE CREATIVE CLOUD {contract_id}",
        "base_amount": 59.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "Dropbox", "category": "Cloud",
        "description_template": "DROPBOX PLUS {contract_id}",
        "base_amount": 11.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "Google One", "category": "Cloud",
        "description_template": "GOOGLE ONE {contract_id}",
        "base_amount": 9.99, "variation": (0.0, 0.0)
    },

    # Fitness & Health (only one per category)
    {
        "partner": "ZEUS BODYPOWER", "category": "Gym",
        "description_template": "MITGLIEDSBEITRAG {partner}",
        "base_amount": 25.00, "variation": (0.0, 0.0)
    },
    {
        "partner": "DKV Deutsche Krankenversicherung", "category": "Health Insurance",
        "description_template": "GESUNDHEITSVERSICHERUNG {partner} {contract_id}",
        "base_amount": 89.50, "variation": (-5.0, 5.0)
    },

    # Banking & Financial Services (only one per category)
    {
        "partner": "ING DiBa", "category": "Banking",
        "description_template": "ING KONTOPREMIUM {contract_id}",
        "base_amount": 9.99, "variation": (-1.0, 1.0)
    },

    # Transportation (only one per category)
    {
        "partner": "Deutsche Bahn", "category": "Transportation",
        "description_template": "DB BAHNCARD {contract_id}",
        "base_amount": 59.00, "variation": (0.0, 0.0)
    },

    # Other Services (multiple allowed for different service types)
    {
        "partner": "HelloFresh", "category": "Food Delivery",
        "description_template": "HELLOFRESH BOX {contract_id}",
        "base_amount": 69.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "Bookbeat", "category": "Books",
        "description_template": "BOOKBEAT UNLIMITED {contract_id}",
        "base_amount": 9.99, "variation": (0.0, 0.0)
    },
    {
        "partner": "Audible", "category": "Audiobooks",
        "description_template": "AUDIBLE PLUS {contract_id}",
        "base_amount": 9.95, "variation": (0.0, 0.0)
    },
    {
        "partner": "IKEA Family", "category": "Shopping",
        "description_template": "IKEA FAMILY {contract_id}",
        "base_amount": 29.00, "variation": (0.0, 0.0)
    },
    {
        "partner": "Saturn Media Markt", "category": "Electronics",
        "description_template": "MEDIA MARKT PLUS {contract_id}",
        "base_amount": 9.99, "variation": (0.0, 0.0)
    }
]

# One-off Payments - Expanded from 5 to 35+ entries
ONE_OFF_PAYMENTS = [
    # Groceries
    {
        "partner_options": ["EDEKA", "REWE", "LIDL", "ALDI SUED", "KAUFLAND", "NETTO", "PENNY", "REAL", "HIT", "NORMA"],
        "category": "Groceries",
        "description_template": "KARTENZ./{date_short} {partner} {city}",
        "amount_range": (15.0, 150.0)
    },
    {
        "partner_options": ["DM Drogerie Markt", "ROSSMANN", "MUELLER", "BUDNIKOWSKY"],
        "category": "Drugstore",
        "description_template": "{partner} {city} {date_short}",
        "amount_range": (8.0, 75.0)
    },

    # Fuel
    {
        "partner_options": ["SHELL", "ARAL", "JET", "ESSO", "TOTAL", "AGIP", "OMV", "SB-TANK"],
        "category": "Fuel",
        "description_template": "KARTENZAHLUNG {partner} TANKSTELLE",
        "amount_range": (40.0, 90.0)
    },
    {
        "partner_options": ["TANKSTELLE", "FREIE TANKSTELLE", "AUTOMATIKTANKSTELLE"],
        "category": "Fuel",
        "description_template": "{partner} {city} KARTE",
        "amount_range": (35.0, 85.0)
    },

    # Online Shopping
    {
        "partner_options": ["AMAZON.DE", "ZALANDO", "EBAY", "OTTO", "ABOUTYOU", "ZARA", "H&M", "UNIQLO"],
        "category": "Shopping",
        "description_template": "{partner} MKTPLC EU {random_str}",
        "amount_range": (10.0, 250.0)
    },
    {
        "partner_options": ["CONRAD", "SATURN", "MEDIA MARKT", "EURONICS", "NOTEBOOKS BILLIGER"],
        "category": "Electronics",
        "description_template": "{partner} {city} {random_str}",
        "amount_range": (25.0, 500.0)
    },

    # Dining
    {
        "partner_options": ["BURGER KING", "MCDONALDS", "SUBWAY", "KFC", "PIZZA HUT", "DOMINOS"],
        "category": "Fast Food",
        "description_template": "{partner} {city}",
        "amount_range": (8.0, 45.0)
    },
    {
        "partner_options": ["STARBUCKS", "COSTA COFFEE", "TIM HORTONS", "BACKWERK"],
        "category": "Coffee",
        "description_template": "{partner} {city} FILIALE",
        "amount_range": (3.0, 15.0)
    },
    {
        "partner_options": ["ITALIANO", "PIZZERIA", "SUSHI", "ASIA IMBISS", "DOENER"],
        "category": "Restaurant",
        "description_template": "{partner} {city}",
        "amount_range": (12.0, 65.0)
    },

    # Transportation
    {
        "partner_options": ["DB BAHN", "DEUTSCHE BAHN", "Flixbus", "BlaBlaCar"],
        "category": "Transportation",
        "description_template": "{partner} TICKET {random_str}",
        "amount_range": (15.0, 120.0)
    },
    {
        "partner_options": ["UBER", "BOLT", "FREE NOW"],
        "category": "Ride Sharing",
        "description_template": "{partner} {city} {random_str}",
        "amount_range": (8.0, 35.0)
    },

    # Healthcare & Pharmacy
    {
        "partner_options": ["APOTHEKE", "LLOYDS APOTHEKE", "BARCELONA APOTHEKE"],
        "category": "Pharmacy",
        "description_template": "{partner} {city}",
        "amount_range": (5.0, 80.0)
    },
    {
        "partner_options": ["ARZT", "HAUSARZT", "FACHARZT", "ZAHNARZT"],
        "category": "Healthcare",
        "description_template": "{partner} {city} {random_str}",
        "amount_range": (20.0, 150.0)
    },

    # Entertainment & Leisure
    {
        "partner_options": ["CINEMAXX", "UCI", "CINESTAR", "KINO"],
        "category": "Cinema",
        "description_template": "{partner} {city} TICKET",
        "amount_range": (8.0, 25.0)
    },
    {
        "partner_options": ["BOWLING", "MINIGOLF", "ESCAPE ROOM", "TRAMPOLIN"],
        "category": "Entertainment",
        "description_template": "{partner} {city}",
        "amount_range": (10.0, 45.0)
    },

    # Services
    {
        "partner_options": ["PAYPAL", "SKRILL", "NETELLER"],
        "category": "Payment Service",
        "description_template": "{partner} {random_str}",
        "amount_range": (5.0, 100.0)
    },
    {
        "partner_options": ["DHL", "HERMES", "DPD", "GLS", "UPS"],
        "category": "Shipping",
        "description_template": "{partner} PAKET {random_str}",
        "amount_range": (3.0, 25.0)
    },
    {
        "partner_options": ["POST", "DEUTSCHE POST", "DPAG"],
        "category": "Postal",
        "description_template": "{partner} {random_str}",
        "amount_range": (1.0, 15.0)
    },

    # Government & Utilities
    {
        "partner_options": ["GEZ", "ARD ZDF", "RUNDFUNK"],
        "category": "Broadcasting Fee",
        "description_template": "{partner} BEITRAG {random_str}",
        "amount_range": (17.50, 17.50)
    },
    {
        "partner_options": ["FINANZAMT", "STEUERAMT"],
        "category": "Taxes",
        "description_template": "{partner} {city} {random_str}",
        "amount_range": (50.0, 500.0)
    },

    # Other
    {
        "partner_options": ["IKEA", "OBI", "HORNBACH", "BAUHAUS"],
        "category": "Home Improvement",
        "description_template": "{partner} {city} {random_str}",
        "amount_range": (20.0, 300.0)
    },
    {
        "partner_options": ["PARFUEMERIE", "DOUGLAS", "FLAIR"],
        "category": "Beauty",
        "description_template": "{partner} {city}",
        "amount_range": (15.0, 120.0)
    },
    {
        "partner_options": ["BOOKSTORE", "THALIA", "HUGENDUBEL"],
        "category": "Books",
        "description_template": "{partner} {city} {random_str}",
        "amount_range": (8.0, 60.0)
    }
]

# Additional configuration for enhanced realism
GERMAN_CITIES = [
    "Berlin", "Hamburg", "München", "Köln", "Frankfurt", "Stuttgart", "Düsseldorf",
    "Dortmund", "Essen", "Leipzig", "Bremen", "Dresden", "Hannover", "Nürnberg",
    "Duisburg", "Bochum", "Wuppertal", "Bielefeld", "Bonn", "Münster", "Karlsruhe",
    "Mannheim", "Augsburg", "Wiesbaden", "Gelsenkirchen", "Mönchengladbach",
    "Braunschweig", "Chemnitz", "Kiel", "Aachen", "Halle", "Magdeburg", "Freiburg",
    "Krefeld", "Lübeck", "Oberhausen", "Erfurt", "Mainz", "Rostock", "Kassel",
    "Hagen", "Hamm", "Saarbrücken", "Mühlheim", "Potsdam", "Ludwigshafen",
    "Oldenburg", "Leverkusen", "Osnabrück", "Solingen", "Heidelberg", "Herne",
    "Neuss", "Darmstadt", "Paderborn", "Regensburg", "Ingolstadt", "Würzburg",
    "Fürth", "Ulm", "Heilbronn", "Pforzheim", "Wolfsburg", "Göttingen", "Bottrop",
    "Reutlingen", "Koblenz", "Bremerhaven", "Bergisch Gladbach", "Jena", "Remscheid"
]

CONTRACT_ID_FORMATS = [
    "K-{digits}",
    "VN-{digits}",
    "POL-{digits}",
    "VERTRAG-{digits}",
    "ABO-{digits}",
    "{digits}",
    "ID-{digits}"
]
