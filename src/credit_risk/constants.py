COLUMN_NAMES = {
    "laufkont": "checking_status", "laufzeit": "duration_months",
    "moral": "credit_history", "verw": "purpose", "hoehe": "credit_amount_dm",
    "sparkont": "savings", "beszeit": "employment_duration",
    "rate": "installment_rate", "famges": "personal_status_sex",
    "buerge": "other_debtors", "wohnzeit": "residence_duration",
    "verm": "property", "alter": "age_years",
    "weitkred": "other_installment_plans", "wohn": "housing",
    "bishkred": "number_credits", "beruf": "job",
    "pers": "people_liable", "telef": "telephone",
    "gastarb": "foreign_worker", "kredit": "credit_risk"
}

CATEGORY_LABELS = {
    "checking_status": {1: "No account", 2: "Below 0 DM", 3: "0–200 DM", 4: "200+ DM"},
    "credit_history": {0: "Past delay", 1: "Critical/other credit", 2: "No/prior paid", 3: "Current paid", 4: "All paid"},
    "purpose": {0: "Other", 1: "New car", 2: "Used car", 3: "Furniture", 4: "Radio/TV", 5: "Appliances", 6: "Repairs", 7: "Education", 8: "Vacation", 9: "Retraining", 10: "Business"},
    "savings": {1: "Unknown/no account", 2: "Below 100 DM", 3: "100–500 DM", 4: "500–1000 DM", 5: "1000+ DM"},
    "employment_duration": {1: "Unemployed", 2: "Below 1 year", 3: "1–4 years", 4: "4–7 years", 5: "7+ years"},
    "installment_rate": {1: "35%+", 2: "25–35%", 3: "20–25%", 4: "Below 20%"},
    "personal_status_sex": {1: "Male separated", 2: "Female non-single/male single", 3: "Male married/widowed", 4: "Female single"},
    "other_debtors": {1: "None", 2: "Co-applicant", 3: "Guarantor"},
    "residence_duration": {1: "Below 1 year", 2: "1–4 years", 3: "4–7 years", 4: "7+ years"},
    "property": {1: "Unknown/none", 2: "Car/other", 3: "Savings/insurance", 4: "Real estate"},
    "other_installment_plans": {1: "Bank", 2: "Stores", 3: "None"},
    "housing": {1: "Free", 2: "Rent", 3: "Own"},
    "number_credits": {1: "1", 2: "2–3", 3: "4–5", 4: "6+"},
    "job": {1: "Unemployed/nonresident", 2: "Unskilled", 3: "Skilled", 4: "Manager/self-employed"},
    "people_liable": {1: "3+", 2: "0–2"},
    "telephone": {1: "No", 2: "Yes"},
    "foreign_worker": {1: "Yes", 2: "No"}
}
