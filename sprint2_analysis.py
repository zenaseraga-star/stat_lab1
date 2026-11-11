SEED = 20261109

def protein_summary(protein):
    return {
        "Всего значений": len(protein),
        "Положительных значений": int((protein > 0).sum()),
        "Нулей": int((protein == 0).sum()),
    }