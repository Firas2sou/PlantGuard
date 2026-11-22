#!/usr/bin/env python
"""Build French progress report 2, following the PlantVillage preparation report."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
BLUE = RGBColor(0x1F, 0x4E, 0x79)


def p(doc, text, italic=False, size=None):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return para


def h1(doc, text):
    doc.add_heading(text, level=1)


def bullets(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def table(doc, headers, rows, widths=None):
    result = doc.add_table(rows=1, cols=len(headers))
    result.style = "Table Grid"
    for i, label in enumerate(headers):
        run = result.rows[0].cells[i].paragraphs[0].add_run(label)
        run.bold = True
        run.font.size = Pt(9.5)
    for row in rows:
        cells = result.add_row().cells
        for i, value in enumerate(row):
            cells[i].paragraphs[0].add_run(str(value)).font.size = Pt(9.5)
    if widths:
        for i, width in enumerate(widths):
            for row in result.rows:
                row.cells[i].width = Inches(width)
    return result


def code(doc, text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    para.paragraph_format.space_after = Pt(6)


def main() -> None:
    doc = Document()
    doc.core_properties.title = "Rapport d'avancement 2 de PlantGuard"
    doc.core_properties.subject = "Intégration du modèle et préparation de la démonstration"

    title = doc.add_heading("Rapport d'avancement 2 de PlantGuard", level=0)
    for run in title.runs:
        run.font.color.rgb = BLUE
    p(doc, "Intégration du modèle et préparation de la démonstration", size=14)
    p(doc, "15 novembre 2026   |   Suite du rapport 1 consacré au jeu de données",
      italic=True, size=10)
    p(doc, "Ce document décrit les avancées réalisées après le premier rapport et l'état actuel de l'application.",
      size=9)

    h1(doc, "1. Continuité du projet")
    p(doc, "Le rapport 1 décrit l'acquisition, la préparation et l'audit du jeu PlantVillage. Il "
           "présente également les expériences d'entraînement et leurs résultats. Ces éléments "
           "restent le document de référence pour les données et la comparaison détaillée des "
           "modèles ; ils ne sont pas repris ici. Le présent rapport d'avancement porte sur la "
           "suite du travail : le choix du modèle principal, son intégration dans le service web "
           "et la préparation de la démonstration.")

    h1(doc, "2. Intégration du modèle principal")
    p(doc, "Le modèle MobileNetV3-Small ajusté finement (E2), décrit dans le rapport 1, a été "
           "exporté au format ONNX et relié au service FastAPI. Il atteint 99,33 % de précision "
           "top-1 sur le test PlantVillage rapporté précédemment. Ce chiffre explique le choix du "
           "modèle pour l'application ; il caractérise les images de ce jeu de test et ne prédit "
           "pas à lui seul les performances sur des photos réelles.")
    p(doc, "Au moment de l'analyse, l'image est convertie en RGB, redimensionnée avec une "
           "interpolation bilinéaire, recadrée au centre en 224 × 224 pixels et normalisée avec "
           "les statistiques ImageNet. Ce traitement est aligné sur celui de l'évaluation du "
           "modèle. ONNX Runtime exécute ensuite l'inférence sans charger PyTorch dans le serveur.")

    h1(doc, "3. Application web réalisée")
    p(doc, "L'application comporte un serveur FastAPI (src/serve/app.py) et une page web "
           "autonome (web/index.html). L'interface accepte une photo prise avec la caméra ou un "
           "fichier JPEG, PNG ou WebP. Elle affiche l'aperçu, les trois prédictions principales, "
           "leur confiance et les renseignements associés issus du catalogue des maladies.")
    p(doc, "La fiche maladie enrichit le service annoncé dans le rapport 1 : elle présente aussi "
           "les symptômes à observer, des mesures possibles et des pistes de prévention. Ces "
           "textes sont stockés dans knowledge/diseases.json et complètent la classification "
           "par des informations pratiques destinées à l'utilisateur.")
    p(doc, "La provenance des conseils n'est pas documentée dans le dépôt : les fiches ne "
           "contiennent ni liens ni références propres à chaque maladie. Il n'est donc pas "
           "possible d'attribuer ces formulations à une source précise à partir des fichiers "
           "actuels. Avant de présenter les mesures comme des recommandations validées, il faut "
           "les vérifier auprès de sources agronomiques reconnues et ajouter ces références.")
    table(doc, ["Point d'accès", "Fonction"], [
        ["GET /", "Ouvre la page de l'application"],
        ["GET /health", "Vérifie le chargement du modèle et des classes"],
        ["POST /predict", "Reçoit une image et renvoie les prédictions et leur fiche"],
    ], widths=[1.4, 4.3])
    p(doc, "Pour limiter les réponses incertaines, l'application signale un doute si la "
           "probabilité maximale est inférieure à 0,60 ou si l'écart avec la deuxième prédiction "
           "est inférieur à 0,15. Ce seuil est une règle simple ; il n'a pas fait l'objet d'une "
           "étude de calibration.")

    h1(doc, "4. Ajustements apportés à l'application")
    table(doc, ["Ajustement", "Résultat"], [
        ["Aperçu d'une image choisie", "L'image est dessinée dans l'aperçu avant l'analyse."],
        ["Contrôle de disponibilité", "Le point /health signale une erreur si le modèle ne se charge pas."],
        ["Limite de téléversement", "Les images de plus de 10 Mo sont refusées pendant la lecture."],
        ["Prétraitement", "L'interpolation bilinéaire correspond à l'évaluation du modèle."],
    ], widths=[2.0, 3.7])
    p(doc, "Ces ajustements rendent le parcours de démonstration plus cohérent et rendent les "
           "problèmes de modèle ou de fichier plus visibles pour l'utilisateur.")

    h1(doc, "5. État actuel et limites")
    p(doc, "Le parcours principal est implémenté : ouvrir la page, fournir une photo, puis lire "
           "les prédictions et les informations sur la classe retenue. En revanche, aucune "
           "évaluation systématique sur des photographies de terrain n'est rapportée. Les "
           "images d'entraînement sont prises dans des conditions contrôlées, et l'application "
           "ne reconnaît que les 38 classes apprises. Une réponse du modèle reste une aide au "
           "repérage, pas un diagnostic agronomique confirmé.")
    p(doc, "Avant la démonstration, il reste à vérifier le fonctionnement de la caméra et de "
           "l'envoi de fichier sur l'ordinateur utilisé, la présence du modèle ONNX et l'affichage "
           "correct d'un résultat incertain.")

    h1(doc, "6. Préparation de la démonstration")
    p(doc, "Le serveur se lance en local avec la commande suivante. Il faut ensuite ouvrir "
           "http://127.0.0.1:8010 dans un navigateur.")
    code(doc, ".venv\\Scripts\\python.exe -m uvicorn src.serve.app:app --host 127.0.0.1 --port 8010")
    bullets(doc, [
        "Choisir une photo nette d'une seule feuille, ou utiliser la caméra.",
        "Montrer l'aperçu, les trois classes proposées et leur niveau de confiance.",
        "Ouvrir la fiche pour présenter les symptômes, les mesures proposées et la prévention.",
        "Expliquer que le résultat dépend des classes PlantVillage et que les photos de terrain restent à évaluer.",
    ])

    h1(doc, "7. Conclusion de l'avancement")
    p(doc, "Depuis le rapport 1, le travail a progressé de l'évaluation des modèles à leur "
           "intégration dans une application utilisable. Le rapport 1 conserve les détails sur "
           "les données et les expériences ; ce rapport 2 rend compte de l'avancement applicatif "
           "et prépare la démonstration, prochaine étape du projet.")

    output = REP / "Rapport_Avancement_2_FR.docx"
    doc.save(output)
    print("saved", output)


if __name__ == "__main__":
    main()
