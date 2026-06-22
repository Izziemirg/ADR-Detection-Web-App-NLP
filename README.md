## Adverse Drug Reaction (ADR) Detection Web Application

An interactive web application designed for clinicians to streamline Adverse Drug Reaction (ADR) extraction and risk assessment from unstructured patient narratives. This application couples a custom fine-tuned DeBERTa-v3-Base severity classification model with a pretrained Named Entity Recognition (NER) pipeline, topped with an integrated mathematical explainability layer.


Developed as part of the school project for my MSBA at UVA.

### Key Features

### Dual-Model NLP Pipeline 

Entity Extraction: A pretrained NER model isolates drug names, symptoms, and clinical markers from raw, unstructured text.

Severity Classification: A fine-tuned DeBERTa-v3-Base model categorizes the overall narrative into Mild or Severe risk thresholds.

XAI Explainability Visualization: Built-in dynamic visualization layer that maps token-level attribution. It highlights localized text based on its mathematical contribution to the final inference:

- Red Highlights: Heavy contribution toward a Severe ADR classification.
- Blue Highlights: Heavy contribution toward a Mild ADR classification.

Clinician-Centric Interface: Built with Gradio, providing medical professionals with instantaneous, zero-code risk assessments and readable visual auditing.
