from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

sequence_to_classify = "Há barulho na rua 24 de Agosto. Podem resolver esta situação?."

candidate_labels = ["Public Safety", "Infrastructure Issues", "Environmental Concerns", "Traffic and Transportation", "Noise and Nuisance", "Public Health", "Parks and Recreation", "Zoning and Land Use", "Community Events and Programs", "Civic Services"]

output = classifier(sequence_to_classify, candidate_labels, multi_label=False)

print(output)