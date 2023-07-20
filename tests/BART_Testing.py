from transformers import pipeline

labels = ["Public Safety", "Infrastructure Issues", "Environmental Concerns", "Traffic and Transportation", "Noise and Nuisance", "Public Health", "Parks and Recreation", "Zoning and Land Use", "Community Events and Programs", "Civic Services"]

lst=["Multiple large potholes on Main Street between 5th Avenue and 10th Avenue. These potholes pose a hazard to vehicles and can potentially cause damage. Urgent repairs are needed to ensure the safety and smooth flow of traffic in the area.",
    "Há um vazamento de esgoto no parque da cidade. O odor é muito forte e está afetando a experiência dos visitantes. É importante resolver esse problema para garantir a saúde e o bem-estar de todos. Solicitamos à prefeitura que envie uma equipe de manutenção para corrigir o vazamento e realizar a limpeza necessária.",
    " I would like to report an issue regarding the excessive noise coming from a construction site located on Elm Street. The construction work starts early in the morning and continues throughout the day, causing a significant disturbance to the surrounding residents. The noise levels are unbearable, making it challenging to concentrate, work from home, or even have a peaceful environment for our families. We kindly request the relevant authorities to address this issue and enforce stricter regulations on noise control for construction activities in residential areas."]

hypothesis_template = 'This text is about {}.'

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
for word in lst:
    sequence = word

    prediction = classifier(sequence, labels, hypothesis_template=hypothesis_template, multi_label=True)

    print(prediction)