#Import Libraries:
##Import the necessary libraries for loading and using the mBART model
from transformers import MBartTokenizer, MBartForConditionalGeneration

# Load mBART Tokenizer and Model:
##First, load the mBART tokenizer, which will help you preprocess the input text and convert it into tokens. Then, load the mBART model, 
##specifying the language variant you want to use
model_name = "facebook/mbart-large-cc25"
tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Encode Input Text:
##Tokenize and encode the input text using the mBART tokenizer. This will convert the text into input IDs and attention masks that can be 
##fed into the model.
source_text = " Há um vazamento de esgoto no parque da cidade. O odor é muito forte e está afetando a experiência dos visitantes."\
              " É importante resolver esse problema para garantir a saúde e o bem-estar de todos."\
              " Solicitamos à prefeitura que envie uma equipe de manutenção para corrigir o vazamento e realizar a limpeza necessária."
input_ids = tokenizer.encode(source_text, return_tensors="pt", add_special_tokens=True)


#----------------------------------TRANSLATION----------------------------------

# Generate Translation (or Other Text Generation Tasks):
##To generate translations or perform other text generation tasks, use the model.generate() method. 
##Specify the target language using the appropriate language code (e.g., "en_XX" for English, "fr_XX" for French).
target_language = "en_XX"  # English language code
translated_ids = model.generate(input_ids, target_language=target_language)
translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

# Decode the Output:
##Finally, decode the generated output IDs to get the human-readable translated text.
print("Translated Text: ", translated_text)



'''
#----------------------------------CLASSIFICATION----------------------------------

#Prepare Input Text and Labels:
##For zero-shot classification, you need to provide a prompt that includes the input text along with a template for the label. 
##The model will generate the text in the template with the most likely class, based on its pre-trained knowledge.
input_text = "The following text needs to be classified: "
candidate_labels = ["label_1", "label_2", "label_3"]  # Replace with your class labels

#Generate Classifications:
##Use the model.generate() method to generate the classifications for each label. 
##The mBART model will automatically fill in the template with the most probable label.
for label in candidate_labels:
    prompt = input_text + label
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    output_ids = model.generate(input_ids, decoder_start_token_id=model.config.eos_token_id)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Label: {label}, Classification: {output_text}")

#Interpret the Results:
##The model will generate text for each candidate label based on the input prompt. 
##The label with the highest probability in the generated text is the predicted class for the input text.
'''