import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
global tokenizer, model, BASE
def get_ner(text, model_name='ai4bharat', metadata=None):
    global tokenizer, model, BASE
    if model_name == 'sagorsarker':
        tokenizer = AutoTokenizer.from_pretrained("sagorsarker/mbert-bengali-ner")
        model = AutoModelForTokenClassification.from_pretrained("sagorsarker/mbert-bengali-ner")
        BASE = 'LABEL_0'
    elif model_name == 'ai4bharat':
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
        model = AutoModelForTokenClassification.from_pretrained("ai4bharat/IndicNER")
        BASE = 'O'
    words = text.split(' ')
    chunk_size = 100
    ner_output = []
    for start in range(0, len(words), chunk_size):
        chunk_words = words[start:min(start + chunk_size, len(words))]
        tok_sentence = tokenizer(chunk_words, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**tok_sentence).logits.argmax(-1)
            predicted_tokens_classes = [
                model.config.id2label[t.item()] for t in logits[0]]
            predicted_labels = []
            word_ids = tok_sentence.word_ids(batch_index=0)
            previous_word_idx = None
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx == previous_word_idx:
                    pass
                else:
                    predicted_labels.append(predicted_tokens_classes[idx])
                    previous_word_idx = word_idx
            min_len = min(len(chunk_words), len(predicted_labels))
            for index in range(min_len):
                ner_output.append(
                    (chunk_words[index], predicted_labels[index]))
    if metadata is not None:
        metadata['tokenizer'] = tokenizer
        metadata['model'] = model
        metadata['base'] = BASE
    return ner_output

if __name__ == "__main__":
    result = get_ner("১০০ বা ২০০ টি নয়, এক মহিলার পেটের ভিতর থেকে বের করা হল হাজার খানেক পাথর। সংখ্যাটা এতটাই বেশি যে গুনে শেষ করতে পারছিলেন না চিকিৎসকরা। প্রায় কয়েকঘণ্টার রুদ্ধশ্বাস অস্ত্রোপচারে সফলভাবে এই পাথরগুলি বের করেন চিকিৎসকরা। টালিগঞ্জের হরিদেবপুর এলাকার এক ৪৫ বছরের মহিলার গলব্লাডারে এতো সংখ্যক পাথর দেখে রীতিমতো চমকে ওঠেন চিকিৎসকরা। জানা গিয়েছে, ওই মহিলার অস্ত্রোপচার করা হয় এম আর বাঙ্গুর হাসপাতালে। চিকিৎসক সূত্রের খবর, বিগত কয়েক মাস ধরে পেটের তীব্র যন্ত্রণায় কষ্ট পাচ্ছিলেন ওই মহিলা। প্রায় দুই মাস আগে তিনি বমি বমি ভাব এবং পেটের ব্যথা নিয়ে হাসপাতালে ভর্তি হন। প্রাথমিক পর্যবেক্ষণ ও পরীক্ষার পরে চিকিৎসকদের হাতে যে রিপোর্ট আসে তা দেখে তাঁরা কার্যত হতভম্ব হয়ে যান। রিপোর্টে দেখা যায়, রোগীর গলব্লাডারে প্রচুর সংখ্যক পাথর রয়েছে। এরপরই অস্ত্রোপচারের সিদ্ধান্ত নেওয়া হয়। সেইমতো ৩ জুন রোগীকে ভর্তি করা হয় হাসপাতালে। আর ৫ জুন দুপুরে শুরু হয় অপারেশন। প্রায় ঘণ্টাখানেকের চেষ্টায় সফলভাবে অস্ত্রোপচার সম্পন্ন করেন শল্য চিকিৎসক ডাঃ নিলয় নারায়ণ সরকার ও ডাঃ জয়দীপ রায়ের নেতৃত্বে একটি চিকিৎসক দল। সঙ্গে ছিলেন অ্যানাস্থেশিয়ার বিশেষজ্ঞ ডাঃ বিএন দাস ও ডাঃ অঙ্কিত পাঁজা। অপারেশনের জটিলতা সম্পর্কে ডাঃ নিলয় নারায়ণ সরকার জানান, গলব্লাডারে পাথর হওয়া নতুন কিছু নয়। কিন্তু, এত সংখ্যক পাথর সচরাচর দেখা যায় না। গোনাও যাচ্ছিল না এমন অবস্থা! তিনি আরও জানান, এই ধরনের ক্ষেত্রে অস্ত্রোপচারের পাশাপাশি ইনফেকশনের ঝুঁকি থাকে বলেও আশঙ্কা ছিল। চিকিৎসকদের মতে, সুপার ডাঃ শিশিরকুমার নস্করের পরামর্শ ও সহায়তায় এবং নার্স ও শিক্ষানবিশ চিকিৎসকদের সহযোগিতায় এই জটিল অস্ত্রোপচার সফলভাবে সম্পন্ন হয়েছে। বর্তমানে রোগী স্থিতিশীল রয়েছেন বলে হাসপাতাল সূত্রে জানা গিয়েছে। উল্লেখ্য, বছর খানেক আগে একইভাবে কলকাতার এক ব্যক্তির গলব্লাডার থেকে ১,০০০-র বেশি পাথর বের করা হয়েছিল। ওই ব্যক্তির অস্ত্রোপচার হয়েছিল হায়দরাবাদে।")
    with open("new.txt", "w", encoding="utf-8") as f:
        for word, label in result:
            f.write(f"{word}\t{label}\n")