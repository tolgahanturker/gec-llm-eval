import ast

def process_gleu_scores(file_path):
    try:
        # Dosyayı oku
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Metin formatındaki listeyi gerçek Python listesine dönüştür
        # ast.literal_eval, güvenli bir şekilde string'i objeye çevirir
        data = ast.literal_eval(content)
        
        print(f"{'ID':<10} | {'GLEU Skoru':<12} | {'Standart Sapma':<15} | {'Güven Aralığı'}")
        print("-" * 65)
        
        # Her bir tripleti (üçlüyü) oku ve yazdır
        for i, triplet in enumerate(data, start=1):
            # triplet içeriği: [ortalama_skor, std_sapma, (alt_sinir, ust_sinir)]
            score = triplet[0]
            std_dev = triplet[1]
            conf_int = triplet[2]
            
            print(f"Cümle {i:<3} | {score:<12} | {std_dev:<15} | {conf_int}")
            
    except FileNotFoundError:
        print(f"Hata: {file_path} dosyası bulunamadı.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    process_gleu_scores('sent_level_gleu_claude_neutral.txt')