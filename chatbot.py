import os
import sys
import json
import time  # <-- YENİ: Yeniden deneme için eklendi
from dotenv import load_dotenv
from flask import Flask, render_template_string, request, jsonify
from datetime import datetime

# LangChain ve Google Gemini kütüphanelerini içe aktar
try:
    from google import genai
    from google.genai.types import HarmCategory, HarmBlockThreshold
    # <-- YENİ: Google API hata yönetimi için eklendi
    from google.api_core import exceptions as google_exceptions 
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("HATA: Gerekli kütüphaneler eksik.")
    print("Lütfen terminalde bu komutu çalıştırın:")
    print("pip install flask google-genai pypdf langchain-community langchain-text-splitters langchain-huggingface sentence-transformers faiss-cpu python-dotenv google-api-core")
    sys.exit(1)

# --- KONFİGÜRASYON ---
load_dotenv()
app = Flask(__name__)

# Hız için Flash-8B modeli, daha detaylı yanıtlar için Pro'ya çevirebilirsiniz
GEMINI_MODEL = "gemini-2.0-flash-exp"  # En hızlı ve güncel model
VECTOR_DB_PATH = "faiss_index"
HOST_IP = "127.0.0.1"
PORT_NUMBER = 5000

PDF_FILES = [
    "psikoloji_sozlugu.pdf",
    "mindfulness_egzersizleri.pdf",
    "bdt_kilavuzu.pdf"
]

client = None
vector_db = None
conversation_history = []  # Sohbet geçmişi (son 6 mesaj)

def setup_vector_db():
    """PDF'leri yükler ve FAISS veritabanını oluşturur (opsiyonel)."""
    global client, vector_db

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY bulunamadı. .env dosyanızı kontrol edin.")
        return False

    try:
        client = genai.Client(api_key=api_key)
        print("✓ Gemini istemcisi hazır")
    except Exception as e:
        print(f"❌ Gemini başlatılamadı: {e}")
        return False

    # PDF yükleme (opsiyonel - varsa yükle, yoksa devam et)
    if os.path.exists(VECTOR_DB_PATH):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✓ Psikoloji bilgi bankası yüklendi")
            return True
        except Exception as e:
            print(f"⚠️ PDF veritabanı yüklenemedi (normal sohbet devam edecek): {e}")
            return True

    # Yeni veritabanı oluştur
    documents = []
    for file_name in PDF_FILES:
        if not os.path.exists(file_name):
            continue
        try:
            loader = PyPDFLoader(file_name)
            documents.extend(loader.load())
            print(f"  ✓ {file_name}")
        except Exception as e:
            print(f"  ⚠️ {file_name} yüklenemedi")

    if documents:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        vector_db = FAISS.from_documents(texts, embeddings)
        vector_db.save_local(VECTOR_DB_PATH)
        print(f"✓ {len(texts)} bilgi parçası indekslendi")
    
    return True

def get_context_if_relevant(query: str) -> tuple:
    """Sadece psikoloji soruları için PDF'lerden bağlam çeker."""
    psi_keywords = [
        "terapi", "psikoloji", "bdt", "mindfulness", "anksiyete", "depresyon",
        "stres", "kaygı", "farkındalık", "nefes", "meditasyon", "ego",
        "bilinçaltı", "travma", "obsesif", "panik", "fobi", "öfke"
    ]
    
    query_lower = query.lower()
    is_psi = any(kw in query_lower for kw in psi_keywords)
    
    if not is_psi or not vector_db:
        return "", ""
    
    try:
        docs = vector_db.similarity_search(query, k=3)
        if docs:
            context = "\n\n".join([d.page_content[:500] for d in docs])
            sources = list(set([f"📚 {os.path.basename(d.metadata.get('source', ''))}" for d in docs]))
            return context, "\n".join(sources[:2])
    except:
        pass
    
    return "", ""


# --- GÜNCELLENMİŞ FONKSİYON ---
def generate_response(user_message: str) -> str:
    """
    Geliştirilmiş, hızlı ve doğal yanıt üretir.
    Geçici API hatalarında (503) yeniden deneme mekanizması içerir.
    """
    global conversation_history
    
    if not client:
        return "Üzgünüm, şu anda bağlantım yok 😔"

    # --- 1. Hazırlık ---
    # Sohbet geçmişini güncelle
    conversation_history.append({"role": "user", "content": user_message})
    if len(conversation_history) > 6:
        conversation_history = conversation_history[-6:]

    # Psikoloji sorusu mu kontrol et
    context, sources = get_context_if_relevant(user_message)
    
    # System prompt
    system = """Sen Ayla, samimi ve zeki bir AI asistanısın. 
ÖZELLİKLERİN:
- Modern yapay zeka asistanları gibi doğal, akıcı konuşursun
- Kısa ve öz cevaplar verirsin (2-4 cümle)
- Emoji kullanabilirsin ama abartma
- "Ben bir AI'yım ama..." gibi klişe cümleler kurma
- Psikoloji konusunda uzman bilgin var
KURALLAR:
1. Her konuda rahat sohbet et (hava, spor, yemek, teknoloji...)
2. Psikoloji/BDT soruları için bilgi bankamı kullan
3. Kriz durumlarında (intihar, zarar verme) 112/155'i öner
4. Direkt cevap ver, fazla açıklama yapma"""

    # Context varsa ekle
    if context:
        system += f"\n\nBİLGİ BANKASI:\n{context[:1000]}"

    # Sohbet geçmişini hazırla
    messages = [{"role": "user", "parts": [{"text": system}]}]
    for msg in conversation_history[-4:]:
        role = "user" if msg["role"] == "user" else "model"
        messages.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    
    # --- 2. Yeniden Deneme Döngüsü ---
    
    maks_deneme = 3  # Toplam 3 kez deneyeceğiz
    temel_bekleme_suresi = 1 # 1 saniye ile başlayacak
    
    for deneme in range(maks_deneme):
        try:
            # --- 3. API Çağrısı ---
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=messages,
                config={
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 300,
                    "safety_settings": [
                        {"category": cat, "threshold": HarmBlockThreshold.BLOCK_NONE}
                        for cat in [
                            HarmCategory.HARM_CATEGORY_HARASSMENT,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
                        ]
                    ],
                }
            )
            
            # --- 4. Başarılı Yanıt İşleme ---
            if response.candidates and response.candidates[0].content.parts:
                answer = response.candidates[0].content.parts[0].text.strip()
                
                # Sohbet geçmişine ekle
                conversation_history.append({"role": "assistant", "content": answer})
                
                # Kaynak varsa ekle
                if sources and context:
                    answer += f"\n\n{sources}"
                
                return answer # Başarılı, cevabı döndür ve fonksiyondan çık
            else:
                # Model boş yanıt dönerse (örn. güvenlik filtresi)
                return "Hmm, şu an yanıt veremedim. Tekrar dener misin? 🤔"
                
        # --- 5. Hata Yönetimi ---
        
        # Sadece '503 Service Unavailable' (veya benzeri geçici) hataları yakala
        except google_exceptions.ServiceUnavailable as e:
            print(f"❌ Yanıt hatası (503): {e}")
            print(f"Uyarı: 503 Hizmet Kullanılamıyor. Deneme {deneme + 1}/{maks_deneme}.")
            
            # Eğer son deneme değilse, bekle
            if deneme < maks_deneme - 1:
                # Üstel geri çekilme: 1*2^0=1s, 1*2^1=2s...
                bekleme_suresi = temel_bekleme_suresi * (2 ** deneme)
                print(f"{bekleme_suresi} saniye bekleniyor...")
                time.sleep(bekleme_suresi)
            else:
                # Son denemeydi, artık hata ver
                print("Hata: Maksimum deneme sayısına ulaşıldı. Model yanıt vermiyor.")
                return "Bağlantı sorunu yaşıyorum, biraz sonra tekrar dene 😊"

        # Diğer tüm kalıcı hatalar (API anahtarı yanlışı, kütüphane sorunu vb.)
        except Exception as e:
            print(f"❌ Kalıcı yanıt hatası: {e}")
            # Bu hatalar için tekrar denemenin anlamı yok, doğrudan hata ver
            return "Baçağlantı sorunu yaşıyorum, biraz sonra tekrar dene 😊"

    # Eğer döngü bir şekilde biterse (normalde dönmemesi gerekir)
    return "Bağlantı sorunu yaşıyorum, biraz sonra tekrar dene 😊"

# --- MODERN WEB ARAYÜZÜ ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ayla AI - Sohbet Asistanı</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 10px;
        }
        
        #chat-container {
            width: 100%;
            max-width: 420px;
            height: 95vh;
            max-height: 800px;
            background: #fff;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: white;
            text-align: center;
            position: relative;
        }
        
        header h1 {
            font-size: 1.4em;
            font-weight: 600;
            margin-bottom: 4px;
        }
        
        header p {
            font-size: 0.85em;
            opacity: 0.9;
        }
        
        .status {
            position: absolute;
            top: 15px;
            right: 15px;
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        #chat-box {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8fafc;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .message {
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 85%;
            line-height: 1.5;
            animation: slideIn 0.3s ease;
            font-size: 0.95em;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .bot-message {
            align-self: flex-start;
            background: white;
            color: #1e293b;
            border: 1px solid #e2e8f0;
            border-bottom-left-radius: 4px;
            white-space: pre-wrap;
        }
        
        .bot-message.typing {
            background: #f1f5f9;
            color: #64748b;
        }
        
        .typing-indicator {
            display: inline-flex;
            gap: 4px;
            align-items: center;
        }
        
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #94a3b8;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        #input-area {
            display: flex;
            padding: 16px;
            background: white;
            border-top: 1px solid #e2e8f0;
            gap: 10px;
        }
        
        #user-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 24px;
            font-size: 0.95em;
            outline: none;
            transition: all 0.2s;
            font-family: inherit;
        }
        
        #user-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 24px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9em;
            transition: all 0.2s;
            white-space: nowrap;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102,126,234,0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        #chat-box::-webkit-scrollbar { width: 6px; }
        #chat-box::-webkit-scrollbar-track { background: #f1f5f9; }
        #chat-box::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        
        @media (max-width: 480px) {
            #chat-container { border-radius: 0; height: 100vh; max-height: none; }
            header h1 { font-size: 1.2em; }
        }
    </style>
</head>
<body>
    <div id="chat-container">
        <header>
            <div class="status"></div>
            <h1>💜 Ayla AI</h1>
            <p>Sohbet & Psikoloji Asistanı</p>
        </header>
        <div id="chat-box">
            <div class="message bot-message">Merhaba! Ben Ayla 😊
Benimle her şey hakkında konuşabilirsin. Psikoloji, BDT ve mindfulness konularında da özel bilgim var.
Hadi sohbete başlayalım!</div>
        </div>
        <div id="input-area">
            <input type="text" id="user-input" placeholder="Mesajını yaz..." autocomplete="off">
            <button onclick="sendMessage()" id="send-btn">Gönder</button>
        </div>
    </div>
    <script>
        let isWaiting = false;
        function sendMessage() {
            if (isWaiting) return;
            
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            const chatBox = document.getElementById('chat-box');
            const sendBtn = document.getElementById('send-btn');
            
            // Kullanıcı mesajı
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-message';
            userDiv.textContent = message;
            chatBox.appendChild(userDiv);
            
            // Typing indicator
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot-message typing';
            typingDiv.id = 'typing-msg';
            typingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
            chatBox.appendChild(typingDiv);
            
            chatBox.scrollTop = chatBox.scrollHeight;
            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;
            isWaiting = true;
            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById('typing-msg')?.remove();
                
                const botDiv = document.createElement('div');
                botDiv.className = 'message bot-message';
                botDiv.textContent = data.response;
                chatBox.appendChild(botDiv);
                
                chatBox.scrollTop = chatBox.scrollHeight;
            })
            .catch(error => {
                document.getElementById('typing-msg')?.remove();
                
                const errorDiv = document.createElement('div');
                errorDiv.className = 'message bot-message';
                errorDiv.textContent = 'Bağlantı hatası 😔 Tekrar dener misin?';
                chatBox.appendChild(errorDiv);
            })
            .finally(() => {
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
                isWaiting = false;
            });
        }
        document.addEventListener('DOMContentLoaded', () => {
            const input = document.getElementById('user-input');
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            input.focus();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    return jsonify({
        "status": "online",
        "model": GEMINI_MODEL,
        "rag_enabled": vector_db is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'response': 'Bir şeyler yaz bakalım 😊'})

        print(f"💬 [{datetime.now().strftime('%H:%M:%S')}] Kullanıcı: {user_message}")
        
        response = generate_response(user_message)
        
        print(f"🤖 [{datetime.now().strftime('%H:%M:%S')}] Ayla: {response[:80]}...")
        
        return jsonify({'response': response})

    except Exception as e:
        print(f"❌ Chat hatası: {e}")
        return jsonify({'response': 'Bir sorun oluştu, tekrar dener misin? 😊'})

# ... (chat_endpoint fonksiyonunun sonu burada olmalı) ...

# --- UYGULAMAYI BAŞLATMA ---
# Gunicorn'un ve lokal çalıştırmanın 'client' ve 'vector_db' değişkenlerini
# başlatabilmesi için bu bloğu '__main__' dışına taşıyoruz.
print("=" * 70)
print("💜 AYLA AI - GELİŞMİŞ SOHBET ASİSTANI")
print("=" * 70)

# setup_vector_db()'yi global scope'ta (en dış katmanda) çağır
if setup_vector_db():
    mode = "Tam Özellikli" if vector_db else "Sohbet Modu"
    print(f"✓ Mod: {mode}")
    print(f"✓ Model: {GEMINI_MODEL}")
    # Not: HF için IP/Port yazdırmak çok önemli değil ama zararı da yok.
    print(f"✓ Adres (Lokal): http://{HOST_IP}:{PORT_NUMBER}") 
    print("=" * 70)
else:
    # Bu hata mesajı artık hem lokalde hem de HF'de görünür olacak
    print("❌ Başlatma başarısız. .env dosyanızı (veya HF Secrets) kontrol edin.")

# Lokal'de 'python chatbot.py' komutuyla çalıştırmak için bu blok kalmalı
# Gunicorn bu bloğu GÖRMEYECEK, bu normal.
if __name__ == '__main__':
    print("🌐 Tarayıcınızda yukarıdaki adresi açın!")
    print("⌨️  Ctrl+C ile durdurun")
    print("=" * 70)
    app.run(host=HOST_IP, port=PORT_NUMBER, debug=False, use_reloader=False)
