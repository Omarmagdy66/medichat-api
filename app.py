from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# التحقق من وجود API key
if not os.getenv('GOOGLE_API_KEY'):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

try:
    # إعداد Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )
    
    # إعداد LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # تحميل أو إنشاء vector store
    persist_directory = "./vector_store"
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        # إنشاء vectorstore فارغ مؤقتاً
        vectorstore = Chroma(
            embedding_function=embeddings
        )
    
    # إنشاء prompt template
    prompt_template = """أنت مساعد طبي ذكي. استخدم المعلومات التالية للإجابة على السؤال.
    إذا لم تكن متأكداً من الإجابة، قل "لا أعرف".
    
    السياق: {context}
    
    السؤال: {question}
    
    الإجابة:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # إنشاء chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
except Exception as e:
    print(f"Error initializing models: {e}")
    qa_chain = None

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>Medical AI Assistant</title>
            <meta charset="utf-8">
            <style>
                body { 
                    font-family: Arial; 
                    max-width: 800px; 
                    margin: 0 auto; 
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 { color: #333; text-align: center; }
                .chat-container { 
                    border: 1px solid #ddd; 
                    padding: 20px; 
                    height: 400px; 
                    overflow-y: auto;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    background: #fafafa;
                }
                .input-group {
                    display: flex;
                    gap: 10px;
                }
                input { 
                    flex: 1;
                    padding: 12px; 
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 16px;
                }
                button { 
                    padding: 12px 30px; 
                    background: #007bff;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                button:hover { background: #0056b3; }
                .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
                .user-message { background: #e3f2fd; text-align: right; }
                .bot-message { background: #f1f1f1; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 المساعد الطبي الذكي</h1>
                <div class="chat-container" id="chat"></div>
                <div class="input-group">
                    <input type="text" id="message" placeholder="اكتب سؤالك الطبي هنا..." onkeypress="if(event.key==='Enter') sendMessage()">
                    <button onclick="sendMessage()">إرسال</button>
                </div>
            </div>
            
            <script>
                async function sendMessage() {
                    const input = document.getElementById('message');
                    const message = input.value.trim();
                    if (!message) return;
                    
                    // إضافة رسالة المستخدم
                    const chatDiv = document.getElementById('chat');
                    chatDiv.innerHTML += `<div class="message user-message"><strong>أنت:</strong> ${message}</div>`;
                    
                    input.value = '';
                    
                    try {
                        const response = await fetch('/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message: message})
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            chatDiv.innerHTML += `<div class="message bot-message"><strong>خطأ:</strong> ${data.error}</div>`;
                        } else {
                            chatDiv.innerHTML += `<div class="message bot-message"><strong>المساعد:</strong> ${data.response}</div>`;
                        }
                    } catch (error) {
                        chatDiv.innerHTML += `<div class="message bot-message"><strong>خطأ:</strong> حدث خطأ في الاتصال</div>`;
                    }
                    
                    chatDiv.scrollTop = chatDiv.scrollHeight;
                }
            </script>
        </body>
    </html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not qa_chain:
            return jsonify({'error': 'النموذج غير مهيأ بعد'}), 500
            
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'الرجاء إدخال سؤال'}), 400
        
        # الحصول على الإجابة
        response = qa_chain.run(user_message)
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': f'حدث خطأ: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': qa_chain is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
