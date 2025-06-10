from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)
CORS(app)

# إعداد Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# إعداد RAG
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv('GOOGLE_API_KEY')
)

# تحميل vector store
vectorstore = Chroma(
    persist_directory="./vector_store",
    embedding_function=embeddings
)

# إنشاء chain
qa_chain = RetrievalQA.from_chain_type(
    llm=genai.GenerativeModel('gemini-pro'),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>Medical Chatbot</title>
            <style>
                body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
                .chat-container { border: 1px solid #ccc; padding: 20px; height: 400px; overflow-y: auto; }
                input { width: 70%; padding: 10px; }
                button { padding: 10px 20px; }
            </style>
        </head>
        <body>
            <h1>Medical AI Assistant</h1>
            <div class="chat-container" id="chat"></div>
            <input type="text" id="message" placeholder="اكتب سؤالك الطبي...">
            <button onclick="sendMessage()">إرسال</button>
            
            <script>
                async function sendMessage() {
                    const input = document.getElementById('message');
                    const message = input.value;
                    
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    });
                    
                    const data = await response.json();
                    document.getElementById('chat').innerHTML += 
                        `<p><strong>أنت:</strong> ${message}</p>
                         <p><strong>البوت:</strong> ${data.response}</p>`;
                    
                    input.value = '';
                }
            </script>
        </body>
    </html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data['message']
        
        # الحصول على الإجابة من RAG
        response = qa_chain.run(user_message)
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
